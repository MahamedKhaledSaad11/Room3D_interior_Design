from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pytest

from room3d_worker.common.manifest import SceneManifest
from room3d_worker.pipeline.s05_track import (
    ManualOverride,
    PromptPoint,
    WarmSamSession,
    _Sam2Image,
    _Sam2Video,
    _to_cpu_if_tensor,
    _to_device_if_tensor,
    apply_manual_mask_overrides,
    filter_neighbor_iou,
    lift_and_reproject_masks,
    load_binary_mask_png,
    load_sam2_backends,
    normalize_binary_mask,
    project_3dgs_mask_hints,
    run_tracking_pipeline,
    save_binary_mask_png,
    score_reid_crops,
)


def _save_rgb(path: Path, width: int, height: int, box: tuple[int, int, int, int] | None) -> None:
    arr = np.full((height, width, 3), 40, dtype=np.uint8)
    if box is not None:
        x0, y0, x1, y1 = box
        arr[y0:y1, x0:x1] = (220, 80, 40)
    Image.fromarray(arr).save(path, format="PNG")


def _manifest(poses: list[np.ndarray], width: int = 160, height: int = 128) -> SceneManifest:
    return SceneManifest.model_validate({
        "schema_version": "1.0.0",
        "project_id": "proj",
        "version_id": "v1",
        "coord_convention": "opencv_colmap",
        "T_viewer_from_colmap": np.eye(4).tolist(),
        "camera_model": "PINHOLE",
        "width": width,
        "height": height,
        "intrinsics": {"fx": 160.0, "fy": 160.0, "cx": width / 2, "cy": height / 2},
        "test_every": 0,
        "frames": [
            {
                "frame_idx": idx,
                "orig_video_frame": idx,
                "timestamp_sec": float(idx) * 0.5,
                "image_relpath": f"images/{idx:05d}.png",
                "colmap_image_id": idx + 1,
                "cam_from_world": pose.tolist(),
            }
            for idx, pose in enumerate(poses)
        ],
        "artifacts": {"sparse_dir": "sparse/0", "spz_relpath": "scene.spz", "sparse_points_relpath": "sparse_points.json"},
    })


class _FakeInteractiveSam:
    def __init__(self, expected_hw: tuple[int, int]) -> None:
        self.encode_calls = 0
        self.expected_hw = expected_hw

    def encode_image(self, image_path: Path) -> Path:
        self.encode_calls += 1
        return image_path

    def predict(self, embedding: Path, points: list[PromptPoint]) -> np.ndarray:
        _ = embedding
        h, w = self.expected_hw
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = int(points[0].x), int(points[0].y)
        mask[max(0, cy - 12) : min(h, cy + 12), max(0, cx - 12) : min(w, cx + 12)] = 255
        return mask


class _FakeVideoSam:
    def __init__(self, masks_by_idx: dict[int, np.ndarray]) -> None:
        self.masks_by_idx = masks_by_idx
        self.last_device: str | None = None

    def propagate(
        self,
        image_paths: dict[int, Path],
        ref_frame_idx: int,
        seed_mask: np.ndarray,
        inference_state_device: str,
    ) -> dict[int, np.ndarray]:
        _ = (image_paths, ref_frame_idx, seed_mask)
        self.last_device = inference_state_device
        return self.masks_by_idx


def test_warm_sam_session_caches_with_lru_and_emits_uint8_masks(tmp_path: Path) -> None:
    p0, p1, p2 = tmp_path / "f0.png", tmp_path / "f1.png", tmp_path / "f2.png"
    for p in (p0, p1, p2):
        _save_rgb(p, 160, 128, (60, 48, 100, 80))
    fake = _FakeInteractiveSam((128, 160))
    session = WarmSamSession(fake, max_cache_items=2)

    m1, _ = session.segment_clicks(p0, [PromptPoint(80, 64, 1)], (128, 160))
    m2, _ = session.segment_clicks(p0, [PromptPoint(82, 64, 1)], (128, 160))
    assert fake.encode_calls == 1
    assert m1.dtype == np.uint8 and set(np.unique(m1)).issubset({0, 255})
    assert m2[64, 82] == 255

    # Evicts p0 after adding p1 and p2 because max_cache_items == 2.
    session.segment_clicks(p1, [PromptPoint(80, 64, 1)], (128, 160))
    session.segment_clicks(p2, [PromptPoint(80, 64, 1)], (128, 160))
    session.segment_clicks(p0, [PromptPoint(80, 64, 1)], (128, 160))
    assert fake.encode_calls == 4

    out_png = save_binary_mask_png(m1, tmp_path / "mask.png", (128, 160))
    loaded = load_binary_mask_png(out_png, (128, 160))
    assert np.array_equal(loaded, m1)

    # Logits float normalization handles threshold=0.0 properly.
    logits = np.full((128, 160, 1), -2.0, dtype=np.float32)
    logits[10:20, 10:20, 0] = 2.5
    norm_logits = normalize_binary_mask(logits, (128, 160))
    assert norm_logits[15, 15] == 255 and norm_logits[0, 0] == 0


def test_five_tier_tracking_handles_exit_reentry_iou_drop_and_overrides(tmp_path: Path) -> None:
    pose0 = np.eye(4)
    pose_oof = np.array([[1, 0, 0, -25.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    pose_back = np.eye(4)
    manifest = _manifest([pose0, pose_oof, pose_back])
    hw = (128, 160)

    imgs = {0: tmp_path / "0.png", 1: tmp_path / "1.png", 2: tmp_path / "2.png"}
    for idx, p in imgs.items():
        _save_rgb(p, 160, 128, None if idx == 1 else (68, 52, 92, 76))
    depth_dir = tmp_path / "depths"
    depth_dir.mkdir()
    for idx in imgs:
        np.save(depth_dir / f"{idx:06d}.npy", np.full(hw, 3.0, dtype=np.float32))

    m0 = np.zeros(hw, dtype=np.uint8)
    m0[52:76, 68:92] = 255
    false_pos = np.zeros(hw, dtype=np.uint8)
    false_pos[10:30, 10:30] = 255
    video_backend = _FakeVideoSam({0: m0, 1: false_pos, 2: np.zeros(hw, dtype=np.uint8)})
    session = WarmSamSession(_FakeInteractiveSam(hw))

    manual_path = tmp_path / "manual_2.png"
    save_binary_mask_png(m0, manual_path, hw)
    saved_paths, stats = run_tracking_pipeline(
        manifest=manifest,
        image_paths=imgs,
        ref_frame_idx=0,
        points=[PromptPoint(80, 64, 1)],
        sam_session=session,
        video_backend=video_backend,
        output_dir=tmp_path / "track_out",
        depth_dir=depth_dir,
        overrides=[ManualOverride(frame_idx=2, mask_path=manual_path)],
    )
    assert video_backend.last_device == "cpu"
    assert stats["visibility"][1] == "out_of_frustum"
    assert 2 in stats["recovered_reentries"]
    assert np.all(load_binary_mask_png(saved_paths[1], hw) == 0)
    assert np.array_equal(load_binary_mask_png(saved_paths[2], hw), m0)
    assert json.loads((tmp_path / "track_out" / "tracking_report.json").read_text())["total_frames"] == 3


def test_neighbor_iou_and_override_skip_frame_enforcement(tmp_path: Path) -> None:
    hw = (64, 64)
    m_good = np.zeros(hw, dtype=np.uint8)
    m_good[20:40, 20:40] = 255
    m_drift = np.zeros(hw, dtype=np.uint8)
    m_drift[0:10, 0:10] = 255
    _, flagged = filter_neighbor_iou({0: m_good, 1: m_drift, 2: m_good}, ref_frame_idx=0, min_iou=0.4)
    assert 1 in flagged

    out_masks, skipped = apply_manual_mask_overrides({0: m_good, 1: m_good}, [ManualOverride(1, skip_frame=True)], hw)
    assert 1 in skipped and np.all(out_masks[1] == 0)

    # Out-of-frustum check allows partial foreground object without false centroid drop when depth is present.
    manifest = _manifest([np.eye(4)], 64, 64)
    partial_mask = np.zeros(hw, dtype=np.uint8)
    partial_mask[30:34, 0:6] = 255
    session = WarmSamSession(_FakeInteractiveSam(hw))
    depth_dir = tmp_path / "partial_depth"
    depth_dir.mkdir()
    np.save(depth_dir / "000000.npy", np.full(hw, 2.5, dtype=np.float32))
    _, vis, _ = lift_and_reproject_masks({0: partial_mask}, manifest, {0: tmp_path / "x.png"}, depth_dir, 0, session)
    assert vis[0] == "visible"


def test_reid_rejects_background_or_wrong_hue_and_projects_3dgs_hint(tmp_path: Path) -> None:
    p0, p1 = tmp_path / "f0.png", tmp_path / "f1.png"
    _save_rgb(p0, 64, 64, (20, 20, 44, 44))
    arr_diff = np.full((64, 64, 3), 240, dtype=np.uint8)
    arr_diff[20:44, 20:44] = (20, 40, 230)
    Image.fromarray(arr_diff).save(p1, format="PNG")

    m = np.zeros((64, 64), dtype=np.uint8)
    m[20:44, 20:44] = 255
    scores, rejected = score_reid_crops({0: p0, 1: p1}, {0: m, 1: m}, ref_frame_idx=0, min_similarity=0.7)
    assert scores[1] < 0.7 and 1 in rejected

    manifest = _manifest([np.eye(4)], 64, 64)
    hint = project_3dgs_mask_hints(np.array([[0.0, 0.0, 2.0]]), manifest.frames[0], manifest.intrinsics.model_dump(), (64, 64))
    assert hint[32, 32] == 255


def test_load_sam2_backends_resolves_single_gpu_ordinal(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_devices: list[str] = []

    def _build_sam2(_cfg: str, _ckpt: str, device: str) -> Any:
        seen_devices.append(device)
        return SimpleNamespace(device=device)

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1))
    fake_build = SimpleNamespace(build_sam2=_build_sam2, build_sam2_video_predictor=_build_sam2)
    fake_pred = SimpleNamespace(SAM2ImagePredictor=lambda model: model)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "sam2.build_sam", fake_build)
    monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", fake_pred)

    load_sam2_backends("cfg.yaml", "sam2.pt", device="cuda:1")
    assert seen_devices == ["cuda:0", "cuda:0"]


class _DeviceTensor:
    def __init__(self, data: np.ndarray, device: str) -> None:
        self.data = data
        self.device = device

    def detach(self) -> _DeviceTensor:
        return self

    def cpu(self) -> _DeviceTensor:
        return _DeviceTensor(self.data, "cpu")

    def to(self, device: str) -> _DeviceTensor:
        return _DeviceTensor(self.data, str(device))

    def numpy(self) -> np.ndarray:
        if self.device != "cpu":
            raise TypeError(f"can't convert {self.device} device type tensor to numpy")
        return self.data

    def __getitem__(self, item: Any) -> _DeviceTensor:
        return _DeviceTensor(self.data[item], self.device)

    def __gt__(self, other: float) -> _DeviceTensor:
        return _DeviceTensor(self.data > other, self.device)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        if self.device != "cpu":
            raise TypeError(f"can't convert {self.device} device type tensor to numpy")
        return np.asarray(self.data, dtype=dtype)


def test_nested_tensor_cpu_cache_and_device_restoration(tmp_path: Path) -> None:
    nested = {
        "image_embed": _DeviceTensor(np.ones((1, 2), dtype=np.float32), "cuda:1"),
        "high_res_feats": [
            _DeviceTensor(np.ones((2, 2), dtype=np.float32), "cuda:1"),
            (_DeviceTensor(np.ones((2, 2), dtype=np.float32), "cuda:1"), "keep_str"),
        ],
    }
    cpu_nested = _to_cpu_if_tensor(nested)
    assert cpu_nested["image_embed"].device == "cpu"
    assert cpu_nested["high_res_feats"][0].device == "cpu"
    assert cpu_nested["high_res_feats"][1][0].device == "cpu"
    assert cpu_nested["high_res_feats"][1][1] == "keep_str"

    gpu_nested = _to_device_if_tensor(cpu_nested, "cuda:1")
    assert gpu_nested["image_embed"].device == "cuda:1"
    assert gpu_nested["high_res_feats"][1][0].device == "cuda:1"

    img_path = tmp_path / "ref.png"
    _save_rgb(img_path, 64, 64, (16, 16, 48, 48))

    class _FakeSam2Predictor:
        def __init__(self) -> None:
            self.device = "cuda:1"
            self._features: Any = None
            self._orig_hw: tuple[int, int] | None = None
            self._is_image_set = False
            self.set_image_calls = 0

        def set_image(self, _rgb: np.ndarray) -> None:
            self.set_image_calls += 1
            self._features = {
                "image_embed": _DeviceTensor(np.ones((1, 2), dtype=np.float32), self.device),
                "high_res_feats": [_DeviceTensor(np.ones((2, 2), dtype=np.float32), self.device)],
            }
            self._orig_hw = (64, 64)

        def predict(self, point_coords: np.ndarray, point_labels: np.ndarray, multimask_output: bool) -> tuple[np.ndarray, None, None]:
            _ = (point_coords, point_labels, multimask_output)
            assert self._features["image_embed"].device == "cuda:1"
            assert self._features["high_res_feats"][0].device == "cuda:1"
            out = np.zeros((1, 64, 64), dtype=bool)
            out[0, 16:48, 16:48] = True
            return out, None, None

    pred = _FakeSam2Predictor()
    session = WarmSamSession(_Sam2Image(pred), max_cache_items=2)
    m1, _ = session.segment_clicks(img_path, [PromptPoint(32, 32, 1)], (64, 64))
    m2, _ = session.segment_clicks(img_path, [PromptPoint(34, 34, 1)], (64, 64))
    assert pred.set_image_calls == 1
    assert m1[32, 32] == 255 and m2[32, 32] == 255


def test_sam2_video_bidirectional_propagation_and_cuda_logits(tmp_path: Path) -> None:
    imgs = {}
    for idx in range(4):
        p = tmp_path / f"v_{idx}.png"
        _save_rgb(p, 32, 32, (8, 8, 24, 24))
        imgs[idx] = p
    seed = np.zeros((32, 32), dtype=np.uint8)
    seed[8:24, 8:24] = 255

    class _BidirPredictor:
        def __init__(self) -> None:
            self.calls: list[tuple[int | None, bool]] = []

        def init_state(self, video_path: str, offload_video_to_cpu: bool, offload_state_to_cpu: bool) -> dict[str, Any]:
            return {"video_path": video_path, "cpu": offload_video_to_cpu and offload_state_to_cpu}

        def add_new_mask(self, state: dict[str, Any], frame_idx: int, obj_id: int, mask: np.ndarray) -> None:
            state["ref"] = (frame_idx, obj_id, mask)

        def propagate_in_video(self, state: dict[str, Any], start_frame_idx: int | None = None, reverse: bool = False):
            _ = state
            self.calls.append((start_frame_idx, reverse))
            positions = range(start_frame_idx or 0, -1, -1) if reverse else range(start_frame_idx or 0, 4)
            for pos in positions:
                arr = np.full((1, 32, 32), -2.0, dtype=np.float32)
                arr[0, 8:24, 8:24] = 3.0
                yield pos, [1], _DeviceTensor(arr, "cuda:1")

    pred = _BidirPredictor()
    out = _Sam2Video(pred).propagate(imgs, ref_frame_idx=2, seed_mask=seed, inference_state_device="cpu")
    assert pred.calls == [(2, False), (2, True)]
    assert sorted(out) == [0, 1, 2, 3]
    assert all(out[i][16, 16] == 255 and out[i][0, 0] == 0 for i in range(4))

    class _LegacyMockPredictor:
        def init_state(self, **_kwargs: Any) -> dict[str, Any]:
            return {}

        def add_new_mask(self, _state: dict[str, Any], **_kwargs: Any) -> None:
            return None

        def propagate_in_video(self, state: dict[str, Any]):
            _ = state
            for pos in range(4):
                arr = np.full((1, 32, 32), 1.0, dtype=np.float32)
                yield pos, [1], arr

    legacy_out = _Sam2Video(_LegacyMockPredictor()).propagate(imgs, ref_frame_idx=2, seed_mask=seed, inference_state_device="cpu")
    assert sorted(legacy_out) == [0, 1, 2, 3]


def test_lift_and_reproject_preserves_vos_when_depth_missing(tmp_path: Path) -> None:
    pose_oof = np.array([[1, 0, 0, -50.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    manifest = _manifest([np.eye(4), pose_oof, pose_oof], width=64, height=64)
    hw = (64, 64)
    imgs = {i: tmp_path / f"d_{i}.png" for i in range(3)}
    for p in imgs.values():
        _save_rgb(p, 64, 64, (20, 20, 44, 44))
    vos_mask = np.zeros(hw, dtype=np.uint8)
    vos_mask[20:44, 20:44] = 255
    masks = {0: vos_mask.copy(), 1: vos_mask.copy(), 2: vos_mask.copy()}
    session = WarmSamSession(_FakeInteractiveSam(hw))

    # No depth maps at all: trust VOS masks directly without inventing constant depth.
    kept, vis_none, recovered_none = lift_and_reproject_masks(masks, manifest, imgs, None, 0, session)
    assert all(vis_none[i] == "no_depth" for i in range(3))
    assert recovered_none == set()
    assert all(np.array_equal(kept[i], vos_mask) for i in range(3))

    # Reference depth + frame 2 depth present, frame 1 depth missing: preserve frame 1 VOS, filter frame 2 OOF.
    depth_dir = tmp_path / "partial_depths"
    depth_dir.mkdir()
    np.save(depth_dir / "000000.npy", np.full(hw, 2.5, dtype=np.float32))
    np.save(depth_dir / "000002.npy", np.full(hw, 2.5, dtype=np.float32))
    partial_out, vis_partial, _ = lift_and_reproject_masks(masks, manifest, imgs, depth_dir, 0, session)
    assert vis_partial[0] == "visible" and np.array_equal(partial_out[0], vos_mask)
    assert vis_partial[1] == "no_depth" and np.array_equal(partial_out[1], vos_mask)
    assert vis_partial[2] == "out_of_frustum" and np.all(partial_out[2] == 0)


def test_load_depth_map_five_digit_and_behind_camera_projection(tmp_path: Path) -> None:
    # Verify 5-digit depth maps from s03_train, 6-digit, and depth_00002.npy naming, plus behind-camera GeometryError guard.
    pose_front = np.eye(4, dtype=np.float64)
    pose_behind = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -10.0], [0, 0, 0, 1]], dtype=np.float64)
    manifest = _manifest([pose_front, pose_behind], width=64, height=64)
    hw = (64, 64)
    imgs = {i: tmp_path / f"b_{i}.png" for i in range(2)}
    for p in imgs.values():
        _save_rgb(p, 64, 64, (20, 20, 44, 44))
    vos_mask = np.zeros(hw, dtype=np.uint8)
    vos_mask[20:44, 20:44] = 255
    masks = {0: vos_mask.copy(), 1: vos_mask.copy()}
    session = WarmSamSession(_FakeInteractiveSam(hw))

    depth_dir = tmp_path / "depths_05d"
    depth_dir.mkdir()
    np.save(depth_dir / "00000.npy", np.full(hw, 2.5, dtype=np.float32))
    np.save(depth_dir / "depth_00001.npy", np.full(hw, 2.5, dtype=np.float32))

    kept, vis, _ = lift_and_reproject_masks(masks, manifest, imgs, depth_dir, 0, session)
    assert vis[0] == "visible" and np.array_equal(kept[0], vos_mask)
    assert vis[1] == "out_of_frustum" and np.all(kept[1] == 0)

    hint = project_3dgs_mask_hints(np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 3.0]]), manifest.frames[0], manifest.intrinsics.model_dump(), (64, 64))
    assert hint.shape == hw and np.any(hint == 255)
    unreg_frame = manifest.frames[0].model_copy(update={"cam_from_world": None})
    assert np.all(project_3dgs_mask_hints(np.array([[0.0, 0.0, 3.0]]), unreg_frame, manifest.intrinsics.model_dump(), hw) == 0)
