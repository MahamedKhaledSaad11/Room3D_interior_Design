from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from room3d_worker.common.coords import (
    Sim3Transform,
    project_world_to_image,
    transform_cam_from_world_sim3,
    umeyama_sim3,
)
from room3d_worker.common.errors import GeometryError, StageExecutionError
from room3d_worker.common.gaussian_state import GaussianState
from room3d_worker.common.manifest import SceneManifest
from room3d_worker.pipeline.s02_poses import (
    BackendOutput,
    ColmapFailureError,
    PoseRequest,
    PyColmapBackend,
    VGGTChunkedBackend,
    _call_optional,
    _copy_sparse,
    _read_colmap_cameras_bin,
    _read_colmap_images_bin,
    _read_colmap_points3d_bin,
    align_sim3,
    estimate_poses,
    prepare_precomputed_colmap_scene,
    write_colmap_binary,
)


def _frames(count: int) -> list[dict[str, Any]]:
    return [
        {
            "frame_idx": i,
            "orig_video_frame": i,
            "timestamp_sec": float(i),
            "image_relpath": f"images/{i}.png",
            "colmap_image_id": i + 1,
            "cam_from_world": np.eye(4).tolist(),
        }
        for i in range(count)
    ]


def _request(tmp_path: Path, backend: str = "colmap_global", count: int = 4) -> PoseRequest:
    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)
    for i in range(count):
        Image.new("RGB", (650, 490), color=(i * 15, 80, 160)).save(image_dir / f"{i}.png")
    return PoseRequest(
        image_dir,
        tmp_path / "scene",
        count,
        650,
        490,
        {"fx": 500.0, "fy": 501.0, "cx": 325.0, "cy": 245.0},
        backend=backend,  # type: ignore[arg-type]
        frame_records=_frames(count),
    )


def _output(request: PoseRequest, registered: int, sparse: Path | None = None) -> BackendOutput:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]], dtype=float)
    cameras = np.array([[0, 2, 0], [1, 2, 0], [0, 2, 1], [1, 2, 1]], dtype=float)
    return BackendOutput(registered, 0.4, 4, points, cameras, _frames(request.total_frames), sparse)


def _make_pose(rot_c2w: np.ndarray, cam_center: np.ndarray) -> np.ndarray:
    r_w2c = rot_c2w.T
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = r_w2c
    pose[:3, 3] = -(r_w2c @ cam_center)
    return pose


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def test_global_registration_below_threshold_falls_back_to_incremental(tmp_path: Path) -> None:
    request = _request(tmp_path)
    calls: list[str] = []

    @dataclass
    class Fake:
        name: str
        registered: int

        def estimate(self, request: PoseRequest) -> BackendOutput:
            calls.append(self.name)
            return _output(request, self.registered)

    manifest, telemetry = estimate_poses(
        request,
        {"colmap_global": Fake("global", 3), "colmap_incremental": Fake("incremental", 4)},  # type: ignore[dict-item]
    )
    assert calls == ["global", "incremental"]
    assert manifest.pose_backend == "colmap_incremental"
    assert telemetry.registered_frames == 4


def test_sim3_chunk_alignment_and_camera_extrinsic_transform() -> None:
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    target = (2.5 * (rotation @ source.T)).T + np.array([3, -2, 4])
    scale, recovered_rotation, translation = align_sim3(source, target)
    recovered = (scale * (recovered_rotation @ source.T)).T + translation
    assert np.allclose(recovered, target, atol=1e-4)

    sim = umeyama_sim3(source, target)
    c_src = np.array([0.4, -0.3, -1.5])
    r_c2w_src = _rot_x(0.35)
    pose_src = _make_pose(r_c2w_src, c_src)
    pose_dst = transform_cam_from_world_sim3(pose_src, sim)
    assert np.allclose(pose_dst[:3, :3].T @ pose_dst[:3, :3], np.eye(3), atol=1e-8)
    assert np.linalg.det(pose_dst[:3, :3]) == pytest.approx(1.0)

    c_dst_expected = sim.apply(c_src)
    c_dst_recovered = -(pose_dst[:3, :3].T @ pose_dst[:3, 3])
    assert np.allclose(c_dst_recovered, c_dst_expected, atol=1e-6)

    pt_src = np.array([0.6, 0.2, 2.5])
    pt_dst = sim.apply(pt_src)
    intr = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}
    u1, v1, d1 = project_world_to_image(pt_src, pose_src, intr)
    u2, v2, d2 = project_world_to_image(pt_dst, pose_dst, intr)
    assert np.allclose([u1, v1], [u2, v2], atol=1e-5)
    assert d2 == pytest.approx(sim.scale * d1, rel=1e-6)


def test_output_is_exactly_sparse_zero_and_uses_calibrated_backend_intrinsics(tmp_path: Path) -> None:
    request = _request(tmp_path)
    source = tmp_path / "raw"
    source.mkdir()
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        (source / filename).write_bytes(filename.encode())

    def _calibrated_est(_: Any, r: PoseRequest) -> BackendOutput:
        out = _output(r, 4, source)
        out.width = 640
        out.height = 480
        out.intrinsics = {"fx": 542.5, "fy": 543.0, "cx": 320.0, "cy": 240.0}
        return out

    manifest, _ = estimate_poses(request, {"colmap_global": type("B", (), {"estimate": _calibrated_est})()})  # type: ignore[dict-item]
    assert (tmp_path / "scene" / "sparse" / "0" / "points3D.bin").exists()
    assert manifest.width == 640 and manifest.height == 480
    assert manifest.intrinsics.fx == pytest.approx(542.5)
    assert manifest.intrinsics.fy == pytest.approx(543.0)
    assert manifest.intrinsics.cx == pytest.approx(320.0)
    assert manifest.intrinsics.cy == pytest.approx(240.0)

    # Guard against SameFileError when sparse source == sparse destination
    same_target = tmp_path / "scene" / "sparse" / "0"
    _copy_sparse(same_target, same_target)
    assert (same_target / "points3D.bin").read_bytes() == b"points3D.bin"


def test_call_optional_inspects_signature_and_never_swallows_internal_typeerror(monkeypatch: pytest.MonkeyPatch) -> None:
    def strict_fn(a: int, known: int = 1) -> int:
        if known == 2:
            raise TypeError("internal failure inside fn")
        return a + known

    assert _call_optional(strict_fn, 3, known=4, unsupported="ignored") == 7
    with pytest.raises(TypeError, match="internal failure inside fn"):
        _call_optional(strict_fn, 3, known=2, unsupported="ignored")

    # C-extension / pybind11 simulation where inspect.signature raises ValueError:
    # kwargs are passed directly without a blanket fn(*args) fallback dropping them.
    import inspect

    monkeypatch.setattr(inspect, "signature", lambda _fn: (_ for _ in ()).throw(ValueError("no signature")))
    assert _call_optional(strict_fn, 3, known=4) == 7
    with pytest.raises(TypeError):
        _call_optional(strict_fn, 3, unsupported="ignored")
    with pytest.raises(TypeError, match="internal failure inside fn"):
        _call_optional(strict_fn, 3, known=2)


def test_pycolmap_extract_passes_single_camera_mode_and_pairing_options(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    class FakeCameraMode:
        SINGLE = "SINGLE_MODE"

    class FakeReaderOptions:
        def __init__(self) -> None:
            self.camera_mode: Any = None

    class FakeSiftOptions:
        def __init__(self) -> None:
            self.max_num_features: int = 8192

    class FakeFeatOptions:
        def __init__(self) -> None:
            self.sift = FakeSiftOptions()

    class FakeMatchOptions:
        pass

    class FakePairOptions:
        def __init__(self) -> None:
            self.overlap: int = 10
            self.loop_detection: bool = False

    class FakePyColmap:
        CameraMode = FakeCameraMode
        ImageReaderOptions = FakeReaderOptions
        FeatureExtractionOptions = FakeFeatOptions
        FeatureMatchingOptions = FakeMatchOptions
        SequentialPairingOptions = FakePairOptions

        @staticmethod
        def extract_features(db: str, img_dir: str, *, camera_mode: Any = None, reader_options: Any = None, extraction_options: Any = None, **extra: Any) -> None:
            captured["extract"] = (db, img_dir, camera_mode, reader_options, extraction_options, extra)

        @staticmethod
        def match_sequential(db: str, *, matching_options: Any = None, pairing_options: Any = None) -> str:
            captured["match"] = (db, matching_options, pairing_options)
            return "matched"

    request = _request(tmp_path, count=2)
    res = PyColmapBackend("global")._extract(FakePyColmap(), tmp_path / "database.db", request)
    assert res == "matched"
    assert captured["extract"][2] == "SINGLE_MODE"
    assert captured["extract"][3].camera_mode == "SINGLE_MODE"
    assert captured["extract"][4].sift.max_num_features == 4096
    assert "image_options" not in captured["extract"][5]
    assert isinstance(captured["match"][1], FakeMatchOptions)
    assert captured["match"][2].overlap == 12
    assert captured["match"][2].loop_detection is False


def test_pycolmap_from_reconstruction_filename_join_mod16_crop_and_failures(tmp_path: Path) -> None:
    request = _request(tmp_path, count=4)
    undist_imgs = request.output_dir / "undistorted" / "images"
    undist_imgs.mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (650, 490), color=(i * 40, 100, 150)).save(undist_imgs / f"{i}.png")

    @dataclass
    class FakeCam:
        model: str
        width: int
        height: int
        params: list[float]

    @dataclass
    class FakeImg:
        image_id: int
        name: str
        camera_id: int
        cam_from_world: np.ndarray

    @dataclass
    class FakePoint:
        xyz: np.ndarray
        color: tuple[int, int, int] = (10, 20, 30)

    @dataclass
    class FakeRec:
        cameras: dict[int, FakeCam]
        images: dict[int, FakeImg]
        points3D: dict[int, FakePoint]

        def compute_mean_reprojection_error(self) -> float:
            return 0.32

    # Register frames 0, 2, 3 with directory-prefixed names; leave frame 1 unregistered
    rec = FakeRec(
        cameras={1: FakeCam("PINHOLE", 650, 490, [530.0, 532.0, 325.0, 245.0])},
        images={
            10: FakeImg(10, "cam0/0.png", 1, _make_pose(np.eye(3), np.array([0.0, 1.0, 0.0]))),
            11: FakeImg(11, "cam0/2.png", 1, _make_pose(np.eye(3), np.array([1.0, 1.0, 0.0]))),
            12: FakeImg(12, "3.png", 1, _make_pose(np.eye(3), np.array([2.0, 1.0, 0.0]))),
        },
        points3D={
            1: FakePoint(np.array([0.0, 0.0, 0.0])),
            2: FakePoint(np.array([1.0, 0.0, 0.0])),
            3: FakePoint(np.array([0.0, 0.0, 1.0])),
        },
    )

    sparse_dir = request.output_dir / "sparse" / "0"
    backend = PyColmapBackend("global")
    out = backend._from_reconstruction(rec, request, sparse_dir)

    assert out.registered_frames == 3
    assert out.width == 640 and out.height == 480
    assert out.intrinsics == {"fx": 530.0, "fy": 532.0, "cx": 320.0, "cy": 240.0}
    assert rec.cameras[1].width == 640 and rec.cameras[1].height == 480
    assert rec.cameras[1].params == [530.0, 532.0, 320.0, 240.0]
    with Image.open(undist_imgs / "0.png") as cropped:
        assert cropped.size == (640, 480)
    # Published undistorted mod-16 frames exist at request.output_dir / image_relpath
    for i in range(4):
        published = request.output_dir / f"images/{i}.png"
        assert published.exists()
        with Image.open(published) as pub_img:
            assert pub_img.size == (640, 480)

    # Joined frames keep original frame_idx/timestamp_sec and keep unregistered frame 1 as None
    assert [f["frame_idx"] for f in out.frames] == [0, 1, 2, 3]
    assert out.frames[0]["colmap_image_id"] == 10
    assert out.frames[1]["colmap_image_id"] is None and out.frames[1]["cam_from_world"] is None
    assert out.frames[2]["colmap_image_id"] == 11
    assert out.frames[3]["colmap_image_id"] == 12
    assert (sparse_dir / "cameras.bin").exists() and (sparse_dir / "images.bin").exists() and (sparse_dir / "points3D.bin").exists()

    # Validate SceneManifest accepts partial registration + calibrated intrinsics
    manifest, _ = estimate_poses(request, {"colmap_global": type("B", (), {"estimate": lambda _, r: out})()})  # type: ignore[dict-item]
    assert manifest.frames[1].colmap_image_id is None
    assert manifest.intrinsics.fx == pytest.approx(530.0)

    # Failure: fewer than 2 registered frames
    rec_one = FakeRec(rec.cameras, {10: rec.images[10]}, rec.points3D)
    with pytest.raises(ColmapFailureError, match="at least 2"):
        backend._from_reconstruction(rec_one, request, sparse_dir)

    # Failure: empty points3D (no fake 3-point fabrication)
    rec_empty_pts = FakeRec(rec.cameras, rec.images, {})
    with pytest.raises(ColmapFailureError, match="empty points3D"):
        backend._from_reconstruction(rec_empty_pts, request, sparse_dir)

    # Failure: distorted non-PINHOLE camera model
    rec_distorted = FakeRec({1: FakeCam("OPENCV", 640, 480, [500, 500, 320, 240])}, rec.images, rec.points3D)
    with pytest.raises(ColmapFailureError, match="PINHOLE"):
        backend._from_reconstruction(rec_distorted, request, sparse_dir)


def test_vggt_chunked_three_plus_chunks_chained_sim3_and_colmap_binary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        VGGTChunkedBackend(type("R", (), {})(), chunk_size=1, overlap=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="overlap"):
        VGGTChunkedBackend(type("R", (), {})(), chunk_size=4, overlap=4)  # type: ignore[arg-type]

    count = 12
    request = _request(tmp_path, backend="vggt_chunked", count=count)
    # Ground-truth circular camera arc and 3D room points
    gt_centers = np.array([[np.cos(i * 0.25), 1.5, np.sin(i * 0.25) + 0.1 * (i % 3)] for i in range(count)], dtype=np.float64)
    gt_rots = [_rot_z(0.1 * i) @ _rot_x(0.05 * i) for i in range(count)]
    gt_poses = [_make_pose(r, c) for r, c in zip(gt_rots, gt_centers)]
    gt_depths = np.full((count, 490, 650), 3.0, dtype=np.float32)

    chunk_points = [
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float64),
        np.array([[1.0, 0.2, 0.5], [2.5, 0.2, 0.5], [1.0, 0.2, 2.0]], dtype=np.float64),
        np.array([[0.5, -0.3, 1.0], [2.0, -0.3, 1.0], [0.5, -0.3, 2.5]], dtype=np.float64),
    ]
    # Inverse Sim(3) applied by runner to chunks 1 and 2 so chunk 2 shares ZERO frames with chunk 0
    inv_sims = [
        Sim3Transform(1.0, np.eye(3), np.zeros(3)),
        Sim3Transform(0.5, _rot_z(0.6), np.array([4.0, -3.0, 2.0])),
        Sim3Transform(2.0, _rot_x(-0.5) @ _rot_z(0.3), np.array([-5.0, 7.0, 1.5])),
    ]

    calls: list[tuple[str, list[str]]] = []

    class MultiChunkRunner:
        def infer(self, images: list[Path], *, dtype: str) -> BackendOutput:
            chunk_idx = len(calls)
            calls.append((dtype, [p.name for p in images]))
            sim = inv_sims[chunk_idx]
            indices = [int(p.stem) for p in images]
            frames_out = []
            cams_out = []
            depths_out = []
            for local_idx, idx in enumerate(indices):
                pose_local = transform_cam_from_world_sim3(gt_poses[idx], sim)
                center_local = sim.apply(gt_centers[idx])
                cams_out.append(center_local)
                depths_out.append(gt_depths[idx] * np.float32(sim.scale))
                # Emit chunk-local frame_idx without orig_video_frame/timestamp_sec to prove filename join restores them
                frames_out.append({
                    "frame_idx": local_idx,
                    "image_relpath": f"images/{idx}.png",
                    "colmap_image_id": None,
                    "cam_from_world": pose_local.tolist(),
                })
            pts_local = sim.apply(chunk_points[chunk_idx])
            return BackendOutput(
                registered_frames=len(indices),
                mean_reproj_px=0.25,
                num_points3d=len(pts_local),
                points_xyz=pts_local,
                cam_positions_xyz=np.asarray(cams_out, dtype=np.float64),
                frames=frames_out,
                depths=np.stack(depths_out, axis=0),
            )

    # chunk_size=6, overlap=3 -> chunks: [0..5], [3..8], [6..11] (chunk 2 only overlaps chunk 1!)
    output = VGGTChunkedBackend(MultiChunkRunner(), chunk_size=6, overlap=3).estimate(request)

    assert len(calls) == 3
    assert all(dtype == "fp16" for dtype, _ in calls)
    assert output.registered_frames == count
    assert [fr["frame_idx"] for fr in output.frames] == list(range(count))
    assert [fr["orig_video_frame"] for fr in output.frames] == list(range(count))
    assert [fr["timestamp_sec"] for fr in output.frames] == [float(i) for i in range(count)]
    assert np.allclose(output.cam_positions_xyz, gt_centers, atol=1e-4)
    assert np.allclose(output.points_xyz, np.vstack(chunk_points), atol=1e-4)
    assert output.depths.shape == (count, 480, 640)
    assert np.allclose(output.depths, 3.0, atol=1e-4)
    for idx, fr in enumerate(output.frames):
        pose = np.asarray(fr["cam_from_world"], dtype=np.float64)
        assert np.allclose(pose, gt_poses[idx], atol=1e-4)
        pub_img_path = request.output_dir / fr["image_relpath"]
        assert pub_img_path.exists()
        with Image.open(pub_img_path) as pub_img:
            assert pub_img.size == (640, 480)
        depth_file = request.output_dir / "depths" / f"{idx:06d}.npy"
        assert depth_file.exists()
        loaded_d = np.load(depth_file, allow_pickle=False)
        assert loaded_d.shape == (480, 640) and np.allclose(loaded_d, 3.0, atol=1e-4)

    # Default backend registry includes vggt_chunked and raises StageExecutionError without weights
    with pytest.raises(StageExecutionError):
        estimate_poses(request)

    # _DefaultVGGTRunner executes vggt.infer_chunk when vggt module exposes it
    import sys
    import types

    def _fake_infer_chunk(imgs: list[Path], *, dtype: str) -> dict[str, Any]:
        ext = np.stack([_make_pose(np.eye(3), np.array([float(i), 0.0, 0.0])) for i in range(len(imgs))], axis=0)
        # Simulate VGGT preprocessed 392x518 resolution for 650x490 original images
        return {
            "extrinsics": ext[:, :3, :],
            "intrinsics": np.array([[518.0, 0.0, 259.0], [0.0, 392.0, 196.0], [0.0, 0.0, 1.0]], dtype=np.float64),
            "world_points": np.array([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0], [0.0, 1.0, 4.0]], dtype=np.float64),
            "depths": np.full((len(imgs), 392, 518), 2.5, dtype=np.float32),
            "width": 518,
            "height": 392,
        }

    fake_vggt = types.ModuleType("vggt")
    fake_vggt.infer_chunk = _fake_infer_chunk  # type: ignore[attr-defined]
    prev_vggt = sys.modules.get("vggt")
    sys.modules["vggt"] = fake_vggt
    try:
        from room3d_worker.pipeline.s02_poses import _DefaultVGGTRunner

        runner_out = _DefaultVGGTRunner().infer([request.image_dir / f"{i}.png" for i in range(3)], dtype="fp16")
        assert runner_out.registered_frames == 3
        assert runner_out.width == 650 and runner_out.height == 490
        assert runner_out.depths is not None and runner_out.depths.shape == (3, 490, 650)
        assert runner_out.intrinsics["fx"] == pytest.approx(650.0)
        assert runner_out.intrinsics["fy"] == pytest.approx(490.0)
        assert runner_out.intrinsics["cx"] == pytest.approx(325.0)
        assert runner_out.intrinsics["cy"] == pytest.approx(245.0)

        # 5D [1, S, H, W, 1] depth + portrait 480x640 cropped to 518x518 (resized 518x686, crop_y=84)
        portrait_dir = tmp_path / "portrait_imgs"
        portrait_dir.mkdir(exist_ok=True)
        portrait_paths = []
        for idx in range(2):
            p_path = portrait_dir / f"{idx}.png"
            Image.new("RGB", (480, 640), color=(idx * 30, 60, 120)).save(p_path)
            portrait_paths.append(p_path)

        from room3d_worker.pipeline.s02_poses import _coerce_vggt_output

        portrait_raw = {
            "extrinsics": np.stack([_make_pose(np.eye(3), np.array([float(i), 0.0, 0.0]))[:3, :] for i in range(2)], axis=0),
            "intrinsics": np.array([[400.0, 0.0, 259.0], [0.0, 400.0, 200.0], [0.0, 0.0, 1.0]], dtype=np.float64),
            "world_points": np.array([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0], [0.0, 1.0, 4.0]], dtype=np.float64),
            "depth": np.full((1, 2, 518, 518, 1), 3.2, dtype=np.float32),
            "width": 518,
            "height": 518,
        }
        portrait_out = _coerce_vggt_output(portrait_raw, portrait_paths)
        assert portrait_out.width == 480 and portrait_out.height == 640
        assert portrait_out.intrinsics["fx"] == pytest.approx(400.0 * (480.0 / 518.0), abs=1e-2)
        assert portrait_out.intrinsics["fy"] == pytest.approx(373.18, abs=1e-2)
        assert portrait_out.intrinsics["cx"] == pytest.approx(259.0 * (480.0 / 518.0), abs=1e-2)
        assert portrait_out.intrinsics["cy"] == pytest.approx(264.96, abs=1e-2)
        assert portrait_out.depths is not None and portrait_out.depths.shape == (2, 640, 480)
        assert np.all(portrait_out.depths[:, :78, :] == 0.0)
        assert np.all(portrait_out.depths[:, 562:, :] == 0.0)
        assert np.allclose(portrait_out.depths[:, 78:562, :], 3.2, atol=1e-4)

        bad_depth = dict(portrait_raw, depth=np.full((2, 2, 518, 518, 1), 1.0, dtype=np.float32))
        with pytest.raises(StageExecutionError, match="5D"):
            _coerce_vggt_output(bad_depth, portrait_paths)

        # Singleton S=1 depth with [1, H, W, 1], [1, 1, H, W], and [1, 1, H, W, 1]
        for s1_shape in ((1, 490, 650, 1), (1, 1, 490, 650), (1, 1, 490, 650, 1)):
            s1_raw = {
                "extrinsics": _make_pose(np.eye(3), np.zeros(3))[None, :3, :],
                "intrinsics": np.array([[500.0, 0.0, 325.0], [0.0, 500.0, 245.0], [0.0, 0.0, 1.0]], dtype=np.float64),
                "world_points": np.array([[0.0, 0.0, 2.0]], dtype=np.float64),
                "depth": np.full(s1_shape, 1.75, dtype=np.float32),
                "width": 650,
                "height": 490,
            }
            s1_out = _coerce_vggt_output(s1_raw, [request.image_dir / "0.png"])
            assert s1_out.depths is not None and s1_out.depths.shape == (1, 490, 650)
            assert np.allclose(s1_out.depths, 1.75)
    finally:
        if prev_vggt is None:
            sys.modules.pop("vggt", None)
        else:
            sys.modules["vggt"] = prev_vggt

    sparse_dir = tmp_path / "scene" / "sparse" / "0"
    assert output.sparse_dir == sparse_dir
    with (sparse_dir / "cameras.bin").open("rb") as fh:
        num_cams = struct.unpack("<Q", fh.read(8))[0]
        assert num_cams == 1
    with (sparse_dir / "images.bin").open("rb") as fh:
        num_imgs = struct.unpack("<Q", fh.read(8))[0]
        assert num_imgs == count
    with (sparse_dir / "points3D.bin").open("rb") as fh:
        num_pts = struct.unpack("<Q", fh.read(8))[0]
        assert num_pts == sum(len(p) for p in chunk_points)


def test_scene_manifest_strict_pinhole_and_finite_transform() -> None:
    base = {
        "schema_version": "1.0.0",
        "project_id": "p1",
        "version_id": "v1",
        "coord_convention": "opencv_colmap",
        "T_viewer_from_colmap": np.eye(4).tolist(),
        "camera_model": "PINHOLE",
        "width": 640,
        "height": 480,
        "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0},
        "test_every": 0,
        "frames": [
            {"frame_idx": 0, "orig_video_frame": 0, "timestamp_sec": 0.0,
             "image_relpath": "images/0.png", "colmap_image_id": None, "cam_from_world": None}
        ],
        "artifacts": {"sparse_dir": "sparse/0", "spz_relpath": "scene.spz", "sparse_points_relpath": "sparse/0/points3D.bin"},
    }
    valid = SceneManifest.model_validate(base)
    assert valid.intrinsics.model == "PINHOLE"
    assert valid.frames[0].colmap_image_id is None

    bad_model = {**base, "intrinsics": {"model": "OPENCV", "fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}}
    with pytest.raises(ValidationError):
        SceneManifest.model_validate(bad_model)

    subpixel_ok = {**base, "intrinsics": {"fx": 500.0, "fy": 500.0, "cx": 319.5, "cy": 239.5}}
    assert SceneManifest.model_validate(subpixel_ok).intrinsics.cx == pytest.approx(319.5)

    for bad_intr in (
        {"fx": 500.0, "fy": 500.0, "cx": 0.0, "cy": 240.0},
        {"fx": 500.0, "fy": 500.0, "cx": 640.0, "cy": 240.0},
        {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 480.0},
    ):
        with pytest.raises(ValidationError):
            SceneManifest.model_validate({**base, "intrinsics": bad_intr})

    bad_nan = {**base, "T_viewer_from_colmap": [[1.0, 0.0, 0.0, float("nan")], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
    with pytest.raises(ValidationError):
        SceneManifest.model_validate(bad_nan)


def test_gaussian_state_roundtrip_and_legacy_checkpoints(tmp_path: Path) -> None:
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rgb = np.array([[1.2, 0.5, -0.1], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    state = GaussianState.from_sparse_points(xyz, rgb)
    assert state.scales.shape == (4, 3) and (state.scales > 0).all()
    assert np.allclose(state.quats, np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)))
    assert np.allclose(state.opacities, 0.5)
    assert np.allclose(state.sh0[0], [1.0, 0.5, 0.0])

    # .npz round-trip with sh_rest
    state_sh = GaussianState(
        xyz=state.xyz, scales=state.scales, quats=state.quats,
        opacities=state.opacities, sh0=state.sh0,
        sh_rest=np.ones((4, 3, 3), dtype=np.float32) * 0.1,
    )
    npz_path = state_sh.save_npz(tmp_path / "gs_state.npz")
    loaded_npz = GaussianState.load_checkpoint(npz_path)
    assert np.allclose(loaded_npz.xyz, state_sh.xyz)
    assert np.allclose(loaded_npz.scales, state_sh.scales)
    assert loaded_npz.sh_rest is not None and np.allclose(loaded_npz.sh_rest, state_sh.sh_rest)

    # .pt round-trip
    pt_path = state.save_pt(tmp_path / "gs_state.pt")
    loaded_pt = GaussianState.load_checkpoint(pt_path)
    assert np.allclose(loaded_pt.xyz, state.xyz)
    assert np.allclose(loaded_pt.sh0, state.sh0)

    # Legacy .pt dict checkpoint (1D log scales, uint8 rgb, logit opacities)
    legacy_pt = tmp_path / "legacy.pt"
    torch.save({
        "xyz": torch.from_numpy(xyz),
        "scales": torch.tensor([-3.0, -3.0, -3.0, -3.0], dtype=torch.float32),
        "rgb": torch.tensor([[255, 0, 128], [0, 255, 0], [0, 0, 255], [255, 255, 255]], dtype=torch.uint8),
        "opacities": torch.tensor([0.0, 2.0, -2.0, 1.0], dtype=torch.float32),
    }, legacy_pt)
    loaded_legacy_pt = GaussianState.load_checkpoint(legacy_pt)
    assert loaded_legacy_pt.scales.shape == (4, 3)
    assert np.allclose(loaded_legacy_pt.scales, np.exp(-3.0), rtol=1e-5)
    assert np.allclose(loaded_legacy_pt.opacities[0], 0.5, atol=1e-5)

    # Standard 3DGS .ply checkpoint
    sh_c0 = 0.28209479177387814
    ply_path = tmp_path / "scene_3dgs.ply"
    dtype_3dgs = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ])
    verts = np.zeros(2, dtype=dtype_3dgs)
    verts["x"] = [1.0, 2.0]
    verts["f_dc_0"] = (0.8 - 0.5) / sh_c0
    verts["f_dc_1"] = (0.2 - 0.5) / sh_c0
    verts["f_dc_2"] = (0.4 - 0.5) / sh_c0
    verts["opacity"] = [0.0, 0.0]
    verts["scale_0"] = verts["scale_1"] = verts["scale_2"] = np.log(0.05)
    verts["rot_0"] = 1.0
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float opacity\n"
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        "end_header\n"
    ).encode("ascii")
    ply_path.write_bytes(header + verts.tobytes())
    loaded_ply = GaussianState.load_checkpoint(ply_path)
    assert np.allclose(loaded_ply.sh0[0], [0.8, 0.2, 0.4], atol=1e-5)
    assert np.allclose(loaded_ply.scales, 0.05, atol=1e-5)
    assert np.allclose(loaded_ply.opacities, 0.5, atol=1e-5)

    # Legacy points_xyz-keyed .npz preserves and normalizes non-identity quats
    legacy_points_npz = tmp_path / "legacy_points.npz"
    np.savez(
        legacy_points_npz,
        points_xyz=xyz,
        scales=np.full((4, 3), 0.1, dtype=np.float32),
        quats=np.tile(np.array([[0.0, 2.0, 0.0, 0.0]], dtype=np.float32), (4, 1)),
        opacities=np.full((4,), 0.6, dtype=np.float32),
        sh0=rgb,
    )
    loaded_legacy_npz = GaussianState.load_npz(legacy_points_npz)
    assert np.allclose(loaded_legacy_npz.quats, np.tile(np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32), (4, 1)))

    # Validation failures
    bad_state = GaussianState(xyz=xyz, scales=-np.ones((4, 3), np.float32), quats=state.quats, opacities=state.opacities, sh0=state.sh0)
    with pytest.raises(GeometryError, match="strictly positive"):
        bad_state.validate()


def test_prepare_precomputed_colmap_scene_scales_intrinsics_and_crops_mod16(tmp_path: Path) -> None:
    import json

    from room3d_worker.common.coords import mod16_crop_or_pad_params

    pts_xyz = np.array([[0.0, 0.0, 4.0], [1.5, 0.0, 4.0], [0.0, 1.5, 4.0], [1.0, 1.0, 5.0]], dtype=np.float64)
    pts_rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [200, 200, 200]], dtype=np.uint8)

    for label, old_w, old_h, fx_old, fy_old, cx_old, cy_old in (
        ("playroom_1264x832", 1264, 832, 1080.0, 1082.0, 632.0, 416.0),
        ("drjohnson_1332x876", 1332, 876, 1140.0, 1145.0, 666.0, 438.0),
    ):
        src_dir = tmp_path / f"{label}_src"
        (src_dir / "images").mkdir(parents=True)
        for idx in range(3):
            Image.new("RGB", (old_w, old_h), color=(20 * idx, 100, 180)).save(src_dir / "images" / f"frame_{idx}.jpg")

        src_frames = [
            {
                "frame_idx": idx,
                "image_relpath": f"images/frame_{idx}.jpg",
                "colmap_image_id": idx + 1,
                "cam_from_world": _make_pose(_rot_z(0.1 * idx), np.array([float(idx), 0.5, 0.0])).tolist(),
            }
            for idx in range(3)
        ]
        write_colmap_binary(
            src_dir / "sparse" / "0",
            old_w,
            old_h,
            {"fx": fx_old, "fy": fy_old, "cx": cx_old, "cy": cy_old},
            src_frames,
            pts_xyz,
            pts_rgb,
        )

        out_dir = tmp_path / f"{label}_out"
        manifest, telemetry = prepare_precomputed_colmap_scene(
            src_dir,
            out_dir,
            project_id="proj_pre",
            version_id="v1",
            max_long_side=960,
        )

        scale = 960.0 / float(max(old_w, old_h))
        resized_w = int(round(old_w * scale))
        resized_h = int(round(old_h * scale))
        scale_x = resized_w / float(old_w)
        scale_y = resized_h / float(old_h)
        exp_w, exp_h, exp_fx, exp_fy, exp_cx, exp_cy, crop_box = mod16_crop_or_pad_params(
            resized_w, resized_h, cx_old * scale_x, cy_old * scale_y, fx_old * scale_x, fy_old * scale_y
        )

        assert manifest.pose_backend == "precomputed_colmap"
        assert manifest.width == exp_w and manifest.height == exp_h
        assert manifest.width == 960 and manifest.height % 16 == 0
        assert manifest.intrinsics.fx == pytest.approx(exp_fx)
        assert manifest.intrinsics.fy == pytest.approx(exp_fy)
        assert manifest.intrinsics.cx == pytest.approx(exp_cx)
        assert manifest.intrinsics.cy == pytest.approx(exp_cy)
        assert telemetry.registered_frames == 3

        cams_out = _read_colmap_cameras_bin(out_dir / "sparse" / "0" / "cameras.bin")
        assert cams_out[1]["width"] == exp_w and cams_out[1]["height"] == exp_h
        assert cams_out[1]["fx"] == pytest.approx(exp_fx)
        assert cams_out[1]["fy"] == pytest.approx(exp_fy)
        assert cams_out[1]["cx"] == pytest.approx(exp_cx)
        assert cams_out[1]["cy"] == pytest.approx(exp_cy)

        imgs_out = _read_colmap_images_bin(out_dir / "sparse" / "0" / "images.bin")
        assert len(imgs_out) == 3
        assert np.allclose(imgs_out[2]["cam_from_world"], src_frames[1]["cam_from_world"], atol=1e-6)
        pts_out, rgb_out = _read_colmap_points3d_bin(out_dir / "sparse" / "0" / "points3D.bin")
        assert np.allclose(pts_out, pts_xyz) and np.array_equal(rgb_out, pts_rgb)

        for idx in range(3):
            with Image.open(out_dir / "images" / f"frame_{idx}.jpg") as saved_im:
                assert saved_im.size == (exp_w, exp_h)
        assert json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))["width"] == exp_w
        assert len(json.loads((out_dir / "sparse_points.json").read_text(encoding="utf-8"))) == 4

    # PrecomputedColmapBackend via estimate_poses() when images are pre-resized to 960x632 against 1264x832 cameras.bin
    pre_dir = tmp_path / "backend_pre"
    (pre_dir / "images").mkdir(parents=True)
    for idx in range(3):
        Image.new("RGB", (960, 632), color=(50, 90, 130)).save(pre_dir / "images" / f"frame_{idx}.jpg")
    write_colmap_binary(
        pre_dir / "sparse" / "0",
        1264,
        832,
        {"fx": 1080.0, "fy": 1082.0, "cx": 632.0, "cy": 416.0},
        [
            {
                "frame_idx": idx,
                "image_relpath": f"images/frame_{idx}.jpg",
                "colmap_image_id": idx + 1,
                "cam_from_world": _make_pose(np.eye(3), np.array([float(idx), 0.0, 0.0])).tolist(),
            }
            for idx in range(3)
        ],
        pts_xyz,
        pts_rgb,
    )
    req = PoseRequest(
        image_dir=pre_dir / "images",
        output_dir=pre_dir / "out",
        total_frames=3,
        width=960,
        height=632,
        intrinsics={"fx": 1080.0, "fy": 1082.0, "cx": 632.0, "cy": 416.0},
        backend="precomputed_colmap",
    )
    est_manifest, _ = estimate_poses(req)
    assert est_manifest.width == 960 and est_manifest.height == 624
    assert est_manifest.pose_backend == "precomputed_colmap"
    assert est_manifest.intrinsics.fx == pytest.approx(1080.0 * (960.0 / 1264.0))
    assert est_manifest.intrinsics.cy == pytest.approx(416.0 * (632.0 / 832.0) - 4.0)


def test_precomputed_colmap_rejects_nonzero_distortion_and_incompatible_multicam(tmp_path: Path) -> None:
    pts_xyz = np.array([[0.0, 0.0, 4.0], [1.0, 0.0, 4.0], [0.0, 1.0, 4.0]], dtype=np.float64)
    pts_rgb = np.full((3, 3), 200, dtype=np.uint8)

    # 1) Non-zero SIMPLE_RADIAL distortion coefficient raises ColmapFailureError
    radial_bin = tmp_path / "radial_cameras.bin"
    with radial_bin.open("wb") as fh:
        fh.write(struct.pack("<Q", 1))
        fh.write(struct.pack("<IiQQ4d", 1, 2, 1264, 832, 1080.0, 632.0, 416.0, -0.05))
    with pytest.raises(ColmapFailureError, match="non-zero distortion"):
        _read_colmap_cameras_bin(radial_bin)

    # 2) Zero distortion SIMPLE_RADIAL succeeds
    radial_zero_bin = tmp_path / "radial_zero_cameras.bin"
    with radial_zero_bin.open("wb") as fh:
        fh.write(struct.pack("<Q", 1))
        fh.write(struct.pack("<IiQQ4d", 1, 2, 1264, 832, 1080.0, 632.0, 416.0, 0.0))
    parsed_zero = _read_colmap_cameras_bin(radial_zero_bin)
    assert parsed_zero[1]["fx"] == pytest.approx(1080.0)

    # 3) Multi-camera scene with differing focal lengths raises ColmapFailureError in both paths
    multi_dir = tmp_path / "multi_cam"
    (multi_dir / "images").mkdir(parents=True)
    for idx in range(2):
        Image.new("RGB", (1264, 832), color=(40, 80, 120)).save(multi_dir / "images" / f"frame_{idx}.jpg")
    write_colmap_binary(
        multi_dir / "sparse" / "0",
        1264,
        832,
        {"fx": 1080.0, "fy": 1080.0, "cx": 632.0, "cy": 416.0},
        [
            {"frame_idx": 0, "image_relpath": "images/frame_0.jpg", "colmap_image_id": 1, "cam_from_world": np.eye(4).tolist()},
            {"frame_idx": 1, "image_relpath": "images/frame_1.jpg", "colmap_image_id": 2, "cam_from_world": np.eye(4).tolist()},
        ],
        pts_xyz,
        pts_rgb,
    )
    # Overwrite cameras.bin with 2 PINHOLE cameras of differing fx, and rewrite images.bin referencing cam 1 and cam 2
    with (multi_dir / "sparse" / "0" / "cameras.bin").open("wb") as fh:
        fh.write(struct.pack("<Q", 2))
        fh.write(struct.pack("<IiQQ4d", 1, 1, 1264, 832, 1080.0, 1080.0, 632.0, 416.0))
        fh.write(struct.pack("<IiQQ4d", 2, 1, 1264, 832, 1150.0, 1150.0, 632.0, 416.0))
    with (multi_dir / "sparse" / "0" / "images.bin").open("wb") as fh:
        fh.write(struct.pack("<Q", 2))
        for img_id, cam_id in ((1, 1), (2, 2)):
            fh.write(struct.pack("<I4d3dI", img_id, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, cam_id))
            fh.write(f"frame_{img_id - 1}.jpg".encode("utf-8") + b"\x00")
            fh.write(struct.pack("<Q", 0))

    with pytest.raises(ColmapFailureError, match="multiple cameras"):
        prepare_precomputed_colmap_scene(multi_dir, tmp_path / "multi_out", project_id="p", version_id="v1")

    with pytest.raises(ColmapFailureError, match="multiple cameras"):
        estimate_poses(PoseRequest(
            image_dir=multi_dir / "images",
            output_dir=tmp_path / "multi_est_out",
            total_frames=2,
            width=1264,
            height=832,
            intrinsics={"fx": 1080.0, "fy": 1080.0, "cx": 632.0, "cy": 416.0},
            backend="precomputed_colmap",
        ))

    # 4) Non-finite NaN / Inf camera parameters or distortion coefficients raise ColmapFailureError
    for bad_val, label in ((float("nan"), "nan"), (float("inf"), "inf")):
        bad_cam_bin = tmp_path / f"bad_{label}_cameras.bin"
        with bad_cam_bin.open("wb") as fh:
            fh.write(struct.pack("<Q", 1))
            fh.write(struct.pack("<IiQQ4d", 1, 2, 1264, 832, 1080.0, 632.0, 416.0, bad_val))
        with pytest.raises(ColmapFailureError, match="non-finite"):
            _read_colmap_cameras_bin(bad_cam_bin)
