from __future__ import annotations

import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pytest

from room3d_worker.common.errors import StageExecutionError
from room3d_worker.common.manifest import load_manifest
from room3d_worker.pipeline.s02_poses import _read_colmap_images_bin
from room3d_worker.pipeline.s04_export import read_ply_splats, read_spz_header
from room3d_worker.pipeline.s07_rebuild import (
    GaussianState,
    RebuildRequest,
    _load_training_views,
    _sync_colmap_images_bin_names,
    compute_masked_l1_loss,
    rebuild_scene,
)


from scene_fixtures import _setup_base_scene


def test_masked_finetune_pytorch_opt_freezes_background_and_updates_roi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    base_dir, edited_dir, masks_dir = _setup_base_scene(tmp_path)
    out_dir = tmp_path / "v2_masked"

    def _fake_rasterize(*, colors: Any, means: Any, width: int, height: int, **_kwargs: Any) -> tuple[Any, ...]:
        _ = means
        fg_color = colors[0]
        canvas = fg_color.view(1, 1, 1, 3).expand(1, height, width, 3)
        return (canvas,)

    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(rasterization=_fake_rasterize))
    metrics = rebuild_scene(
        RebuildRequest(
            base_dir=base_dir, edited_frames_dir=edited_dir, masks_dir=masks_dir, output_dir=out_dir,
            project_id="proj-1", version_id="v2_masked", parent_version_id="v1",
            mode="masked_finetune", fine_tune_steps=25, exclude_frames=(1,), roi_pad_px=2, min_mask_hits=1, device="cpu",
        )
    )
    assert metrics["roi_gaussian_count"] == 2
    assert metrics["frozen_gaussian_count"] == 2
    assert metrics["max_pose_drift"] == 0.0
    assert metrics["max_unmasked_gaussian_drift"] == 0.0
    assert metrics["excluded_frame_count"] == 0
    rebuilt_manifest = load_manifest(out_dir / "manifest.json")
    assert all(not f.excluded for f in rebuilt_manifest.frames)
    with np.load(out_dir / "gs_state.npz", allow_pickle=False) as archive:
        cols = archive["colors"]
        assert cols[0, 0] > 120
        assert tuple(cols[1]) == (120, 120, 120)
        assert archive["quats"].shape == (4, 4)
        assert archive["scales"].shape == (4, 3)
    assert (out_dir / "sparse" / "0" / "cameras.bin").read_bytes() == b"cam"
    colmap_imgs = _read_colmap_images_bin(out_dir / "sparse" / "0" / "images.bin")
    assert [colmap_imgs[k]["name"] for k in sorted(colmap_imgs)] == ["frame_000000.png", "frame_000001.jpg"]
    for img in colmap_imgs.values():
        assert (out_dir / "images" / img["name"]).is_file()
    assert (out_dir / "depths" / "00000.npy").exists()
    assert rebuilt_manifest.frames[0].image_relpath.endswith(".png")
    assert rebuilt_manifest.frames[1].image_relpath.endswith(".jpg")
    for frame in rebuilt_manifest.frames:
        img_out = out_dir / frame.image_relpath
        assert img_out.is_file()
        with Image.open(img_out) as im:
            arr = np.asarray(im.convert("RGB"))
            if frame.frame_idx == 0:
                with Image.open(edited_dir / "00000.png") as ed:
                    assert int(np.max(np.abs(arr.astype(np.int16) - np.asarray(ed.convert("RGB")).astype(np.int16)))) == 0
            else:
                assert np.mean(arr) == pytest.approx(55.0, abs=3.0)
    assert read_ply_splats(out_dir / "scene.ply").scales.shape == (4, 3)
    assert read_spz_header(out_dir / "scene.spz")["magic"] == 0x5053474E


def test_rebuild_merges_partial_depths_and_fails_on_missing_source_image(tmp_path: Path) -> None:
    base_dir, edited_dir, masks_dir = _setup_base_scene(tmp_path)
    out_partial = tmp_path / "v2_partial"
    (out_partial / "depths").mkdir(parents=True, exist_ok=True)
    np.save(out_partial / "depths" / "00000.npy", np.full((128, 160), 9.0, dtype=np.float32))
    rebuild_scene(
        RebuildRequest(
            base_dir=base_dir, edited_frames_dir=edited_dir, masks_dir=masks_dir, output_dir=out_partial,
            project_id="proj-1", version_id="v2_partial", parent_version_id="v1", mode="from_scratch",
        ),
        backend=lambda _r, _m, state, _roi: state,
    )
    assert np.allclose(np.load(out_partial / "depths" / "00000.npy"), 9.0)
    assert (out_partial / "depths" / "00001.npy").exists()

    (base_dir / "images" / "frame_000001.jpg").unlink()
    (edited_dir / "00001.png").unlink()
    with pytest.raises(StageExecutionError, match="Missing base frame image"):
        rebuild_scene(
            RebuildRequest(
                base_dir=base_dir, edited_frames_dir=edited_dir, masks_dir=masks_dir, output_dir=tmp_path / "v2_missing",
                project_id="proj-1", version_id="v2_missing", parent_version_id="v1", mode="from_scratch",
            ),
            backend=lambda _r, _m, state, _roi: state,
        )


def test_leaking_backend_mutating_background_fails_loudly(tmp_path: Path) -> None:
    base_dir, edited_dir, masks_dir = _setup_base_scene(tmp_path)

    def _bad_backend(_r: Any, _m: Any, state: GaussianState, _roi: np.ndarray) -> GaussianState:
        mutated = state.colors.copy()
        mutated[:] = 250
        return GaussianState(points_xyz=state.points_xyz, colors=mutated, opacity=state.opacity, scales=state.scales, quats=state.quats)

    with pytest.raises(StageExecutionError, match="mutated frozen background Gaussians"):
        rebuild_scene(
            RebuildRequest(
                base_dir=base_dir, edited_frames_dir=edited_dir, masks_dir=masks_dir, output_dir=tmp_path / "fail",
                project_id="proj-1", version_id="v2_bad", parent_version_id="v1", mode="masked_finetune",
            ),
            backend=_bad_backend,
        )


def test_from_scratch_reinitializes_from_sparse_points_and_missing_sparse_fails(tmp_path: Path) -> None:
    base_dir, edited_dir, _ = _setup_base_scene(tmp_path, with_points_bin=True)
    out_scratch = tmp_path / "v2_scratch"
    seen_counts: list[int] = []

    def _record_backend(_r: Any, _m: Any, state: GaussianState, _roi: np.ndarray) -> GaussianState:
        seen_counts.append(len(state.points_xyz))
        return state

    metrics = rebuild_scene(
        RebuildRequest(
            base_dir=base_dir, edited_frames_dir=edited_dir, masks_dir=tmp_path / "empty_masks",
            output_dir=out_scratch, project_id="proj-1", version_id="v2_scratch", parent_version_id="v1",
            mode="from_scratch",
        ),
        backend=_record_backend,
    )
    assert seen_counts == [2]
    assert metrics["total_gaussian_count"] == 2
    assert metrics["steps"] == 15000
    assert metrics["frozen_gaussian_count"] == 0

    no_sparse, ed2, m2 = _setup_base_scene(tmp_path / "ns", with_sparse=False)
    with pytest.raises(StageExecutionError, match="sparse/0"):
        rebuild_scene(
            RebuildRequest(
                base_dir=no_sparse, edited_frames_dir=ed2, masks_dir=m2,
                output_dir=tmp_path / "v2_ns", project_id="proj-1", version_id="v2_ns", parent_version_id="v1",
            ),
            backend=lambda _r, _m, state, _roi: state,
        )


def test_unedited_or_excluded_frames_fallback_to_original_frames_with_zero_mask(tmp_path: Path) -> None:
    base_dir, edited_dir, masks_dir = _setup_base_scene(tmp_path)
    (edited_dir / "00001.png").unlink()
    orig_dir = base_dir / "frames"
    orig_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((128, 160, 3), 42, dtype=np.uint8)).save(orig_dir / "00001.png", format="PNG")
    manifest = load_manifest(base_dir / "manifest.json")
    views = _load_training_views(manifest, edited_dir, masks_dir, orig_frames_dir=orig_dir, skip_frames={1})
    assert len(views) == 2
    assert np.allclose(views[0][1][1:, 1:], 240.0 / 255.0, atol=1e-3)
    assert np.allclose(views[0][1][0, 0], np.array([13, 77, 211], dtype=np.float32) / 255.0, atol=1e-3)
    assert views[0][2] is not None and views[0][2].any()
    assert np.allclose(views[1][1], 42.0 / 255.0, atol=1e-3)
    assert views[1][2] is not None and not views[1][2].any()

    # Unregistered frame (cam_from_world=None) must be skipped cleanly across ROI + view loading + rebuild.
    unreg_manifest = manifest.model_copy(
        update={"frames": [*manifest.frames, manifest.frames[1].model_copy(update={"frame_idx": 2, "colmap_image_id": None, "cam_from_world": None})]}
    )
    assert len(_load_training_views(unreg_manifest, edited_dir, masks_dir, orig_frames_dir=orig_dir)) == 2


def test_sync_colmap_images_bin_names_rejects_malformed_inputs(tmp_path: Path) -> None:
    frames = [{"colmap_image_id": 1, "image_relpath": "images/frame_000000.png"}]
    with pytest.raises(StageExecutionError, match="file missing"):
        _sync_colmap_images_bin_names(tmp_path / "missing.bin", frames)

    short_path = tmp_path / "short.bin"
    short_path.write_bytes(b"\x01\x00\x00")
    with pytest.raises(StageExecutionError, match="short count"):
        _sync_colmap_images_bin_names(short_path, frames)

    trunc_hdr = tmp_path / "trunc_hdr.bin"
    trunc_hdr.write_bytes(struct.pack("<Q", 1) + b"\x00" * 10)
    with pytest.raises(StageExecutionError, match="truncated image header"):
        _sync_colmap_images_bin_names(trunc_hdr, frames)

    valid_hdr = struct.pack("<I4d3dI", 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)
    unterm = tmp_path / "unterm.bin"
    unterm.write_bytes(struct.pack("<Q", 1) + valid_hdr + b"frame_0.jpg")
    with pytest.raises(StageExecutionError, match="unterminated filename"):
        _sync_colmap_images_bin_names(unterm, frames)

    bad_utf8 = tmp_path / "bad_utf8.bin"
    bad_utf8.write_bytes(struct.pack("<Q", 1) + valid_hdr + b"\xff\xfe\x00" + struct.pack("<Q", 0))
    with pytest.raises(StageExecutionError, match="Malformed COLMAP images.bin"):
        _sync_colmap_images_bin_names(bad_utf8, frames)

    trunc_cnt = tmp_path / "trunc_cnt.bin"
    trunc_cnt.write_bytes(struct.pack("<Q", 1) + valid_hdr + b"a.jpg\x00\x01\x00")
    with pytest.raises(StageExecutionError, match="truncated num_pts2d"):
        _sync_colmap_images_bin_names(trunc_cnt, frames)

    trunc_pts = tmp_path / "trunc_pts.bin"
    trunc_pts.write_bytes(struct.pack("<Q", 1) + valid_hdr + b"a.jpg\x00" + struct.pack("<Q", 1) + b"\x00" * 8)
    with pytest.raises(StageExecutionError, match="truncated 2D point tracks"):
        _sync_colmap_images_bin_names(trunc_pts, frames)

    trailing = tmp_path / "trailing.bin"
    trailing.write_bytes(struct.pack("<Q", 1) + valid_hdr + b"a.jpg\x00" + struct.pack("<Q", 0) + b"\xde\xad")
    with pytest.raises(StageExecutionError, match="trailing bytes"):
        _sync_colmap_images_bin_names(trailing, frames)


def test_masked_l1_loss_ignores_background_pixels() -> None:
    rendered = np.zeros((16, 16, 3), dtype=np.float32)
    target = np.zeros((16, 16, 3), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 255
    rendered[4:8, 4:8] = 1.0
    target[:4, :4] = 999.0
    assert compute_masked_l1_loss(rendered, target, mask) == pytest.approx(1.0)
