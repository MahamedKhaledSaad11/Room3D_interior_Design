from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from room3d_worker.common.errors import StageExecutionError
from room3d_worker.pipeline.edit_validation import image_acceptance
from room3d_worker.pipeline.recolor import recolor_rgb
from room3d_worker.pipeline.s03_train import split_evaluation_views
from room3d_worker.pipeline.s07_rebuild import (
    RebuildRequest, compute_roi_gaussian_mask, geometry_metrics, load_base_scene, rebuild_scene,
)
from scene_fixtures import _setup_base_scene


def test_recolor_changes_appearance_without_optimization_or_geometric_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir, edited_dir, masks_dir = _setup_base_scene(tmp_path)
    from room3d_worker.pipeline import s07_rebuild

    def forbidden_optimizer(*args: object) -> None:
        raise AssertionError("Paint must not invoke a GPU optimizer or diffusion")

    monkeypatch.setattr(s07_rebuild, "_default_gpu_rebuild", forbidden_optimizer)
    base_manifest, base = load_base_scene(base_dir)
    request = RebuildRequest(base_dir, edited_dir, masks_dir, tmp_path / "paint", "proj-1", "paint", "v1",
                             device="cpu", operation="recolor", anchor_frames=(0,))
    metrics = rebuild_scene(request)
    with np.load(request.output_dir / "roi_selection.npz", allow_pickle=False) as selected:
        assert selected["selected_indices"].tolist() == [0, 3]
    assert metrics["steps"] == 0
    manifest, result = load_base_scene(request.output_dir)
    for name in ("points_xyz", "scales", "opacity", "quats"):
        np.testing.assert_array_equal(getattr(base, name), getattr(result, name))
    assert result.colors[0, 2] > result.colors[0, 0]
    np.testing.assert_array_equal(result.colors[1:3], base.colors[1:3])
    assert manifest.frames[0].cam_from_world == base_manifest.frames[0].cam_from_world
    assert metrics["quality_acceptance"]["status"] == "review_required"
    original = np.asarray(Image.open(base_dir / base_manifest.frames[0].image_relpath).convert("RGB"))
    painted = np.asarray(Image.open(request.output_dir / manifest.frames[0].image_relpath).convert("RGB"))
    mask = np.asarray(Image.open(masks_dir / "00000.png")) == 255
    np.testing.assert_array_equal(original[~mask], painted[~mask])


def test_roi_rejects_surface_behind_mask_and_low_alpha_depth(tmp_path: Path) -> None:
    base_dir, _, masks_dir = _setup_base_scene(tmp_path)
    manifest, _ = load_base_scene(base_dir)
    depth_dir = tmp_path / "trusted_depth"
    depth_dir.mkdir()
    for frame in manifest.frames:
        np.save(depth_dir / f"{frame.frame_idx:06d}.npy", np.full((128, 160), 4, dtype=np.float32))
        np.save(depth_dir / f"{frame.frame_idx:06d}_alpha.npy", np.ones((128, 160), dtype=np.float32))
    points = np.array([[0, 0, 4], [0, 0, 8]], dtype=np.float32)
    roi = compute_roi_gaussian_mask(points, manifest, masks_dir, min_hits=2, roi_pad_px=0,
                                    depth_dir=depth_dir, anchor_frames=(0,))
    assert roi.tolist() == [True, False]
    np.save(depth_dir / "000000_alpha.npy", np.zeros((128, 160), dtype=np.float32))
    roi = compute_roi_gaussian_mask(points, manifest, masks_dir, min_hits=2, depth_dir=depth_dir)
    assert not roi.any()


def test_roi_requires_anchor_and_consensus_in_visible_views(tmp_path: Path) -> None:
    base_dir, _, masks_dir = _setup_base_scene(tmp_path)
    manifest, state = load_base_scene(base_dir)
    Image.fromarray(np.zeros((128, 160), dtype=np.uint8)).save(masks_dir / "00000.png")
    roi = compute_roi_gaussian_mask(state.points_xyz, manifest, masks_dir, min_hits=1, anchor_frames=(0,))
    assert not roi.any()


def test_corrupted_extent_or_recolor_geometry_is_rejected(tmp_path: Path) -> None:
    base_dir, _, _ = _setup_base_scene(tmp_path)
    _, state = load_base_scene(base_dir)
    with pytest.raises(StageExecutionError, match="scale growth"):
        geometry_metrics(state, replace(state, scales=state.scales * 33), False)
    with pytest.raises(StageExecutionError, match="Recolor changed"):
        geometry_metrics(state, replace(state, opacity=state.opacity * 0.99), True)


def test_render_acceptance_rejects_background_corruption_and_noop() -> None:
    before = np.full((64, 64, 3), 0.6, dtype=np.float32)
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:40] = True
    target = before.copy()
    target[mask] = [0.1, 0.2, 0.8]
    assert image_acceptance(before, target, target, mask)["passed"]
    corrupted = target.copy()
    corrupted[:10] = 0
    assert not image_acceptance(before, corrupted, target, mask)["passed"]
    assert not image_acceptance(before, before, target, mask)["passed"]


def test_recolor_keeps_ordered_shading_and_one_transform_across_batches() -> None:
    rgb = np.array([[40, 40, 40], [120, 120, 120], [220, 220, 220]], dtype=np.uint8)
    painted = recolor_rgb(rgb, "#2563eb")
    assert np.all(np.diff(painted[:, 2].astype(int)) > 0)
    assert np.all(painted[:, 2] > painted[:, 0])
    np.testing.assert_array_equal(painted, np.concatenate([recolor_rgb(rgb[:1], "#2563eb"), recolor_rgb(rgb[1:], "#2563eb")]))


def test_evaluation_split_keeps_camera_records_and_training_disjoint() -> None:
    views = [{"frame_idx": idx} for idx in range(17)]
    train, evaluate = split_evaluation_views(views, 8)
    assert [view["frame_idx"] for view in evaluate] == [0, 8, 16]
    assert not {view["frame_idx"] for view in train} & {view["frame_idx"] for view in evaluate}
    assert len(train) + len(evaluate) == len(views)
