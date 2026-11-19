from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from room3d_worker.common.errors import StageExecutionError
from room3d_worker.common.manifest import SceneManifest
from room3d_worker.pipeline import s07_rebuild as rebuild
from room3d_worker.pipeline.s04_export import normalize_quats

if TYPE_CHECKING:
    from torch import Tensor


@pytest.fixture
def state() -> rebuild.GaussianState:
    rng = np.random.default_rng(42)
    return rebuild.GaussianState(
        points_xyz=rng.uniform(-1, 1, (128, 3)).astype(np.float32),
        colors=rng.integers(100, 150, (128, 3), dtype=np.uint8),
        opacity=np.full(128, 0.8, dtype=np.float32),
        scales=np.full((128, 3), 1e-7, dtype=np.float32),
        quats=normalize_quats(rng.normal(size=(128, 4)).astype(np.float32), 128),
    )


@pytest.fixture
def manifest() -> SceneManifest:
    return SceneManifest.model_validate({
        "schema_version": "1.0.0", "project_id": "room", "version_id": "v1",
        "coord_convention": "opencv_colmap", "T_viewer_from_colmap": np.eye(4).tolist(),
        "camera_model": "PINHOLE", "width": 64, "height": 64,
        "intrinsics": {"fx": 64, "fy": 64, "cx": 32, "cy": 32}, "test_every": 0,
        "frames": [{"frame_idx": 0, "orig_video_frame": 0, "timestamp_sec": 0,
                    "image_relpath": "images/0.png", "cam_from_world": np.eye(4).tolist()}],
        "artifacts": {"sparse_dir": "sparse/0", "spz_relpath": "scene.spz",
                      "sparse_points_relpath": "sparse_points.json"},
    })


@pytest.mark.parametrize(("mode", "safeguards"), [("masked_finetune", False), ("masked_finetune", True), ("from_scratch", False)])
@pytest.mark.parametrize("scale", [1e-7, 0.02])
def test_optimizer_preserves_frozen_values_through_checkpoint_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: rebuild.GaussianState,
    manifest: SceneManifest, mode: rebuild.RebuildMode, scale: float, safeguards: bool,
) -> None:
    pytest.importorskip("torch")
    state.scales[:] = scale
    roi = np.arange(len(state.points_xyz)) % 2 == 0
    if mode == "from_scratch":
        roi[:] = True
    original = {name: getattr(state, name).copy() for name in state.__dataclass_fields__ if name != "extra"}

    def rasterize(*, colors: Tensor, width: int, height: int, **_kwargs: object) -> tuple[Tensor]:
        return (colors[0].view(1, 1, 1, 3).expand(1, height, width, 3),)

    views = [(np.eye(4), np.full((64, 64, 3), 0.9, np.float32), np.ones((64, 64), bool))]
    monkeypatch.setitem(sys.modules, "gsplat", SimpleNamespace(rasterization=rasterize))
    monkeypatch.setattr(rebuild, "_load_training_views", lambda *_args: views)
    request = rebuild.RebuildRequest(
        base_dir=tmp_path, edited_frames_dir=tmp_path, masks_dir=tmp_path, output_dir=tmp_path,
        project_id="room", version_id="v2", parent_version_id="v1", mode=mode,
        device="cpu", fine_tune_steps=3, from_scratch_steps=3,
        rebuild_safeguards=safeguards,
    )
    updated = rebuild._default_gpu_rebuild(request, manifest, state, roi)
    assert rebuild.assert_frozen_gaussians_unchanged(state, updated, ~roi) == 0.0
    assert np.any(updated.colors[roi] != state.colors[roi])
    assert np.all(updated.scales[roi] > 0)
    assert np.all(updated.scales[roi] <= np.maximum(state.scales[roi], 1e-6) * 2.0001)
    rebuild.save_gaussian_state(tmp_path, updated)
    with np.load(tmp_path / "gs_state.npz", allow_pickle=False) as saved:
        for name, values in original.items():
            np.testing.assert_array_equal(getattr(state, name), values)
            np.testing.assert_array_equal(getattr(updated, name)[~roi], values[~roi])
            np.testing.assert_array_equal(saved[name][~roi], values[~roi])
    import torch

    checkpoint = torch.load(tmp_path / "gs_ckpt.pt", map_location="cpu", weights_only=True)
    np.testing.assert_array_equal(checkpoint["quats"].numpy()[~roi], original["quats"][~roi])


@pytest.mark.parametrize("name", ["points_xyz", "colors", "opacity", "scales", "quats"])
def test_frozen_guard_rejects_actual_changes(state: rebuild.GaussianState, name: str) -> None:
    changed = getattr(state, name).copy()
    if name == "quats":
        changed[0] *= 2
    elif name == "colors":
        changed[0, 0] += 1
    else:
        changed.flat[0] = np.nextafter(changed.flat[0], np.float32(np.inf))
    updated = replace(state, **{name: changed})
    frozen = np.ones(len(state.points_xyz), dtype=bool)
    with pytest.raises(StageExecutionError, match="mutated frozen background Gaussians"):
        rebuild.assert_frozen_gaussians_unchanged(state, updated, frozen)


def test_near_camera_pruning_preserves_frozen_and_visible_opaque_splats(
    tmp_path: Path, manifest: SceneManifest,
) -> None:
    points = np.array([[0, 0, .2], [0, 0, .2], [0, 0, 2], [1, 0, .2], [0, 0, -.2], [0, 0, .2]], np.float32)
    state = rebuild.GaussianState(points, np.full((6, 3), 128, np.uint8),
                                  np.array([.01, .01, .01, .01, .01, .9], np.float32),
                                  np.full((6, 3), .01, np.float32), normalize_quats(None, 6))
    request = rebuild.RebuildRequest(tmp_path, tmp_path, tmp_path, tmp_path, "room", "v2", "v1",
                                      rebuild_safeguards=True, device="cpu")
    updated, count = rebuild.prune_near_camera_floaters(state, manifest, np.array([True, False, True, True, True, True]), request)
    assert count == 1
    np.testing.assert_array_equal(updated.points_xyz, state.points_xyz[1:])
    np.testing.assert_array_equal(updated.opacity, state.opacity[1:])


def test_safeguard_bounds_use_foreground_depth_and_keep_weight_at_least_one(
    tmp_path: Path, manifest: SceneManifest, state: rebuild.GaussianState,
) -> None:
    from PIL import Image

    masks = tmp_path / "masks"
    depths = tmp_path / "depths"
    masks.mkdir()
    depths.mkdir()
    mask = np.zeros((64, 64), np.uint8)
    mask[28:36, 28:36] = 255
    Image.fromarray(mask).save(masks / "000000.png")
    np.save(depths / "000000.npy", np.ones((64, 64), np.float32))
    request = rebuild.RebuildRequest(tmp_path, tmp_path, masks, tmp_path, "room", "v2", "v1", device="cpu")
    assert request.rebuild_safeguards is False
    assert (request.fine_tune_steps, request.from_scratch_steps) == (5000, 15000)
    lower, upper = rebuild.foreground_bounds(request, manifest, state, np.ones(128, bool))
    assert np.max(np.abs(lower[:2])) < .1 and np.max(np.abs(upper[:2])) < .1
    assert lower[2] < 1 < upper[2]
    with pytest.raises(ValueError, match="unmasked_weight"):
        replace(request, unmasked_weight=.5)
    with pytest.raises(ValueError, match="masked_finetune"):
        replace(request, rebuild_safeguards=True, mode="from_scratch")
