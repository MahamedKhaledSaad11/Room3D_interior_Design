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


@pytest.mark.parametrize("mode", ["masked_finetune", "from_scratch"])
@pytest.mark.parametrize("scale", [1e-7, 0.02])
def test_optimizer_preserves_frozen_values_through_checkpoint_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: rebuild.GaussianState,
    manifest: SceneManifest, mode: rebuild.RebuildMode, scale: float,
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
    )
    updated = rebuild._default_gpu_rebuild(request, manifest, state, roi)
    assert rebuild.assert_frozen_gaussians_unchanged(state, updated, ~roi) == 0.0
    assert np.any(updated.colors[roi] != state.colors[roi])
    assert np.all(updated.scales[roi] >= 1e-5)
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
