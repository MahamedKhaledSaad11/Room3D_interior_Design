from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pytest

from room3d_worker.common.manifest import save_manifest
from room3d_worker.pipeline.s03_train import (
    GaussianModelState,
    TrainConfig,
    build_train_config,
    initialize_scales_knn,
    radial_outlier_mask,
    train_scene,
)


def _sample_manifest(width: int = 64, height: int = 64) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "project_id": "proj",
        "version_id": "v1",
        "coord_convention": "opencv_colmap",
        "T_viewer_from_colmap": np.eye(4).tolist(),
        "camera_model": "PINHOLE",
        "width": width,
        "height": height,
        "intrinsics": {"fx": 64.0, "fy": 64.0, "cx": width / 2.0, "cy": height / 2.0},
        "test_every": 0,
        "frames": [
            {
                "frame_idx": 0,
                "orig_video_frame": 0,
                "timestamp_sec": 0.0,
                "image_relpath": "images/00000.png",
                "colmap_image_id": 1,
                "cam_from_world": np.eye(4).tolist(),
            }
        ],
        "artifacts": {"sparse_dir": "sparse/0", "spz_relpath": "scene.spz", "sparse_points_relpath": "sparse_points.json"},
    }


def test_train_config_allows_small_and_clamps_large_cap_max(tmp_path: Path) -> None:
    small = build_train_config(tmp_path, cap_max=100, iterations=5)
    assert small.device == "cuda:0"
    assert small.cap_max == 100

    clamped = build_train_config(tmp_path, cap_max=5_000_000)
    assert clamped.cap_max == 1_500_000

    with pytest.raises(ValueError, match="cap_max"):
        TrainConfig(output_dir=tmp_path, cap_max=0)


def test_knn_and_gaussian_model_state_hold_full_3dgs_attributes() -> None:
    pts = np.random.default_rng(0).normal(size=(32, 3)).astype(np.float32)
    scales = initialize_scales_knn(pts)
    assert scales.shape == (32,)
    state = GaussianModelState.from_sparse(pts, np.full((32, 3), 0.25, dtype=np.float32))
    assert state.scales.shape == (32, 3)
    assert state.quats.shape == (32, 4)
    assert np.allclose(state.quats[:, 0], 1.0)
    assert state.opacities.shape == (32,)
    assert state.sh0.shape == (32, 3)
    assert radial_outlier_mask(np.vstack([pts, [[500.0, 500.0, 500.0]]]))[-1] == np.bool_(False)


def test_train_scene_synthetic_fallback_saves_full_state_and_depths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gsplat", None)
    pts = np.random.default_rng(2).uniform(-1.0, 1.0, size=(20, 3)).astype(np.float32)
    imgs = np.zeros((1, 16, 16, 3), dtype=np.float32)
    depths = np.ones((1, 16, 16), dtype=np.float32) * 2.5
    telemetry = train_scene(build_train_config(tmp_path, cap_max=50, allow_synthetic_fallback=True), imgs, pts, depth_maps=depths)
    assert telemetry.gaussian_count == 20
    with np.load(tmp_path / "gs_state.npz", allow_pickle=False) as archive:
        assert archive["scales"].shape == (20, 3)
        assert archive["quats"].shape == (20, 4)
        assert archive["opacities"].shape == (20,)
        assert archive["sh0"].shape == (20, 3)
    assert (tmp_path / "depths" / "00000.npy").exists()

    uint8_pts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float32)
    uint8_rgb = np.array([[50, 100, 200], [10, 20, 30], [200, 150, 100], [128, 64, 32]], dtype=np.uint8)
    uint8_dir = tmp_path / "uint8_rgb"
    train_scene(build_train_config(uint8_dir, cap_max=50, allow_synthetic_fallback=True), imgs, uint8_pts, points_rgb=uint8_rgb)
    with np.load(uint8_dir / "gs_state.npz", allow_pickle=False) as archive:
        assert archive["sh0"].max() < 0.9
        assert np.allclose(archive["sh0"][0], np.array([50, 100, 200], dtype=np.float32) / 255.0, atol=1e-5)

    with pytest.raises(Exception, match="strict mode"):
        train_scene(build_train_config(tmp_path / "strict"), imgs, pts)


def test_train_scene_runs_gsplat_optimization_and_persists_depths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    from room3d_worker.common.manifest import SceneManifest

    manifest = SceneManifest.model_validate(_sample_manifest())
    save_manifest(tmp_path / "manifest.json", manifest)
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8)).save(img_dir / "00000.png")

    class _FakeMCMCStrategy:
        def __init__(self, cap_max: int) -> None:
            self.cap_max = cap_max
            self.steps: list[int] = []
            self.lrs: list[float] = []

        def initialize_state(self) -> dict[str, Any]:
            return {"initialized": True}

        def step_pre_backward(self, *_args: Any) -> None:
            pass

        def step_post_backward(
            self, params: dict[str, Any], _o: Any, _s: Any, step: int, _i: Any, packed: bool = False, lr: float = 0.0
        ) -> None:
            _ = packed
            self.steps.append(step)
            self.lrs.append(lr)
            # Simulate MCMC densification replacing tensors inside params dict.
            if step == 1:
                for key, val in list(params.items()):
                    params[key] = torch.nn.Parameter(torch.cat([val.detach(), val.detach()[:2]], dim=0))

    def _fake_rasterize(*, colors: Any, means: Any, width: int, height: int, render_mode: str = "RGB", **_kw: Any) -> tuple[Any, ...]:
        _ = means
        if render_mode == "ED":
            return (torch.ones((1, height, width, 1), dtype=torch.float32) * 3.0,)
        rgb = colors.mean(dim=0).view(1, 1, 1, 3).expand(1, height, width, 3)
        return (rgb, torch.ones((1, height, width, 1)), {"radii": torch.ones((1, len(colors)))})

    fake_gsplat = SimpleNamespace(
        rasterization=_fake_rasterize,
        strategy=SimpleNamespace(MCMCStrategy=_FakeMCMCStrategy),
        optimizers=SimpleNamespace(SelectiveAdam=torch.optim.Adam),
    )
    monkeypatch.setitem(sys.modules, "gsplat", fake_gsplat)

    pts = np.random.default_rng(3).uniform(-0.5, 0.5, size=(12, 3)).astype(np.float32)
    cfg = build_train_config(tmp_path, device="cuda:9", iterations=3, cap_max=100)
    telemetry = train_scene(cfg, np.empty((0, 64, 64, 3), dtype=np.float32), pts, manifest=manifest)
    assert telemetry.gaussian_count == 14
    assert telemetry.psnr > 0.0
    assert (tmp_path / "gs_ckpt.pt").exists()
    assert (tmp_path / "depths" / "00000.npy").exists()
    assert np.allclose(np.load(tmp_path / "depths" / "00000.npy"), 3.0)
