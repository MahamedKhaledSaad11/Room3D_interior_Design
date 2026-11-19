from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from room3d_worker.common.gaussian_state import GaussianState
from room3d_worker.common.manifest import load_manifest
from room3d_worker.pipeline.s03_train import build_train_config, initialize_scales_knn, radial_outlier_mask
from room3d_worker.pipeline.s04_export import (
    SPZ_MAGIC,
    ExportRequest,
    export_scene,
    export_splat_artifacts,
    read_ply_splats,
    read_spz_header,
)


def test_knn_initialization_does_not_use_pairwise_cdist() -> None:
    points = np.random.default_rng(0).normal(size=(2_000, 3))
    scales = initialize_scales_knn(points)
    assert scales.shape == (2_000,)
    assert np.isfinite(scales).all()


def test_radial_pruning_retains_dense_extreme_wall() -> None:
    rng = np.random.default_rng(1)
    room = rng.uniform(-1.0, 1.0, size=(500, 3))
    wall = np.column_stack((np.full(100, 20.0), rng.uniform(-2, 2, (100, 2))))
    isolated = np.array([[1000.0, 1000.0, 1000.0]])
    keep = radial_outlier_mask(np.vstack((room, wall, isolated)))
    assert keep[500:600].all()
    assert not keep[-1]


def test_train_config_defaults_to_production_holdout_and_cuda0(tmp_path: Path) -> None:
    config = build_train_config(tmp_path)
    assert config.test_every == 0
    assert config.device == "cuda:0"


def test_export_writes_anisotropic_ply_and_spec_spz_v2(tmp_path: Path) -> None:
    points = np.array([[0.1, 0.2, 1.0], [1.0, -0.5, 1.5], [-0.3, 0.4, 2.0], [9.0, 9.0, 9.0]], dtype=np.float32)
    opacity = np.array([0.95, 0.8, 0.6, 0.001], dtype=np.float32)
    scales = np.array([[0.05, 0.1, 0.2], [0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.1, 0.1, 0.1]], dtype=np.float32)
    quats = np.array([[0.0, 1.0, 0.0, 0.0], [0.707, 0.0, 0.707, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[20, 120, 220], [200, 40, 80], [90, 180, 30], [0, 0, 0]], dtype=np.uint8)

    manifest = export_scene(ExportRequest(tmp_path), points, opacity, scales, colors=colors, quats=quats)
    assert manifest.artifacts.spz_relpath == "scene.spz"
    assert len(json.loads((tmp_path / "sparse_points.json").read_text())) == 3
    assert load_manifest(tmp_path / "manifest.json").project_id == "project"

    ply = read_ply_splats(tmp_path / "scene.ply")
    assert ply.scales.shape == (3, 3)
    assert np.allclose(ply.scales, scales[:3], atol=1e-4)
    assert np.allclose(ply.opacity, opacity[:3], atol=1e-4)
    assert np.allclose(np.abs(np.sum(ply.quats * quats[:3], axis=1)), 1.0, atol=1e-3)

    spz_path = tmp_path / "scene.spz"
    header = read_spz_header(spz_path)
    assert header["magic"] == SPZ_MAGIC
    assert header["version"] == 2
    assert header["num_points"] == 3
    assert header["sh_degree"] == 0
    assert header["fractional_bits"] == 12
    with gzip.open(spz_path, "rb") as handle:
        raw = handle.read()
    # 16 header + 3*(9 pos + 1 alpha + 3 color + 3 scale + 3 rot)
    assert len(raw) == 16 + 3 * (9 + 1 + 3 + 3 + 3)


def test_export_splat_artifacts_from_isotropic_scales_and_ckpt(tmp_path: Path) -> None:
    iso_dir = tmp_path / "iso"
    pts = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]], dtype=np.float32)
    op = np.array([0.8, 0.9], dtype=np.float32)
    iso_scales = np.array([0.05, 0.1], dtype=np.float32)
    m_iso = export_splat_artifacts(ExportRequest(iso_dir), points_xyz=pts, opacity=op, scales=iso_scales)
    assert m_iso.metrics is not None and m_iso.metrics["gaussian_count"] == 2
    assert read_ply_splats(iso_dir / "scene.ply").scales.shape == (2, 3)

    state = GaussianState(
        xyz=pts,
        scales=np.array([[0.03, 0.06, 0.09], [0.08, 0.04, 0.02]], dtype=np.float32),
        quats=np.array([[1.0, 0.0, 0.0, 0.0], [0.7071, 0.7071, 0.0, 0.0]], dtype=np.float32),
        opacities=op,
        sh0=np.array([[0.2, 0.4, 0.8], [0.9, 0.1, 0.3]], dtype=np.float32),
    )
    ckpt_path = state.save_npz(tmp_path / "gs_state.npz")
    ckpt_dir = tmp_path / "from_ckpt"
    export_splat_artifacts(ExportRequest(ckpt_dir), ckpt_path=ckpt_path)
    loaded = read_ply_splats(ckpt_dir / "scene.ply")
    assert np.allclose(loaded.scales, state.scales, atol=1e-4)
    assert read_spz_header(ckpt_dir / "scene.spz")["magic"] == pytest.approx(SPZ_MAGIC)
