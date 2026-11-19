from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from room3d_worker.common.errors import GeometryError
from room3d_worker.pipeline import s04_export as export


def _splats(points: NDArray[np.float32]) -> export.SplatArrays:
    count = len(points)
    angles = np.arange(count, dtype=np.float32) * 0.1
    return export.SplatArrays(
        points=points,
        colors=np.arange(count * 3, dtype=np.uint8).reshape(count, 3),
        opacity=np.linspace(0.02, 0.9, count, dtype=np.float32),
        scales=np.linspace(0.01, 0.1, count * 3, dtype=np.float32).reshape(count, 3),
        quats=np.column_stack((np.cos(angles), np.sin(angles), np.zeros((count, 2)))).astype(np.float32),
    )


@pytest.mark.parametrize(
    ("coordinate", "expected_bits"),
    [(1.25, 12), (-2048.0, 12), (2047.999755859375, 12), (2047.9998779296875, 11),
     (2048.0, 11), (4096.0, 10), (-4096.0, 11), (16384.0, 8), (-16384.0, 9),
     (32768.0, 7), (8388607.0, 0), (-8388608.0, 0)],
)
def test_spz_precision_matches_header_and_signed_positions(coordinate: float, expected_bits: int) -> None:
    points = np.array([[coordinate, -0.375, 0.125]], dtype=np.float32)
    raw = export.pack_spz_v2(_splats(points))
    magic, version, count, degree, bits, flags, reserved = struct.unpack("<IIIBBBB", raw[:16])
    assert (magic, version, count, degree, flags, reserved) == (export.SPZ_MAGIC, 2, 1, 0, 0, 0)
    assert bits == expected_bits
    packed = np.frombuffer(raw[16:25], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    fixed = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
    signed = np.where(fixed & 0x800000, fixed - 0x1000000, fixed)
    decoded = signed.reshape(points.shape) / (1 << bits)
    np.testing.assert_allclose(decoded, points, atol=0.5 / (1 << bits), rtol=0)


@pytest.mark.parametrize("coordinate", [np.nan, np.inf, -np.inf, 8388608.0, -8388609.0, 1e30])
def test_spz_rejects_positions_that_cannot_be_encoded(coordinate: float) -> None:
    with pytest.raises(GeometryError, match="finite|24-bit range"):
        export.pack_spz_v2(_splats(np.array([[coordinate, 0, 0]], dtype=np.float32)))


def test_spz_empty_cloud_keeps_default_precision() -> None:
    raw = export.pack_spz_v2(_splats(np.empty((0, 3), dtype=np.float32)))
    assert len(raw) == 16
    assert struct.unpack("<IIIBBBB", raw)[2:5] == (0, 0, 12)


def test_export_prunes_relocated_floaters_from_all_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    room = np.random.default_rng(42).uniform(-1, 1, size=(100, 3)).astype(np.float32)
    wall = np.array([[20, 0, 0]], dtype=np.float32)
    floaters = np.array([[0, 500, 0], [1024, 0, 0], [0, 0, -1024], [1e20, 0, 0]], dtype=np.float32)
    splats = _splats(np.vstack((room[:50], floaters, room[50:], wall)))
    expected = np.r_[0:50, 54:105]

    def unavailable_writer(path: Path, arrays: export.SplatArrays) -> None:
        raise ModuleNotFoundError("test without optional SPZ writer")

    monkeypatch.setattr(export, "_write_native_spz", unavailable_writer)
    monkeypatch.setattr(export.shutil, "which", lambda command: None)
    with pytest.warns(RuntimeWarning, match="built-in SPZ v2 packer"):
        manifest = export.export_scene(
            export.ExportRequest(tmp_path), splats.points, splats.opacity,
            splats.scales, splats.colors, splats.quats,
        )
    ply = export.read_ply_splats(tmp_path / "scene.ply")
    for field in ("points", "colors", "opacity", "scales", "quats"):
        np.testing.assert_allclose(getattr(ply, field), getattr(splats, field)[expected], atol=1e-6)
    header = export.read_spz_header(tmp_path / "scene.spz")
    assert header["num_points"] == len(expected)
    assert header["fractional_bits"] == 12
    with gzip.open(tmp_path / "scene.spz", "rb") as handle:
        assert handle.read() == export.pack_spz_v2(export.SplatArrays(
            **{field: getattr(splats, field)[expected] for field in splats.__dataclass_fields__}
        ))
    sparse = json.loads((tmp_path / "sparse_points.json").read_text())
    np.testing.assert_array_equal([point["position"] for point in sparse], splats.points[expected])
    assert manifest.metrics is not None and manifest.metrics["gaussian_count"] == len(expected)


@pytest.mark.parametrize(
    ("points", "expected"),
    [([[0, 0, 0]] * 10 + [[1, 0, 0]], [True] * 10 + [False]),
     ([[1024, 0, 0], [-1024, 0, 0]], [False, False]),
     ([[np.nan, 0, 0], [np.inf, 0, 0], [0, 0, 0]], [False, False, True]),
     ([], [])],
)
def test_spatial_filter_handles_degenerate_clouds(points: list[list[float]], expected: list[bool]) -> None:
    splats = _splats(np.asarray(points, dtype=np.float32).reshape(-1, 3))
    keep = export.filter_floaters(splats.points, splats.opacity, splats.scales)
    np.testing.assert_array_equal(keep, expected)
