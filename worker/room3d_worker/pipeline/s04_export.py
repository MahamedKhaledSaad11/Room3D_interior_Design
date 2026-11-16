from __future__ import annotations

import gzip
import importlib
import json
import shutil
import struct
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from room3d_worker.common.errors import GeometryError, StageExecutionError
from room3d_worker.common.manifest import SceneManifest, save_manifest

ArrayF32 = NDArray[np.float32]
ArrayU8 = NDArray[np.uint8]

SH_C0 = 0.28209479177387814
SPZ_MAGIC = 0x5053474E
SPZ_VERSION = 2
SPZ_FRACTIONAL_BITS = 12
# Niantic's packer maps SH DC coefficients into uint8 with this fixed gain.
SPZ_COLOR_SCALE = 0.15
_OPACITY_EPS = 1e-6


@dataclass(frozen=True)
class ExportRequest:
    output_dir: Path
    project_id: str = "project"
    version_id: str = "version"
    parent_version_id: str | None = None
    width: int = 640
    height: int = 480
    intrinsics: dict[str, float] | None = None
    frames: list[dict[str, Any]] | None = None
    test_every: int = 0
    pose_backend: str | None = None
    T_viewer_from_colmap: list[list[float]] | None = None
    max_spz_bytes: int = 30 * 1024 * 1024


@dataclass(frozen=True)
class SplatArrays:
    """Export-ready Gaussians: linear (N,3) scales, wxyz unit quats, linear opacity, uint8 RGB."""

    points: ArrayF32
    colors: ArrayU8
    opacity: ArrayF32
    scales: ArrayF32
    quats: ArrayF32


def normalize_scales(scales: NDArray[Any], count: int) -> ArrayF32:
    """Accept isotropic (N,) or anisotropic (N,3) linear scales and return (N,3)."""
    size = np.asarray(scales, dtype=np.float32)
    if size.ndim == 1 or (size.ndim == 2 and size.shape[1] == 1):
        size = np.repeat(size.reshape(-1, 1), 3, axis=1)
    if size.shape != (count, 3):
        raise GeometryError(f"scales must be shaped (N,) or (N, 3) with N={count}, got {size.shape}")
    return size


def normalize_quats(quats: NDArray[Any] | None, count: int) -> ArrayF32:
    """Return unit wxyz quaternions; degenerate rows collapse to identity."""
    if quats is None:
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1))
    q = np.asarray(quats, dtype=np.float32)
    if q.shape != (count, 4):
        raise GeometryError(f"quats must be shaped (N, 4) wxyz with N={count}, got {q.shape}")
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    safe = np.where(norms > 1e-12, q / np.maximum(norms, 1e-12), identity)
    return safe.astype(np.float32)


def filter_floaters(points_xyz: np.ndarray, opacity: np.ndarray, scales: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32)
    alpha = np.asarray(opacity, dtype=np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(alpha):
        raise ValueError("points, opacity, and scales must have the same Gaussian count")
    try:
        size = normalize_scales(scales, len(points))
    except GeometryError as exc:
        raise ValueError("points, opacity, and scales must have the same Gaussian count") from exc
    finite = np.isfinite(points).all(1) & np.isfinite(alpha) & np.isfinite(size).all(1)
    radius = np.linalg.norm(size, axis=1)
    return finite & (alpha >= 0.01) & (alpha <= 1.0) & (radius >= 1e-5) & (radius <= 10.0)


def _log_scales(scales: ArrayF32) -> ArrayF32:
    # Linear scales can arrive signed from upstream log-space helpers; magnitude is the extent.
    return np.log(np.clip(np.abs(scales), 1e-8, None)).astype(np.float32)


def _sh_dc(colors: ArrayU8) -> ArrayF32:
    return ((colors.astype(np.float32) / 255.0 - 0.5) / SH_C0).astype(np.float32)


def _logit(alpha: ArrayF32) -> ArrayF32:
    clipped = np.clip(alpha, _OPACITY_EPS, 1.0 - _OPACITY_EPS)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


_PLY_FLOAT_FIELDS = (
    "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
)
PLY_DTYPE = np.dtype(
    [(name, "<f4") for name in _PLY_FLOAT_FIELDS[:9]]
    + [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    + [(name, "<f4") for name in _PLY_FLOAT_FIELDS[9:]]
)


def _write_ply(path: Path, splats: SplatArrays) -> None:
    """Standard 3DGS PLY (logit opacity, log scales, wxyz rot) plus uchar RGB for PLYLoader."""
    vertices = np.zeros(len(splats.points), dtype=PLY_DTYPE)
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = splats.points[:, axis]
    dc, log_scale = _sh_dc(splats.colors), _log_scales(splats.scales)
    for axis in range(3):
        vertices[f"f_dc_{axis}"] = dc[:, axis]
        vertices[f"scale_{axis}"] = log_scale[:, axis]
    for axis, name in enumerate(("red", "green", "blue")):
        vertices[name] = splats.colors[:, axis]
    for axis in range(4):
        vertices[f"rot_{axis}"] = splats.quats[:, axis]
    vertices["opacity"] = _logit(splats.opacity)
    properties = "".join(
        f"property {'uchar' if PLY_DTYPE[name].kind == 'u' else 'float'} {name}\n" for name in PLY_DTYPE.names or ()
    )
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\n{properties}end_header\n"
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices.tobytes())


def _to_u8(values: NDArray[Any]) -> ArrayU8:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _pack_positions(points: ArrayF32) -> bytes:
    limit = (1 << 23) - 1
    fixed = np.rint(points.astype(np.float64) * (1 << SPZ_FRACTIONAL_BITS)).astype(np.int64)
    if np.abs(fixed).max(initial=0) > limit:
        raise GeometryError("positions exceed the SPZ 24-bit fixed-point range (|x| < 2048 at 12 bits)")
    as_u32 = (fixed & 0xFFFFFF).astype("<u4").reshape(-1, 1).view(np.uint8)
    return as_u32[:, :3].tobytes()


def _pack_rotations(quats: ArrayF32) -> bytes:
    # SPZ v2 stores xyz of an xyzw quaternion whose w is forced non-negative.
    xyz = quats[:, 1:4] * np.where(quats[:, :1] < 0.0, -1.0, 1.0)
    return _to_u8(xyz * 127.5 + 127.5).tobytes()


def pack_spz_v2(splats: SplatArrays) -> bytes:
    """Encode an uncompressed SPZ v2 payload (shDegree 0) per nianticlabs/spz."""
    header = struct.pack("<IIIBBBB", SPZ_MAGIC, SPZ_VERSION, len(splats.points), 0, SPZ_FRACTIONAL_BITS, 0, 0)
    alphas = _to_u8(splats.opacity * 255.0)
    colors = _to_u8(_sh_dc(splats.colors) * (SPZ_COLOR_SCALE * 255.0) + 127.5)
    scales = _to_u8((_log_scales(splats.scales) + 10.0) * 16.0)
    return b"".join(
        (header, _pack_positions(splats.points), alphas.tobytes(), colors.tobytes(), scales.tobytes(),
         _pack_rotations(splats.quats))
    )


def _write_fallback_spz(path: Path, splats: SplatArrays) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as handle:
            handle.write(pack_spz_v2(splats))


def read_spz_header(path: Path) -> dict[str, int]:
    with gzip.open(path, "rb") as handle:
        raw = handle.read(16)
    if len(raw) != 16:
        raise StageExecutionError(f"SPZ header truncated: {path}")
    magic, version, count, sh_degree, frac_bits, flags, reserved = struct.unpack("<IIIBBBB", raw)
    return {"magic": magic, "version": version, "num_points": count, "sh_degree": sh_degree,
            "fractional_bits": frac_bits, "flags": flags, "reserved": reserved}


def _write_native_spz(path: Path, splats: SplatArrays) -> None:
    spz = importlib.import_module("spz")
    cloud = spz.GaussianCloud()
    cloud.numPoints = len(splats.points)
    cloud.shDegree = 0
    cloud.positions = splats.points.reshape(-1).tolist()
    cloud.scales = _log_scales(splats.scales).reshape(-1).tolist()
    cloud.rotations = splats.quats[:, [1, 2, 3, 0]].reshape(-1).tolist()
    cloud.alphas = _logit(splats.opacity).tolist()
    cloud.colors = _sh_dc(splats.colors).reshape(-1).tolist()
    options = spz.PackOptions()
    options.version = SPZ_VERSION
    if not spz.save_spz(cloud, options, str(path)):
        raise OSError("spz.save_spz reported failure")


def _write_splat_transform_spz(ply_path: Path, spz_path: Path) -> bool:
    cli = shutil.which("splat-transform")
    if cli is None or not ply_path.exists():
        return False
    completed = subprocess.run([cli, str(ply_path), str(spz_path)], check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not spz_path.exists():
        return False
    header = read_spz_header(spz_path)
    return header["magic"] == SPZ_MAGIC and header["version"] in (2, 3)


def _write_spz(path: Path, splats: SplatArrays, ply_path: Path | None = None) -> None:
    """Prefer native spz, then splat-transform CLI, else built-in SPZ v2 packer."""
    try:
        _write_native_spz(path, splats)
        header = read_spz_header(path)
        if header["magic"] == SPZ_MAGIC and header["version"] in (2, 3):
            return
        raise ValueError(f"native spz produced unsupported header {header}")
    except (ImportError, AttributeError, TypeError, ValueError, OSError, StageExecutionError) as exc:
        if ply_path is not None and _write_splat_transform_spz(ply_path, path):
            return
        warnings.warn(f"Native spz writer unavailable ({exc}); using built-in SPZ v2 packer", RuntimeWarning, stacklevel=2)
    _write_fallback_spz(path, splats)


_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
              "int": "<i4", "int32": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}


def _ply_vertex_dtype(header: str, path: Path) -> np.dtype[Any]:
    if "format binary_little_endian" not in header:
        raise StageExecutionError(f"Only binary_little_endian PLY is supported: {path}")
    fields: list[tuple[str, str]] = []
    for line in header.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "property":
            if parts[1] not in _PLY_TYPES:
                raise StageExecutionError(f"Unsupported PLY property type '{parts[1]}' in {path}")
            fields.append((parts[2], _PLY_TYPES[parts[1]]))
    return np.dtype(fields)


def _ply_column(data: NDArray[Any], names: tuple[str, ...]) -> ArrayF32 | None:
    if not all(name in (data.dtype.names or ()) for name in names):
        return None
    return np.stack([data[name] for name in names], axis=1).astype(np.float32)


def read_ply_splats(path: Path) -> SplatArrays:
    """Read a 3DGS PLY (logit opacity, log scales, wxyz rot) or a plain xyz+rgb point cloud."""
    raw = path.read_bytes()
    marker = b"end_header\n"
    idx = raw.find(marker)
    if idx < 0:
        raise StageExecutionError(f"Malformed PLY header: {path}")
    data = np.frombuffer(raw[idx + len(marker):], dtype=_ply_vertex_dtype(raw[:idx].decode("ascii", "replace"), path))
    count = len(data)
    points = _ply_column(data, ("x", "y", "z"))
    if points is None:
        raise StageExecutionError(f"PLY lacks x/y/z vertex properties: {path}")
    rgb = _ply_column(data, ("red", "green", "blue"))
    dc = _ply_column(data, ("f_dc_0", "f_dc_1", "f_dc_2"))
    colors = rgb if rgb is not None else (dc * SH_C0 + 0.5) * 255.0 if dc is not None else np.full((count, 3), 200.0)
    log_scale = _ply_column(data, ("scale_0", "scale_1", "scale_2"))
    logit = _ply_column(data, ("opacity",))
    return SplatArrays(
        points=points,
        colors=_to_u8(colors),
        opacity=np.full(count, 0.9, np.float32) if logit is None else (1.0 / (1.0 + np.exp(-logit[:, 0]))).astype(np.float32),
        scales=np.full((count, 3), 0.01, np.float32) if log_scale is None else np.exp(log_scale).astype(np.float32),
        quats=normalize_quats(_ply_column(data, ("rot_0", "rot_1", "rot_2", "rot_3")), count),
    )


def _default_frames() -> list[dict[str, Any]]:
    return [{"frame_idx": 0, "orig_video_frame": 0, "timestamp_sec": 0.0,
             "image_relpath": "images/0.png", "colmap_image_id": 1,
             "cam_from_world": np.eye(4).tolist()}]


def _prepare_splats(points_xyz: np.ndarray, opacity: np.ndarray, scales: np.ndarray,
                    colors: np.ndarray | None, quats: np.ndarray | None) -> SplatArrays:
    count = len(np.asarray(points_xyz))
    keep = filter_floaters(points_xyz, opacity, scales)
    rgb = np.asarray(colors if colors is not None else np.full((count, 3), 200), np.uint8)
    return SplatArrays(
        points=np.asarray(points_xyz, np.float32)[keep],
        colors=rgb[keep],
        opacity=np.asarray(opacity, np.float32).reshape(-1)[keep],
        scales=normalize_scales(scales, count)[keep],
        quats=normalize_quats(quats, count)[keep],
    )


def _write_sparse_points(path: Path, splats: SplatArrays) -> None:
    step = max(1, len(splats.points) // 100_000)
    sparse = [
        {"position": [float(v) for v in pt], "x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2]),
         "color": [int(v) for v in clr]}
        for pt, clr in zip(splats.points[::step], splats.colors[::step])
    ]
    path.write_text(json.dumps(sparse, separators=(",", ":")) + "\n", encoding="utf-8")


def export_scene(request: ExportRequest, points_xyz: np.ndarray, opacity: np.ndarray,
                 scales: np.ndarray, colors: np.ndarray | None = None,
                 quats: np.ndarray | None = None) -> SceneManifest:
    out = request.output_dir
    out.mkdir(parents=True, exist_ok=True)
    splats = _prepare_splats(points_xyz, opacity, scales, colors, quats)
    ply_path = out / "scene.ply"
    _write_ply(ply_path, splats)
    _write_spz(out / "scene.spz", splats, ply_path=ply_path)
    spz_bytes = (out / "scene.spz").stat().st_size
    if spz_bytes >= request.max_spz_bytes:
        warnings.warn("scene.spz exceeds the 30 MB viewer budget", RuntimeWarning, stacklevel=2)
    _write_sparse_points(out / "sparse_points.json", splats)
    intrinsic = request.intrinsics or {"fx": 500.0, "fy": 500.0, "cx": request.width / 2, "cy": request.height / 2}
    manifest = SceneManifest.model_validate({"schema_version": "1.0.0", "project_id": request.project_id,
        "version_id": request.version_id, "parent_version_id": request.parent_version_id,
        "coord_convention": "opencv_colmap",
        "T_viewer_from_colmap": request.T_viewer_from_colmap or np.eye(4).tolist(), "camera_model": "PINHOLE",
        "width": request.width, "height": request.height, "intrinsics": intrinsic, "test_every": request.test_every,
        "pose_backend": request.pose_backend, "frames": request.frames or _default_frames(),
        "artifacts": {"sparse_dir": "sparse/0", "ply_relpath": "scene.ply", "spz_relpath": "scene.spz",
                       "sparse_points_relpath": "sparse_points.json", "ckpt_relpath": "gs_ckpt.pt"},
        "metrics": {"gaussian_count": int(len(splats.points)), "spz_bytes": spz_bytes}})
    save_manifest(out / "manifest.json", manifest)
    return manifest


def export_splat_artifacts(
    request: ExportRequest,
    points_xyz: np.ndarray | None = None,
    opacity: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    colors: np.ndarray | None = None,
    quats: np.ndarray | None = None,
    ckpt_path: Path | None = None,
) -> SceneManifest:
    """Export from explicit arrays, or from a gs_state.npz / gs_ckpt.pt / .ply checkpoint."""
    if ckpt_path is not None:
        from room3d_worker.common.gaussian_state import load_checkpoint

        state = load_checkpoint(Path(ckpt_path))
        rgb = _to_u8(np.asarray(state.sh0, dtype=np.float32) * 255.0)
        return export_scene(request, state.xyz, state.opacities, state.scales, rgb, state.quats)
    if points_xyz is None or opacity is None or scales is None:
        raise GeometryError("export_splat_artifacts needs points_xyz, opacity and scales, or a ckpt_path")
    return export_scene(request, points_xyz, opacity, scales, colors, quats)
