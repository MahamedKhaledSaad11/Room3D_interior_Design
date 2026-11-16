from __future__ import annotations

import importlib
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from room3d_worker.common.errors import GeometryError, StageExecutionError

Float32Array = NDArray[np.float32]
_SH_C0 = 0.28209479177387814

_PLY_DTYPE_MAP: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _normalize_quats(quats: NDArray[np.float32]) -> Float32Array:
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    result = np.where(norms > 1e-8, quats / np.maximum(norms, 1e-8), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    return result.astype(np.float32, copy=False)


def _sigmoid(values: NDArray[Any]) -> Float32Array:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _identity_quats(count: int) -> Float32Array:
    quats = np.zeros((count, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    return quats


@dataclass(slots=True)
class GaussianState:
    """Canonical 3D Gaussian Splatting parameter bundle shared across pipeline stages."""

    xyz: Float32Array
    scales: Float32Array
    quats: Float32Array
    opacities: Float32Array
    sh0: Float32Array
    sh_rest: Float32Array | None = None

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float32)
        self.scales = np.asarray(self.scales, dtype=np.float32)
        self.quats = np.asarray(self.quats, dtype=np.float32)
        self.opacities = np.asarray(self.opacities, dtype=np.float32)
        self.sh0 = np.asarray(self.sh0, dtype=np.float32)
        if self.sh_rest is not None:
            self.sh_rest = np.asarray(self.sh_rest, dtype=np.float32)

    def validate(self) -> GaussianState:
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise GeometryError(f"GaussianState.xyz must have shape (N, 3), got {self.xyz.shape}")
        count = len(self.xyz)
        for name, arr, tail in (
            ("scales", self.scales, (3,)),
            ("quats", self.quats, (4,)),
            ("opacities", self.opacities, ()),
            ("sh0", self.sh0, (3,)),
        ):
            expected = (count, *tail)
            if arr.shape != expected:
                raise GeometryError(f"GaussianState.{name} must have shape {expected}, got {arr.shape}")
            if not np.isfinite(arr).all():
                raise GeometryError(f"GaussianState.{name} contains non-finite values")
        if not np.isfinite(self.xyz).all():
            raise GeometryError("GaussianState.xyz contains non-finite values")
        if count > 0:
            if not (self.scales > 0.0).all():
                raise GeometryError("GaussianState.scales must be strictly positive")
            norms = np.linalg.norm(self.quats, axis=1)
            if not (norms > 1e-6).all() or not np.allclose(norms, 1.0, atol=1e-3):
                raise GeometryError("GaussianState.quats must be unit wxyz quaternions")
            if not ((self.opacities >= 0.0).all() and (self.opacities <= 1.0).all()):
                raise GeometryError("GaussianState.opacities must lie in [0, 1]")
            if not ((self.sh0 >= 0.0).all() and (self.sh0 <= 1.0).all()):
                raise GeometryError("GaussianState.sh0 RGB values must lie in [0, 1]")
        if self.sh_rest is not None:
            if self.sh_rest.ndim != 3 or self.sh_rest.shape[0] != count or self.sh_rest.shape[2] != 3:
                raise GeometryError(
                    f"GaussianState.sh_rest must have shape ({count}, K, 3), got {self.sh_rest.shape}"
                )
            if not np.isfinite(self.sh_rest).all():
                raise GeometryError("GaussianState.sh_rest contains non-finite values")
        return self

    def save_npz(self, path: str | Path) -> Path:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "xyz": self.xyz,
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
            "sh0": self.sh0,
        }
        if self.sh_rest is not None:
            payload["sh_rest"] = self.sh_rest
        fd, temp_name = tempfile.mkstemp(prefix="gs_state.", suffix=".npz", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(handle, **payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        except OSError:
            Path(temp_name).unlink(missing_ok=True)
            raise
        return target

    @classmethod
    def load_npz(cls, path: str | Path) -> GaussianState:
        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            keys = set(archive.files)
            if "xyz" in keys:
                xyz = np.asarray(archive["xyz"], dtype=np.float32)
                scales = np.asarray(archive["scales"], dtype=np.float32)
                quats = _normalize_quats(np.asarray(archive["quats"], dtype=np.float32))
                opacities = np.asarray(archive["opacities"], dtype=np.float32).reshape(-1)
                sh0 = np.clip(np.asarray(archive["sh0"], dtype=np.float32), 0.0, 1.0)
                sh_rest = np.asarray(archive["sh_rest"], dtype=np.float32) if "sh_rest" in keys else None
                return cls(xyz=xyz, scales=scales, quats=quats, opacities=opacities, sh0=sh0, sh_rest=sh_rest).validate()
            if "points_xyz" in keys:
                xyz = np.asarray(archive["points_xyz"], dtype=np.float32)
                count = len(xyz)
                raw_scales = np.asarray(archive["scales"], dtype=np.float32)
                scales = np.repeat(raw_scales[:, None], 3, axis=1) if raw_scales.ndim == 1 else raw_scales
                color_key = next((k for k in ("colors", "sh0", "rgb") if k in keys), None)
                if color_key is None:
                    raise GeometryError(f"NPZ archive at {source} is missing colors/sh0/rgb")
                raw_colors = np.asarray(archive[color_key])
                sh0 = raw_colors.astype(np.float32) / 255.0 if np.issubdtype(raw_colors.dtype, np.integer) or raw_colors.max(initial=0.0) > 1.0 else raw_colors.astype(np.float32)
                opacity_key = "opacity" if "opacity" in keys else ("opacities" if "opacities" in keys else None)
                opacities = (
                    np.asarray(archive[opacity_key], dtype=np.float32).reshape(-1)
                    if opacity_key is not None
                    else np.full((count,), 0.5, dtype=np.float32)
                )
                quats = (
                    _normalize_quats(np.asarray(archive["quats"], dtype=np.float32))
                    if "quats" in keys
                    else _identity_quats(count)
                )
                return cls(
                    xyz=xyz,
                    scales=scales,
                    quats=quats,
                    opacities=np.clip(opacities, 0.0, 1.0),
                    sh0=np.clip(sh0, 0.0, 1.0),
                    sh_rest=None,
                ).validate()
        raise GeometryError(f"NPZ archive at {source} is missing canonical GaussianState keys")

    def save_pt(self, path: str | Path) -> Path:
        self.validate()
        torch = importlib.import_module("torch")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "xyz": torch.from_numpy(self.xyz.copy()),
            "scales": torch.from_numpy(self.scales.copy()),
            "quats": torch.from_numpy(self.quats.copy()),
            "opacities": torch.from_numpy(self.opacities.copy()),
            "sh0": torch.from_numpy(self.sh0.copy()),
        }
        if self.sh_rest is not None:
            payload["sh_rest"] = torch.from_numpy(self.sh_rest.copy())
        torch.save(payload, target)
        return target

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> GaussianState:
        source = Path(path)
        suffix = source.suffix.lower()
        if suffix == ".npz":
            return cls.load_npz(source)
        if suffix in {".pt", ".pth"}:
            return _load_pt_checkpoint(source)
        if suffix == ".ply":
            return _load_ply_checkpoint(source)
        raise StageExecutionError(f"Unsupported Gaussian checkpoint format: {source.suffix}")

    @classmethod
    def from_sparse_points(cls, xyz: np.ndarray, rgb: np.ndarray) -> GaussianState:
        points = np.asarray(xyz, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise GeometryError(f"xyz must have shape (N, 3), got {points.shape}")
        if not np.isfinite(points).all():
            raise GeometryError("xyz contains non-finite values")
        raw_rgb = np.asarray(rgb)
        if raw_rgb.shape != points.shape:
            raise GeometryError(f"rgb must have shape {points.shape}, got {raw_rgb.shape}")
        if np.issubdtype(raw_rgb.dtype, np.integer):
            sh0 = np.clip(raw_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
        else:
            rgb_f32 = raw_rgb.astype(np.float32)
            if not np.isfinite(rgb_f32).all():
                raise GeometryError("rgb contains non-finite values")
            sh0 = np.clip(rgb_f32, 0.0, 1.0)
        count = len(points)
        if count == 0:
            scales = np.empty((0, 3), dtype=np.float32)
        elif count == 1:
            scales = np.full((1, 3), 0.01, dtype=np.float32)
        else:
            k = min(3, count - 1)
            distances = cKDTree(points.astype(np.float64)).query(points.astype(np.float64), k=k + 1, workers=1)[0][:, 1:]
            iso = np.median(distances, axis=1).clip(min=1e-6).astype(np.float32)
            scales = np.repeat(iso[:, None], 3, axis=1)
        return cls(
            xyz=points,
            scales=scales,
            quats=_identity_quats(count),
            opacities=np.full((count,), 0.5, dtype=np.float32),
            sh0=sh0.astype(np.float32, copy=False),
            sh_rest=None,
        ).validate()


def _to_numpy_cpu(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy())
    return np.asarray(value)


def _load_pt_checkpoint(path: Path) -> GaussianState:
    torch = importlib.import_module("torch")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, GaussianState):
        return ckpt.validate()
    if not isinstance(ckpt, dict):
        raise GeometryError(f"Expected dict checkpoint in {path}, got {type(ckpt).__name__}")
    xyz_key = next((k for k in ("xyz", "points_xyz", "means") if k in ckpt), None)
    if xyz_key is None:
        raise GeometryError(f"Checkpoint {path} is missing xyz tensor")
    xyz = _to_numpy_cpu(ckpt[xyz_key]).astype(np.float32)
    count = len(xyz)

    if "scales" in ckpt or "log_scales" in ckpt:
        scale_key = "scales" if "scales" in ckpt else "log_scales"
        raw_scales = _to_numpy_cpu(ckpt[scale_key]).astype(np.float32)
        if raw_scales.ndim == 1:
            raw_scales = np.repeat(raw_scales[:, None], 3, axis=1)
        if scale_key == "log_scales" or (count > 0 and np.any(raw_scales <= 0.0)):
            scales = np.exp(np.clip(raw_scales, -20.0, 20.0)).astype(np.float32)
        else:
            scales = raw_scales
    else:
        scales = np.full((count, 3), 0.01, dtype=np.float32)

    quat_key = next((k for k in ("quats", "quaternions", "rotations") if k in ckpt), None)
    if quat_key is not None:
        quats = _normalize_quats(_to_numpy_cpu(ckpt[quat_key]).astype(np.float32))
    else:
        quats = _identity_quats(count)

    op_key = next((k for k in ("opacities", "opacity") if k in ckpt), None)
    if op_key is not None:
        raw_op = _to_numpy_cpu(ckpt[op_key]).astype(np.float32).reshape(-1)
        opacities = _sigmoid(raw_op) if (count > 0 and np.any((raw_op < 0.0) | (raw_op > 1.0))) else raw_op
    else:
        opacities = np.full((count,), 0.5, dtype=np.float32)

    color_key = next((k for k in ("sh0", "rgb", "colors") if k in ckpt), None)
    if color_key is not None:
        raw_color = _to_numpy_cpu(ckpt[color_key])
        if np.issubdtype(raw_color.dtype, np.integer) or raw_color.max(initial=0.0) > 1.0:
            sh0 = np.clip(raw_color.astype(np.float32) / 255.0, 0.0, 1.0)
        else:
            sh0 = np.clip(raw_color.astype(np.float32), 0.0, 1.0)
    elif "f_dc" in ckpt:
        sh0 = np.clip(_to_numpy_cpu(ckpt["f_dc"]).astype(np.float32) * _SH_C0 + 0.5, 0.0, 1.0)
    else:
        sh0 = np.full((count, 3), 0.5, dtype=np.float32)

    sh_rest = _to_numpy_cpu(ckpt["sh_rest"]).astype(np.float32) if ckpt.get("sh_rest") is not None else None
    return GaussianState(xyz=xyz, scales=scales, quats=quats, opacities=opacities, sh0=sh0, sh_rest=sh_rest).validate()


def _parse_ply_header(raw: bytes) -> tuple[str, int, list[tuple[str, str]], int]:
    lf = raw.find(b"end_header\n")
    crlf = raw.find(b"end_header\r\n")
    if lf >= 0 and (crlf < 0 or lf < crlf):
        header_end, body_start = lf, lf + len(b"end_header\n")
    elif crlf >= 0:
        header_end, body_start = crlf, crlf + len(b"end_header\r\n")
    else:
        raise StageExecutionError("Malformed PLY file: missing end_header")
    lines = raw[:header_end].decode("ascii", errors="strict").splitlines()
    if not lines or lines[0].strip() != "ply":
        raise StageExecutionError("Malformed PLY file: missing magic header")
    fmt = "binary_little_endian"
    vertex_count = 0
    in_vertex = False
    properties: list[tuple[str, str]] = []
    for line in lines[1:]:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "format" and len(parts) >= 2:
            fmt = parts[1]
        elif parts[0] == "element":
            in_vertex = len(parts) >= 3 and parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif parts[0] == "property" and in_vertex and len(parts) >= 3 and parts[1] != "list":
            properties.append((parts[1], parts[2]))
    return fmt, vertex_count, properties, body_start


def _load_ply_checkpoint(path: Path) -> GaussianState:
    raw = path.read_bytes()
    fmt, count, properties, body_start = _parse_ply_header(raw)
    if fmt == "binary_little_endian":
        endian = "<"
    elif fmt == "binary_big_endian":
        endian = ">"
    elif fmt == "ascii":
        endian = ""
    else:
        raise StageExecutionError(f"Unsupported PLY format {fmt} in {path}")

    names = [name for _, name in properties]
    if fmt == "ascii":
        text = raw[body_start:].decode("ascii").splitlines()[:count]
        dtype = np.dtype([(name, _PLY_DTYPE_MAP[ptype]) for ptype, name in properties])
        data = np.empty(count, dtype=dtype)
        for row_idx, row in enumerate(text):
            vals = row.split()
            for col_idx, (ptype, name) in enumerate(properties):
                cast = int if _PLY_DTYPE_MAP[ptype][0] in {"i", "u"} else float
                data[name][row_idx] = cast(vals[col_idx])
    else:
        dtype = np.dtype([(name, endian + _PLY_DTYPE_MAP[ptype]) for ptype, name in properties])
        expected_bytes = count * dtype.itemsize
        payload = raw[body_start : body_start + expected_bytes]
        if len(payload) < expected_bytes:
            raise StageExecutionError(f"Truncated PLY body in {path}")
        data = np.frombuffer(payload, dtype=dtype, count=count)

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32) if count else np.empty((0, 3), dtype=np.float32)
    name_set = set(names)
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(name_set):
        f_dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1).astype(np.float32)
        sh0 = np.clip(f_dc * _SH_C0 + 0.5, 0.0, 1.0)
    elif {"red", "green", "blue"}.issubset(name_set):
        rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1)
        sh0 = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0) if np.issubdtype(rgb.dtype, np.integer) or rgb.max(initial=0.0) > 1.0 else np.clip(rgb.astype(np.float32), 0.0, 1.0)
    else:
        sh0 = np.full((count, 3), 0.5, dtype=np.float32)

    if {"scale_0", "scale_1", "scale_2"}.issubset(name_set):
        log_scales = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1).astype(np.float32)
        scales = np.exp(np.clip(log_scales, -20.0, 20.0)).astype(np.float32)
    else:
        scales = np.full((count, 3), 0.01, dtype=np.float32)

    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(name_set):
        quats = _normalize_quats(np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1).astype(np.float32))
    else:
        quats = _identity_quats(count)

    has_3dgs = bool({"f_dc_0", "scale_0", "rot_0"} & name_set)
    if "opacity" in name_set:
        raw_op = np.asarray(data["opacity"], dtype=np.float32).reshape(-1)
        opacities = _sigmoid(raw_op) if (has_3dgs or (count > 0 and np.any((raw_op < 0.0) | (raw_op > 1.0)))) else raw_op
    else:
        opacities = np.full((count,), 0.5, dtype=np.float32)

    rest_names = sorted((n for n in names if n.startswith("f_rest_")), key=lambda s: int(s.split("_")[-1]))
    sh_rest: Float32Array | None = None
    if rest_names and len(rest_names) % 3 == 0:
        k = len(rest_names) // 3
        stacked = np.stack([data[n] for n in rest_names], axis=1).astype(np.float32)
        sh_rest = stacked.reshape(count, 3, k).transpose(0, 2, 1).copy()

    return GaussianState(xyz=xyz, scales=scales, quats=quats, opacities=opacities, sh0=sh0, sh_rest=sh_rest).validate()


load_npz = GaussianState.load_npz
load_checkpoint = GaussianState.load_checkpoint
from_sparse_points = GaussianState.from_sparse_points
