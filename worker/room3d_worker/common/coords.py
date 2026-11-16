from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from room3d_worker.common.errors import GeometryError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Sim3Transform:
    """Similarity transform X_dst = scale * (rotation @ X_src) + translation."""

    scale: float
    rotation: FloatArray
    translation: FloatArray

    def __post_init__(self) -> None:
        scale_val = float(self.scale)
        rot = np.asarray(self.rotation, dtype=np.float64)
        trans = np.asarray(self.translation, dtype=np.float64)
        if not np.isfinite(scale_val) or scale_val <= 0.0:
            raise GeometryError(f"Sim3Transform scale must be finite and positive, got {scale_val}")
        if rot.shape != (3, 3) or not np.isfinite(rot).all():
            raise GeometryError("Sim3Transform rotation must be a finite 3x3 matrix")
        if trans.shape != (3,) or not np.isfinite(trans).all():
            raise GeometryError("Sim3Transform translation must be a finite 3-vector")
        object.__setattr__(self, "scale", scale_val)
        object.__setattr__(self, "rotation", rot)
        object.__setattr__(self, "translation", trans)

    def apply(self, points_xyz: np.ndarray) -> FloatArray:
        pts = np.asarray(points_xyz, dtype=np.float64)
        if pts.shape == (3,):
            if not np.isfinite(pts).all():
                raise GeometryError("points_xyz contains non-finite values")
            return self.scale * (self.rotation @ pts) + self.translation
        checked = _array(pts, "points_xyz", (3,))
        return (self.scale * (self.rotation @ checked.T)).T + self.translation


def _array(value: NDArray[Any], name: str, shape_tail: tuple[int, ...]) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != len(shape_tail) + 1 or result.shape[1:] != shape_tail:
        raise GeometryError(f"{name} must have shape (N, {', '.join(map(str, shape_tail))})")
    if not np.isfinite(result).all():
        raise GeometryError(f"{name} contains non-finite values")
    return result


def _rotation_between(source: FloatArray, target: FloatArray) -> FloatArray:
    source, target = source / np.linalg.norm(source), target / np.linalg.norm(target)
    cross = np.cross(source, target)
    cosine = float(np.dot(source, target))
    if np.linalg.norm(cross) < 1e-10:
        if cosine > 0:
            return np.eye(3)
        axis = np.cross(source, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-10:
            axis = np.cross(source, np.array([0.0, 0.0, 1.0]))
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / np.dot(cross, cross))


def _floor_normal(points: FloatArray) -> FloatArray:
    if len(points) < 3:
        raise GeometryError("At least three points are required to estimate room up")
    rng = np.random.default_rng(0)
    best: tuple[int, FloatArray, float] | None = None
    tolerance = max(float(np.linalg.norm(np.ptp(points, axis=0))) * 0.01, 1e-6)
    for indices in rng.integers(0, len(points), size=(min(256, len(points) * 8), 3)):
        a, b, c = points[indices]
        normal = np.cross(b - a, c - a)
        magnitude = float(np.linalg.norm(normal))
        if magnitude < 1e-10:
            continue
        normal /= magnitude
        offset = float(np.dot(normal, a))
        inliers = int(np.count_nonzero(np.abs(points @ normal - offset) <= tolerance))
        if best is None or inliers > best[0]:
            best = inliers, normal, offset
    if best is not None and best[0] >= max(3, int(len(points) * 0.15)):
        return best[1]
    centered = points - points.mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    return vectors[-1]


def align_gravity_up(points_xyz: np.ndarray, cam_positions_xyz: np.ndarray) -> np.ndarray:
    points = _array(points_xyz, "points_xyz", (3,))
    cameras = _array(cam_positions_xyz, "cam_positions_xyz", (3,))
    normal = _floor_normal(points)
    if len(cameras) and float(np.dot(normal, cameras.mean(axis=0) - points.mean(axis=0))) < 0:
        normal = -normal
    rotation = _rotation_between(normal, np.array([0.0, 1.0, 0.0]))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ points.mean(axis=0)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8) or np.linalg.det(rotation) < 0.999999:
        raise GeometryError("Gravity alignment did not produce a proper rigid rotation")
    return transform


def _intrinsic(intrinsics: dict[str, float]) -> tuple[float, float, float, float]:
    try:
        fx, fy, cx, cy = (float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy"))
    except (KeyError, TypeError, ValueError) as exc:
        raise GeometryError(f"Invalid PINHOLE intrinsics: {exc}") from exc
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise GeometryError("Focal lengths must be positive and all intrinsics finite")
    return fx, fy, cx, cy


def _pose(matrix: np.ndarray) -> FloatArray:
    pose = np.asarray(matrix, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise GeometryError("cam_from_world must be a finite 4x4 matrix")
    return pose


def project_world_to_image(point_world: np.ndarray, cam_from_world: np.ndarray, intrinsics: dict[str, float]) -> tuple[float, float, float]:
    point = np.asarray(point_world, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise GeometryError("point_world must be a finite 3-vector")
    camera = _pose(cam_from_world) @ np.append(point, 1.0)
    depth = float(camera[2])
    if depth <= 1e-6:
        raise GeometryError(f"Point is behind camera or too near (depth={depth})")
    fx, fy, cx, cy = _intrinsic(intrinsics)
    return fx * camera[0] / depth + cx, fy * camera[1] / depth + cy, depth


def unproject_pixel_to_ray(u: float, v: float, cam_from_world: np.ndarray, intrinsics: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    pose = _pose(cam_from_world)
    fx, fy, cx, cy = _intrinsic(intrinsics)
    direction_camera = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    direction_world = pose[:3, :3].T @ direction_camera
    direction_world /= np.linalg.norm(direction_world)
    origin_world = -pose[:3, :3].T @ pose[:3, 3]
    return origin_world, direction_world


def mod16_crop_or_pad_params(width: int, height: int, cx: float, cy: float, fx: float, fy: float, multiple: int = 16) -> tuple[int, int, float, float, float, float, tuple[int, int, int, int]]:
    if min(width, height, multiple) <= 0 or fx <= 0 or fy <= 0:
        raise GeometryError("Dimensions, multiple, and focal lengths must be positive")
    new_w, new_h = width - width % multiple, height - height % multiple
    if new_w == 0 or new_h == 0:
        raise GeometryError("Image dimensions are smaller than the requested multiple")
    x0, y0 = (width - new_w) // 2, (height - new_h) // 2
    return new_w, new_h, fx, fy, cx - x0, cy - y0, (x0, y0, x0 + new_w, y0 + new_h)


def umeyama_sim3(source_xyz: np.ndarray, target_xyz: np.ndarray) -> Sim3Transform:
    """Estimate Sim(3) mapping source_xyz -> target_xyz via Umeyama's closed-form SVD."""
    source = _array(source_xyz, "source_xyz", (3,))
    target = _array(target_xyz, "target_xyz", (3,))
    if len(source) < 3 or source.shape != target.shape:
        raise GeometryError("Sim(3) alignment requires equal point sets of length >= 3")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    x = source - source_mean
    y = target - target_mean
    covariance = (y.T @ x) / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    correction[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(x * x, axis=1)))
    if variance <= 1e-12:
        raise GeometryError("Sim(3) source points have zero variance")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 0.0:
        raise GeometryError(f"Sim(3) estimated invalid scale {scale}")
    translation = target_mean - scale * (rotation @ source_mean)
    return Sim3Transform(scale=scale, rotation=rotation, translation=translation)


def transform_cam_from_world_sim3(cam_from_world: np.ndarray, sim3: Sim3Transform) -> FloatArray:
    """Transform a 4x4 world-to-camera pose under world Sim(3) X_dst = s * R_sim @ X_src + t_sim."""
    pose = _pose(cam_from_world)
    r_w2c_src = pose[:3, :3]
    t_w2c_src = pose[:3, 3]
    r_c2w_src = r_w2c_src.T
    c_src = -(r_c2w_src @ t_w2c_src)

    c_dst = sim3.scale * (sim3.rotation @ c_src) + sim3.translation
    r_c2w_dst = sim3.rotation @ r_c2w_src
    u, _, vt = np.linalg.svd(r_c2w_dst.T)
    if np.linalg.det(u @ vt) < 0.0:
        u[:, -1] *= -1.0
    r_w2c_dst = u @ vt

    transformed = np.eye(4, dtype=np.float64)
    transformed[:3, :3] = r_w2c_dst
    transformed[:3, 3] = -(r_w2c_dst @ c_dst)
    return transformed
