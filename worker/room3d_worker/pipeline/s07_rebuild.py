from __future__ import annotations

import importlib
import json
import shutil
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from room3d_worker.common.errors import GeometryError, StageExecutionError
from room3d_worker.common.manifest import FrameRecord, SceneManifest, load_manifest
from room3d_worker.common.progress import report_progress
from room3d_worker.pipeline.s04_export import (
    ExportRequest,
    export_scene,
    normalize_quats,
    normalize_scales,
    read_ply_splats,
)
from room3d_worker.pipeline.s05_track import (
    _lift_foreground_points,
    _load_depth_map,
    load_binary_mask_png,
)

ArrayF32 = NDArray[np.float32]
ArrayBool = NDArray[np.bool_]
ArrayU8 = NDArray[np.uint8]
RebuildMode = Literal["masked_finetune", "from_scratch"]


@dataclass(frozen=True)
class RebuildRequest:
    base_dir: Path
    edited_frames_dir: Path
    masks_dir: Path
    output_dir: Path
    project_id: str
    version_id: str
    parent_version_id: str
    mode: RebuildMode = "masked_finetune"
    fine_tune_steps: int = 5000
    from_scratch_steps: int = 15000
    roi_pad_px: int = 8
    min_mask_hits: int = 1
    masked_loss_weight: float = 4.0
    exclude_frames: tuple[int, ...] = ()
    device: str = "cuda:1"
    seed: int = 0
    test_every: int = 0
    rebuild_safeguards: bool = False
    unmasked_weight: float = 1.0
    near_plane_distance: float = 0.35
    floater_opacity_threshold: float = 0.05
    bbox_pad_fraction: float = 0.1

    def __post_init__(self) -> None:
        if self.device not in {"cuda:0", "cuda:1", "cpu"}:
            raise ValueError("Week 2 3DGS rebuild device must be cuda:1, cuda:0, or cpu")
        if self.mode not in {"masked_finetune", "from_scratch"}:
            raise ValueError(f"Unsupported rebuild mode: {self.mode}")
        if self.rebuild_safeguards and self.mode != "masked_finetune":
            raise ValueError("Background-preserving safeguards require masked_finetune mode")
        if (
            self.fine_tune_steps <= 0
            or self.from_scratch_steps <= 0
            or self.roi_pad_px < 0
            or self.min_mask_hits < 1
            or self.masked_loss_weight <= 0.0
        ):
            raise ValueError("Invalid optimization step, loss weight, or ROI padding configuration")
        if (not np.isfinite([self.unmasked_weight, self.near_plane_distance, self.floater_opacity_threshold, self.bbox_pad_fraction]).all()
                or self.unmasked_weight < 1 or self.near_plane_distance <= 0
                or not 0 <= self.floater_opacity_threshold <= 1 or self.bbox_pad_fraction < 0):
            raise ValueError("Rebuild safeguards require unmasked_weight >= 1 and valid near-plane/opacity bounds")


@dataclass(frozen=True)
class GaussianState:
    points_xyz: ArrayF32
    colors: ArrayU8
    opacity: ArrayF32
    scales: ArrayF32
    quats: ArrayF32 | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def normalized_quats(self) -> ArrayF32:
        return normalize_quats(self.quats, len(self.points_xyz))

    def quats_or_identity(self) -> ArrayF32:
        return self.quats if self.quats is not None else self.normalized_quats()


def _resolve_frame_file(folder: Path, frame_idx: int, image_relpath: str | None = None) -> Path | None:
    candidates = [
        folder / f"{frame_idx:06d}.png",
        folder / f"{frame_idx:05d}.png",
        folder / f"frame_{frame_idx:06d}.png",
    ]
    if image_relpath:
        rel = Path(image_relpath)
        candidates.extend([folder / rel.name, folder / rel])
    return next((c for c in candidates if c.exists()), None)


def _state_from_common(common: Any) -> GaussianState:
    return GaussianState(
        points_xyz=np.asarray(common.xyz, dtype=np.float32),
        colors=np.clip(np.rint(np.asarray(common.sh0, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8),
        opacity=np.asarray(common.opacities, dtype=np.float32).reshape(-1),
        scales=normalize_scales(common.scales, len(common.xyz)),
        quats=normalize_quats(common.quats, len(common.xyz)),
    )


def _state_to_common(state: GaussianState) -> Any:
    from room3d_worker.common.gaussian_state import GaussianState as CommonGaussianState

    return CommonGaussianState(
        xyz=state.points_xyz.astype(np.float32),
        scales=normalize_scales(state.scales, len(state.points_xyz)),
        quats=state.quats_or_identity(),
        opacities=state.opacity.astype(np.float32).reshape(-1),
        sh0=(state.colors.astype(np.float32) / 255.0),
    )


def _load_state_from_npz(path: Path) -> GaussianState:
    with np.load(path, allow_pickle=False) as archive:
        if "quats" in archive and "points_xyz" in archive:
            xyz = np.asarray(archive["points_xyz"], dtype=np.float32)
            return GaussianState(
                points_xyz=xyz,
                colors=np.asarray(archive["colors"], dtype=np.uint8),
                opacity=np.asarray(archive["opacity"], dtype=np.float32).reshape(-1),
                scales=normalize_scales(archive["scales"], len(xyz)),
                quats=normalize_quats(archive["quats"], len(xyz)),
            )
    from room3d_worker.common.gaussian_state import load_npz

    return _state_from_common(load_npz(path))


def _load_state_from_pt(path: Path) -> GaussianState:
    try:
        from room3d_worker.common.gaussian_state import load_checkpoint

        return _state_from_common(load_checkpoint(path))
    except ImportError:
        torch = importlib.import_module("torch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        xyz = np.asarray(payload["xyz"].detach().cpu().numpy() if hasattr(payload["xyz"], "detach") else payload["xyz"], dtype=np.float32)
        raw_scales = payload["scales"].detach().cpu().numpy() if hasattr(payload["scales"], "detach") else payload["scales"]
        raw_quats = payload.get("quats")
        quats_np = raw_quats.detach().cpu().numpy() if hasattr(raw_quats, "detach") else raw_quats
        raw_op = payload.get("opacities", payload.get("opacity"))
        op_np = np.full(len(xyz), 0.9, dtype=np.float32) if raw_op is None else np.asarray(
            raw_op.detach().cpu().numpy() if hasattr(raw_op, "detach") else raw_op, dtype=np.float32
        )
        raw_col = payload.get("sh0", payload.get("rgb"))
        if raw_col is None:
            colors = np.full((len(xyz), 3), 200, dtype=np.uint8)
        else:
            col_arr = np.asarray(raw_col.detach().cpu().numpy() if hasattr(raw_col, "detach") else raw_col, dtype=np.float32)
            colors = np.clip(np.rint(col_arr * (255.0 if col_arr.max(initial=0.0) <= 1.5 else 1.0)), 0, 255).astype(np.uint8)
        return GaussianState(
            points_xyz=xyz,
            colors=colors,
            opacity=op_np.reshape(-1),
            scales=normalize_scales(raw_scales, len(xyz)),
            quats=normalize_quats(quats_np, len(xyz)),
        )


def load_base_scene(base_dir: Path) -> tuple[SceneManifest, GaussianState]:
    manifest = load_manifest(base_dir / "manifest.json")
    npz_path = base_dir / "gs_state.npz"
    if npz_path.exists():
        return manifest, _load_state_from_npz(npz_path)
    ckpt_rel = manifest.artifacts.ckpt_relpath or "gs_ckpt.pt"
    ckpt_path = base_dir / ckpt_rel
    if ckpt_path.exists():
        return manifest, _load_state_from_pt(ckpt_path)
    ply_path = base_dir / (manifest.artifacts.ply_relpath or "scene.ply")
    if not ply_path.exists():
        raise StageExecutionError(f"Base scene artifacts missing under {base_dir}")
    splats = read_ply_splats(ply_path)
    return manifest, GaussianState(
        points_xyz=splats.points,
        colors=splats.colors,
        opacity=splats.opacity,
        scales=splats.scales,
        quats=splats.quats,
    )


def save_gaussian_state(output_dir: Path, state: GaussianState) -> None:
    quats = state.quats_or_identity()
    scales = normalize_scales(state.scales, len(state.points_xyz))
    np.savez_compressed(
        output_dir / "gs_state.npz",
        points_xyz=state.points_xyz.astype(np.float32),
        colors=state.colors.astype(np.uint8),
        opacity=state.opacity.astype(np.float32),
        scales=scales,
        quats=quats,
    )
    try:
        _state_to_common(state).save_pt(output_dir / "gs_ckpt.pt")
    except ImportError:
        try:
            torch = importlib.import_module("torch")
            torch.save(
                {
                    "xyz": torch.as_tensor(state.points_xyz.astype(np.float32)),
                    "scales": torch.as_tensor(scales),
                    "quats": torch.as_tensor(quats),
                    "opacities": torch.as_tensor(state.opacity.astype(np.float32)),
                    "sh0": torch.as_tensor(state.colors.astype(np.float32) / 255.0),
                },
                output_dir / "gs_ckpt.pt",
            )
        except ImportError:
            pass


def _read_colmap_points3d_bin(path: Path) -> tuple[ArrayF32, ArrayU8]:
    """Parse COLMAP binary points3D.bin into (N, 3) xyz and (N, 3) uint8 rgb."""
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        xyz_list: list[tuple[float, float, float]] = []
        rgb_list: list[tuple[int, int, int]] = []
        for _ in range(count):
            _, x, y, z, r, g, b, _ = struct.unpack("<Q3d3Bd", handle.read(43))
            track_len = struct.unpack("<Q", handle.read(8))[0]
            handle.seek(8 * track_len, 1)
            xyz_list.append((x, y, z))
            rgb_list.append((r, g, b))
    if not xyz_list:
        raise StageExecutionError(f"points3D.bin contains zero points: {path}")
    return np.asarray(xyz_list, dtype=np.float32), np.asarray(rgb_list, dtype=np.uint8)


def _init_from_sparse_dir(sparse_dir: Path) -> GaussianState | None:
    bin_path = sparse_dir / "points3D.bin"
    npy_path = sparse_dir / "points3d_xyz.npy"
    xyz: ArrayF32 | None = None
    rgb: ArrayU8 | None = None
    if bin_path.exists() and bin_path.stat().st_size >= 8:
        xyz, rgb = _read_colmap_points3d_bin(bin_path)
    elif npy_path.exists():
        xyz = np.asarray(np.load(npy_path, allow_pickle=False), dtype=np.float32)
        rgb_path = sparse_dir / "points3d_rgb.npy"
        rgb = np.asarray(np.load(rgb_path, allow_pickle=False), dtype=np.uint8) if rgb_path.exists() else np.full((len(xyz), 3), 180, dtype=np.uint8)
    if xyz is None or len(xyz) == 0:
        return None
    assert rgb is not None
    try:
        from room3d_worker.common.gaussian_state import from_sparse_points

        return _state_from_common(from_sparse_points(xyz, rgb.astype(np.float32) / 255.0))
    except ImportError:
        if len(xyz) > 1:
            dists = cKDTree(xyz.astype(np.float64)).query(xyz.astype(np.float64), k=min(4, len(xyz)), workers=1)[0][:, 1:]
            iso = np.median(dists, axis=1).clip(min=1e-4).astype(np.float32)
        else:
            iso = np.full(len(xyz), 0.02, dtype=np.float32)
        return GaussianState(
            points_xyz=xyz,
            colors=rgb,
            opacity=np.full(len(xyz), 0.5, dtype=np.float32),
            scales=np.repeat(iso[:, None], 3, axis=1),
            quats=normalize_quats(None, len(xyz)),
        )


def _dilate_mask(mask_uint8: ArrayU8, pad_px: int) -> ArrayBool:
    binary = mask_uint8 == 255
    if pad_px <= 0 or not binary.any():
        return binary
    size = 2 * pad_px + 1
    return np.asarray(binary_dilation(binary, structure=np.ones((size, size), dtype=bool)), dtype=bool)


def compute_roi_gaussian_mask(
    points_xyz: ArrayF32,
    manifest: SceneManifest,
    masks_dir: Path,
    skip_frames: set[int] | None = None,
    roi_pad_px: int = 8,
    min_hits: int = 1,
) -> ArrayBool:
    """Identify Gaussians projecting inside dilated foreground masks across active views."""
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise GeometryError("points_xyz must have shape (N, 3)")
    skipped = skip_frames or set()
    intrinsics = manifest.intrinsics.model_dump()
    hits = np.zeros(len(points), dtype=np.int32)
    used_masks = 0
    for frame in manifest.frames:
        mask_path = _resolve_frame_file(masks_dir, frame.frame_idx)
        if frame.excluded or frame.frame_idx in skipped or mask_path is None or frame.cam_from_world is None:
            continue
        dilated = _dilate_mask(load_binary_mask_png(mask_path, (manifest.height, manifest.width)), roi_pad_px)
        used_masks += 1
        pose = np.asarray(frame.cam_from_world, dtype=np.float64)
        cam = (pose[:3, :3] @ points.T) + pose[:3, 3:4]
        z = cam[2]
        valid = z > 1e-4
        u = np.rint(intrinsics["fx"] * (cam[0, valid] / z[valid]) + intrinsics["cx"]).astype(np.int64)
        v = np.rint(intrinsics["fy"] * (cam[1, valid] / z[valid]) + intrinsics["cy"]).astype(np.int64)
        in_bounds = (u >= 0) & (u < manifest.width) & (v >= 0) & (v < manifest.height)
        idx = np.flatnonzero(valid)[in_bounds]
        hits[idx[dilated[v[in_bounds], u[in_bounds]]]] += 1
    if used_masks == 0:
        raise StageExecutionError("No active binary masks found for Week 2 ROI computation")
    return hits >= min_hits


def compute_masked_l1_loss(
    rendered_rgb: ArrayF32,
    target_rgb: ArrayF32,
    mask_uint8: ArrayU8,
    unmasked_weight: float = 0.0,
) -> float:
    rendered = np.asarray(rendered_rgb, dtype=np.float32)
    target = np.asarray(target_rgb, dtype=np.float32)
    mask = np.asarray(mask_uint8, dtype=np.uint8)
    if rendered.shape != target.shape or rendered.ndim != 3 or mask.shape != rendered.shape[:2]:
        raise GeometryError("rendered, target, and mask shapes must agree")
    if not (np.isfinite(rendered).all() and np.isfinite(target).all()):
        raise GeometryError("rendered and target RGB tensors must be finite")
    fg = mask == 255
    fg_loss = float(np.abs(rendered[fg] - target[fg]).mean()) if fg.any() else 0.0
    if unmasked_weight <= 0.0 or (~fg).sum() == 0:
        return fg_loss
    return fg_loss + unmasked_weight * float(np.abs(rendered[~fg] - target[~fg]).mean())


def foreground_bounds(
    request: RebuildRequest, manifest: SceneManifest, state: GaussianState, roi_mask: ArrayBool,
) -> tuple[ArrayF32, ArrayF32]:
    samples: list[NDArray[np.float64]] = []
    for frame in manifest.frames:
        mask_path = _resolve_frame_file(request.masks_dir, frame.frame_idx)
        if frame.excluded or frame.frame_idx in request.exclude_frames or frame.cam_from_world is None or mask_path is None:
            continue
        depth = _load_depth_map(request.base_dir / "depths", frame.frame_idx)
        if depth is not None and depth.shape != (manifest.height, manifest.width):
            raise GeometryError(f"Depth dimensions do not match manifest for frame {frame.frame_idx}")
        mask = load_binary_mask_png(mask_path, (manifest.height, manifest.width))
        points = _lift_foreground_points(mask, np.asarray(frame.cam_from_world), manifest.intrinsics.model_dump(), depth)
        if len(points):
            samples.append(points)
    points = np.concatenate(samples) if samples else state.points_xyz[roi_mask].astype(np.float64)
    if len(points) == 0:
        raise StageExecutionError("Cannot bound edited foreground: masks contain no supported 3D points")
    lower, upper = np.percentile(points, [1, 99], axis=0)
    padding = np.maximum(upper - lower, 1e-4) * request.bbox_pad_fraction
    return (lower - padding).astype(np.float32), (upper + padding).astype(np.float32)


def prune_near_camera_floaters(
    state: GaussianState, manifest: SceneManifest, roi_mask: ArrayBool, request: RebuildRequest,
) -> tuple[GaussianState, int]:
    remove = np.zeros(len(state.points_xyz), dtype=bool)
    candidates = roi_mask & (state.opacity < request.floater_opacity_threshold)
    points = state.points_xyz[candidates]
    intr = manifest.intrinsics
    for frame in manifest.frames:
        if frame.excluded or frame.frame_idx in request.exclude_frames or frame.cam_from_world is None:
            continue
        pose = np.asarray(frame.cam_from_world)
        camera = points @ pose[:3, :3].T + pose[:3, 3]
        z = camera[:, 2]
        positive = z > 0
        u = intr.fx * camera[:, 0] / np.maximum(z, 1e-8) + intr.cx
        v = intr.fy * camera[:, 1] / np.maximum(z, 1e-8) + intr.cy
        visible = positive & (z < request.near_plane_distance) & (u >= 0) & (u < manifest.width) & (v >= 0) & (v < manifest.height)
        remove[np.flatnonzero(candidates)[visible]] = True
    keep = ~remove
    if not keep.any():
        raise StageExecutionError("Near-camera pruning would remove the entire scene")
    return GaussianState(state.points_xyz[keep], state.colors[keep], state.opacity[keep],
                         state.scales[keep], state.quats_or_identity()[keep], state.extra), int(remove.sum())


def assert_frozen_gaussians_unchanged(base: GaussianState, updated: GaussianState, frozen_mask: ArrayBool) -> float:
    if not frozen_mask.any():
        return 0.0
    drifts = {
        key: float(np.max(np.abs(
            getattr(updated, key)[frozen_mask].astype(np.float64)
            - getattr(base, key)[frozen_mask].astype(np.float64)
        )))
        for key in ("points_xyz", "colors", "opacity", "scales")
    }
    drifts["quats"] = float(np.max(np.abs(
        updated.quats_or_identity()[frozen_mask].astype(np.float64)
        - base.quats_or_identity()[frozen_mask].astype(np.float64)
    )))
    changed_field = max(drifts, key=drifts.__getitem__)
    max_drift = drifts[changed_field]
    if max_drift != 0.0:
        raise StageExecutionError(
            f"Masked finetune mutated frozen background Gaussians: field={changed_field}, drift={max_drift}"
        )
    return max_drift


def _resolve_orig_frames_dir(request: RebuildRequest, orig_frames_dir: Path | None) -> Path | None:
    if orig_frames_dir is not None:
        return orig_frames_dir
    for name in ("frames", "images"):
        candidate = request.base_dir / name
        if candidate.is_dir():
            return candidate
    return None


def _load_training_views(
    request_or_manifest: RebuildRequest | SceneManifest,
    manifest_or_edited_dir: SceneManifest | Path,
    skip_or_masks_dir: set[int] | Path,
    orig_frames_dir: Path | None = None,
    skip_frames: set[int] | None = None,
) -> list[tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.bool_] | None]]:
    """Load training views, falling back to original unedited frames with zero masks when needed."""
    if isinstance(request_or_manifest, RebuildRequest):
        request = request_or_manifest
        manifest = manifest_or_edited_dir
        assert isinstance(manifest, SceneManifest)
        skip_set = skip_or_masks_dir if isinstance(skip_or_masks_dir, set) else set()
        edited_dir = request.edited_frames_dir
        masks_dir = request.masks_dir
        orig_dir = _resolve_orig_frames_dir(request, orig_frames_dir)
    else:
        manifest = request_or_manifest
        edited_dir = Path(manifest_or_edited_dir)
        masks_dir = Path(skip_or_masks_dir)
        skip_set = skip_frames or set()
        orig_dir = orig_frames_dir
    views: list[tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.bool_] | None]] = []
    for frame in manifest.frames:
        if frame.cam_from_world is None:
            continue
        is_excluded = frame.excluded or frame.frame_idx in skip_set
        rgb_path = None if is_excluded else _resolve_frame_file(edited_dir, frame.frame_idx, frame.image_relpath)
        used_original = False
        if rgb_path is None and orig_dir is not None and not frame.excluded:
            # Unedited or manually excluded-from-edit frames keep original appearance with a zero edit mask.
            rgb_path = _resolve_frame_file(orig_dir, frame.frame_idx, frame.image_relpath)
            used_original = rgb_path is not None
        if rgb_path is None:
            continue
        with Image.open(rgb_path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        if used_original:
            fg: NDArray[np.bool_] | None = np.zeros((manifest.height, manifest.width), dtype=bool)
        else:
            mask_path = _resolve_frame_file(masks_dir, frame.frame_idx)
            fg = load_binary_mask_png(mask_path, (manifest.height, manifest.width)) == 255 if mask_path is not None else None
        views.append((np.asarray(frame.cam_from_world, dtype=np.float64), rgb, fg))
    if not views:
        raise StageExecutionError("No edited or fallback original RGB views found for 3DGS rebuild")
    return views


def _build_k_matrix(manifest: SceneManifest, torch_mod: Any, device: str) -> Any:
    intr = manifest.intrinsics
    k_mat = [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]]
    return torch_mod.tensor(k_mat, dtype=torch_mod.float32, device=device).unsqueeze(0)


def _compose_scene_tensors(
    base_state: GaussianState, roi_idx: Any, bg_idx: Any, active_vars: tuple[Any, ...], bg_vars: tuple[Any, ...], torch_mod: Any
) -> tuple[Any, Any, Any, Any, Any]:
    n_pts = len(base_state.points_xyz)
    out: list[Any] = []
    for act, bg in zip(active_vars, bg_vars, strict=True):
        buf = torch_mod.empty((n_pts, *act.shape[1:]), dtype=act.dtype, device=act.device)
        buf[roi_idx] = act
        if len(bg_idx) > 0:
            buf[bg_idx] = bg
        out.append(buf)
    return out[0], out[1], out[2], out[3], out[4]


def _run_opt_loop(
    request: RebuildRequest,
    manifest: SceneManifest,
    base_state: GaussianState,
    views: list[tuple[NDArray[np.float64], NDArray[np.float32], NDArray[np.bool_] | None]],
    roi_mask: ArrayBool,
    torch_mod: Any,
    rasterize: Any,
) -> tuple[Any, Any, tuple[Any, ...], tuple[Any, ...], Any]:
    dev = request.device
    roi_idx = torch_mod.as_tensor(np.flatnonzero(roi_mask), dtype=torch_mod.long, device=dev)
    bg_idx = torch_mod.as_tensor(np.flatnonzero(~roi_mask), dtype=torch_mod.long, device=dev)
    pts_t = torch_mod.as_tensor(base_state.points_xyz, dtype=torch_mod.float32, device=dev)
    cols_t = torch_mod.as_tensor(base_state.colors.astype(np.float32) / 255.0, dtype=torch_mod.float32, device=dev)
    op_t = torch_mod.as_tensor(base_state.opacity, dtype=torch_mod.float32, device=dev)
    sc_t = torch_mod.as_tensor(base_state.scales, dtype=torch_mod.float32, device=dev)
    q_t = torch_mod.as_tensor(base_state.normalized_quats(), dtype=torch_mod.float32, device=dev)
    tensors = (pts_t, cols_t, op_t, sc_t, q_t)
    active = tuple(t[roi_idx].detach().clone().requires_grad_(True) for t in tensors)
    frozen = tuple(t[bg_idx].detach() for t in tensors)
    ks = _build_k_matrix(manifest, torch_mod, dev)
    opt = torch_mod.optim.Adam([{"params": list(active), "lr": 5e-3, "name": "roi"}])
    steps = request.fine_tune_steps if request.mode == "masked_finetune" else request.from_scratch_steps
    bounds = foreground_bounds(request, manifest, base_state, roi_mask) if request.rebuild_safeguards and request.mode == "masked_finetune" else None
    limits = tuple(torch_mod.as_tensor(value, device=dev) for value in bounds) if bounds is not None else None
    for step in range(steps):
        pose_np, tgt_np, fg_np = views[step % len(views)]
        opt.zero_grad()
        m_all, c_all, o_all, s_all, q_all = _compose_scene_tensors(base_state, roi_idx, bg_idx, active, frozen, torch_mod)
        q_norm = q_all / q_all.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        viewmat = torch_mod.as_tensor(pose_np, dtype=torch_mod.float32, device=dev).unsqueeze(0)
        rendered, *_ = rasterize(
            means=m_all, quats=q_norm, scales=s_all.clamp(min=1e-5), opacities=o_all.clamp(0.0, 1.0),
            colors=c_all.clamp(0.0, 1.0), viewmats=viewmat, Ks=ks, width=manifest.width, height=manifest.height,
        )
        pred, target = rendered[0], torch_mod.as_tensor(tgt_np, dtype=torch_mod.float32, device=dev)
        if request.mode == "masked_finetune" and fg_np is not None and fg_np.any():
            fg_mask = torch_mod.as_tensor(fg_np, dtype=torch_mod.bool, device=dev)
            bg_mask = ~fg_mask
            fg_loss = (pred[fg_mask] - target[fg_mask]).abs().mean()
            bg_loss = (pred[bg_mask] - target[bg_mask]).abs().mean() if bg_mask.any() else 0.0
            if request.rebuild_safeguards:
                bg_loss = request.unmasked_weight * bg_loss
            loss = request.masked_loss_weight * fg_loss + bg_loss
        else:
            loss = (pred - target).abs().mean()
        loss.backward()
        opt.step()
        if limits is not None:
            with torch_mod.no_grad():
                active[0].clamp_(min=limits[0], max=limits[1])
        if (step + 1) % 50 == 0 or step == steps - 1:
            report_progress("07_rebuild", 10 + 85 * (step + 1) / steps,
                            f"Rebuilding 3DGS step {step + 1}/{steps}", iteration=step + 1)
    return roi_idx, bg_idx, active, frozen, torch_mod


def _merge_optimized_roi(base: GaussianState, optimized: GaussianState, roi_mask: ArrayBool) -> GaussianState:
    # Frozen rows must bypass clipping and quaternion normalization, including when checkpoints are saved.
    merged = GaussianState(
        points_xyz=base.points_xyz.copy(), colors=base.colors.copy(),
        opacity=base.opacity.copy(), scales=base.scales.copy(),
        quats=base.quats_or_identity().copy(), extra=base.extra.copy(),
    )
    for name in ("points_xyz", "colors", "opacity", "scales"):
        getattr(merged, name)[roi_mask] = getattr(optimized, name)
    merged.quats_or_identity()[roi_mask] = optimized.quats_or_identity()
    return merged


def _default_gpu_rebuild(
    request: RebuildRequest,
    manifest: SceneManifest,
    base_state: GaussianState,
    roi_mask: ArrayBool,
) -> GaussianState:
    """Run differentiable masked gsplat optimization where only ROI Gaussians receive gradients."""
    try:
        torch = importlib.import_module("torch")
        gsplat = importlib.import_module("gsplat")
    except ImportError as exc:
        raise StageExecutionError("s07_rebuild requires torch and gsplat when no backend is injected") from exc
    if request.device != "cpu" and not torch.cuda.is_available():
        raise StageExecutionError("s07_rebuild requires CUDA GPU or explicit cpu device")
    rasterize = getattr(gsplat, "rasterization", None)
    if rasterize is None:
        raise StageExecutionError("installed gsplat does not expose rasterization")
    torch.manual_seed(request.seed)
    views = _load_training_views(request, manifest, set(request.exclude_frames))
    _, _, active, _, torch_mod = _run_opt_loop(request, manifest, base_state, views, roi_mask, torch, rasterize)
    with torch_mod.no_grad():
        m_f, c_f, o_f, s_f, q_f = active
        pts_out = m_f.detach().cpu().numpy().astype(np.float32)
        cols_out = np.rint(c_f.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
        op_out = o_f.detach().cpu().numpy().clip(0.0, 1.0).astype(np.float32)
        sc_out = s_f.detach().cpu().numpy().clip(min=1e-5).astype(np.float32)
        q_out = normalize_quats(q_f.detach().cpu().numpy(), len(pts_out))
    optimized = GaussianState(points_xyz=pts_out, colors=cols_out, opacity=op_out, scales=sc_out, quats=q_out)
    return _merge_optimized_roi(base_state, optimized, roi_mask)


def _apply_frame_exclusions(frames: list[FrameRecord], skip_frames: set[int]) -> list[dict[str, Any]]:
    # exclude_frames applies only to diffusion editing; rebuild retains unedited originals for those frames.
    _ = skip_frames
    return [frame.model_dump() for frame in frames]


def _verify_pose_invariance(base: SceneManifest, rebuilt: SceneManifest) -> float:
    if rebuilt.intrinsics != base.intrinsics or rebuilt.width != base.width or rebuilt.height != base.height:
        raise StageExecutionError("Week 2 rebuild altered canonical camera intrinsics or dimensions")
    if rebuilt.T_viewer_from_colmap != base.T_viewer_from_colmap:
        raise StageExecutionError("Week 2 rebuild altered canonical viewer transform")
    drifts: list[float] = []
    for b, r in zip(base.frames, rebuilt.frames, strict=True):
        if b.cam_from_world is None and r.cam_from_world is None:
            continue
        if b.cam_from_world is None or r.cam_from_world is None:
            raise StageExecutionError("Week 2 rebuild altered camera registration status")
        drifts.append(float(np.max(np.abs(np.asarray(r.cam_from_world) - np.asarray(b.cam_from_world)))))
    max_drift = max(drifts, default=0.0)
    if max_drift != 0.0:
        raise StageExecutionError(f"Week 2 rebuild drifted camera poses: max_drift={max_drift}")
    return max_drift


def _validate_rebuild_preconditions(request: RebuildRequest, base_manifest: SceneManifest) -> None:
    if request.parent_version_id != base_manifest.version_id:
        raise StageExecutionError(f"parent_version_id '{request.parent_version_id}' != base version '{base_manifest.version_id}'")
    if not (request.base_dir / "sparse" / "0").exists():
        raise StageExecutionError(f"Missing canonical sparse/0 directory under {request.base_dir}")


def _populate_rebuilt_media(request: RebuildRequest, base_manifest: SceneManifest, skip_set: set[int]) -> list[dict[str, Any]]:
    """Populate output_dir/<frame.image_relpath> losslessly and merge depths/ into output_dir."""
    orig_dir = _resolve_orig_frames_dir(request, None)
    updated_frames: list[dict[str, Any]] = []
    for frame in base_manifest.frames:
        frame_dict = frame.model_dump()
        rel_path = Path(frame.image_relpath)
        use_edited = not frame.excluded and frame.frame_idx not in skip_set
        edited_src = _resolve_frame_file(request.edited_frames_dir, frame.frame_idx, frame.image_relpath) if use_edited else None
        if edited_src is not None:
            # Preserve lossless PNG edited frames without JPEG re-compression so unmasked pixels have max_abs_diff == 0.
            if edited_src.suffix.lower() == ".png" and rel_path.suffix.lower() != ".png":
                rel_path = rel_path.with_suffix(".png")
                frame_dict["image_relpath"] = rel_path.as_posix()
            dst_img = request.output_dir / rel_path
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if edited_src.suffix.lower() == dst_img.suffix.lower():
                shutil.copy2(edited_src, dst_img)
            else:
                with Image.open(edited_src) as img:
                    img.convert("RGB").save(dst_img)
            updated_frames.append(frame_dict)
            continue
        dst_img = request.output_dir / rel_path
        base_src = request.base_dir / rel_path
        orig_src = _resolve_frame_file(orig_dir, frame.frame_idx, frame.image_relpath) if orig_dir is not None else None
        src = base_src if base_src.is_file() else orig_src
        if src is not None and src.is_file():
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix.lower() == dst_img.suffix.lower():
                shutil.copy2(src, dst_img)
            else:
                with Image.open(src) as img:
                    img.convert("RGB").save(dst_img)
            updated_frames.append(frame_dict)
            continue
        if dst_img.is_file():
            updated_frames.append(frame_dict)
            continue
        raise StageExecutionError(f"Missing base frame image for rebuild: {frame.image_relpath}")
    base_depths = request.base_dir / "depths"
    out_depths = request.output_dir / "depths"
    if base_depths.is_dir():
        out_depths.mkdir(parents=True, exist_ok=True)
        for depth_file in base_depths.iterdir():
            if depth_file.is_file() and not (out_depths / depth_file.name).exists():
                shutil.copy2(depth_file, out_depths / depth_file.name)
    return updated_frames


def _sync_colmap_images_bin_names(images_bin_path: Path, rebuilt_frames: list[dict[str, Any]]) -> None:
    """Keep COLMAP images.bin filenames aligned when edited frames switch from .jpg to lossless .png."""
    if not images_bin_path.is_file():
        raise StageExecutionError(f"Malformed COLMAP images.bin at {images_bin_path}: file missing")
    data = images_bin_path.read_bytes()
    if len(data) < 8:
        raise StageExecutionError(f"Malformed COLMAP images.bin at {images_bin_path}: short count header")
    by_image_id = {
        int(f["colmap_image_id"]): Path(str(f["image_relpath"])).name
        for f in rebuilt_frames
        if f.get("colmap_image_id") is not None and f.get("image_relpath")
    }
    by_stem = {
        Path(str(f["image_relpath"])).stem: Path(str(f["image_relpath"])).name
        for f in rebuilt_frames
        if f.get("image_relpath")
    }
    hdr_size = struct.calcsize("<I4d3dI")
    try:
        num_images = struct.unpack_from("<Q", data, 0)[0]
        offset = 8
        out_chunks = [data[:8]]
        for _ in range(num_images):
            if offset + hdr_size > len(data):
                raise StageExecutionError(f"Malformed COLMAP images.bin at {images_bin_path}: truncated image header")
            image_id = struct.unpack_from("<I", data, offset)[0]
            header_bytes = data[offset : offset + hdr_size]
            offset += hdr_size
            nul = data.find(b"\x00", offset)
            if nul < 0:
                raise StageExecutionError(f"Malformed COLMAP images.bin at {images_bin_path}: unterminated filename")
            old_name = data[offset:nul].decode("utf-8")
            offset = nul + 1
            if offset + 8 > len(data):
                raise StageExecutionError(f"Malformed COLMAP images.bin at {images_bin_path}: truncated num_pts2d")
            num_pts2d = struct.unpack_from("<Q", data, offset)[0]
            pts_end = offset + 8 + num_pts2d * 24
            if pts_end > len(data):
                raise StageExecutionError(f"Malformed COLMAP images.bin at {images_bin_path}: truncated 2D point tracks")
            new_leaf = by_image_id.get(int(image_id)) or by_stem.get(Path(old_name).stem)
            if new_leaf is not None and "/" in old_name:
                new_name = f"{old_name.rsplit('/', 1)[0]}/{new_leaf}"
            else:
                new_name = new_leaf or old_name
            out_chunks.append(header_bytes)
            out_chunks.append(new_name.encode("utf-8") + b"\x00")
            out_chunks.append(data[offset:pts_end])
            offset = pts_end
        if offset != len(data):
            raise StageExecutionError(f"Malformed COLMAP images.bin at {images_bin_path}: trailing bytes")
    except (struct.error, UnicodeDecodeError) as exc:
        raise StageExecutionError(f"Malformed COLMAP images.bin at {images_bin_path}: {exc}") from exc
    images_bin_path.write_bytes(b"".join(out_chunks))


def _export_rebuilt_scene(
    request: RebuildRequest, base_manifest: SceneManifest, updated: GaussianState, skip_set: set[int]
) -> SceneManifest:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(request.base_dir / "sparse", request.output_dir / "sparse", dirs_exist_ok=True)
    rebuilt_frames = _populate_rebuilt_media(request, base_manifest, skip_set)
    _sync_colmap_images_bin_names(request.output_dir / "sparse" / "0" / "images.bin", rebuilt_frames)
    save_gaussian_state(request.output_dir, updated)
    return export_scene(
        ExportRequest(
            output_dir=request.output_dir,
            project_id=request.project_id,
            version_id=request.version_id,
            parent_version_id=request.parent_version_id,
            width=base_manifest.width,
            height=base_manifest.height,
            intrinsics=base_manifest.intrinsics.model_dump(),
            frames=rebuilt_frames,
            test_every=request.test_every,
            pose_backend=base_manifest.pose_backend,
            T_viewer_from_colmap=base_manifest.T_viewer_from_colmap,
        ),
        points_xyz=updated.points_xyz,
        opacity=updated.opacity,
        scales=updated.scales,
        colors=updated.colors,
        quats=updated.normalized_quats(),
    )


def rebuild_scene(
    request: RebuildRequest,
    backend: Callable[[RebuildRequest, SceneManifest, GaussianState, ArrayBool], GaussianState] | None = None,
) -> dict[str, Any]:
    """Execute Week 2 localized 3DGS fine-tune or from-scratch ablation without touching COLMAP poses."""
    started = time.perf_counter()
    report_progress("07_rebuild", 10, "Loading base scene and foreground masks")
    base_manifest, base_state = load_base_scene(request.base_dir)
    _validate_rebuild_preconditions(request, base_manifest)
    skip_set = set(request.exclude_frames)
    if request.mode == "masked_finetune":
        roi_mask = compute_roi_gaussian_mask(
            base_state.points_xyz, base_manifest, request.masks_dir, skip_set, request.roi_pad_px, request.min_mask_hits
        )
        init_state = base_state
        if request.rebuild_safeguards:
            lower, upper = foreground_bounds(request, base_manifest, base_state, roi_mask)
            roi_mask &= ((base_state.points_xyz >= lower) & (base_state.points_xyz <= upper)).all(axis=1)
            if not roi_mask.any():
                raise StageExecutionError("No editable Gaussians lie inside the foreground bounds; check mask/depth alignment")
    else:
        sparse_init = _init_from_sparse_dir(request.base_dir / "sparse" / "0")
        init_state = sparse_init if sparse_init is not None else base_state
        roi_mask = np.ones(len(init_state.points_xyz), dtype=bool)
    updated = (backend or _default_gpu_rebuild)(request, base_manifest, init_state, roi_mask)
    max_unmasked_drift = assert_frozen_gaussians_unchanged(base_state, updated, ~roi_mask) if request.mode == "masked_finetune" else 0.0
    pruned = 0
    if request.rebuild_safeguards:
        updated, pruned = prune_near_camera_floaters(updated, base_manifest, roi_mask, request)
    rebuilt_manifest = _export_rebuilt_scene(request, base_manifest, updated, skip_set)
    max_pose_drift = _verify_pose_invariance(base_manifest, rebuilt_manifest)
    steps = request.fine_tune_steps if request.mode == "masked_finetune" else request.from_scratch_steps
    metrics = {
        "mode": request.mode, "steps": steps, "runtime_sec": round(time.perf_counter() - started, 4),
        "total_gaussian_count": int(len(updated.points_xyz)), "roi_gaussian_count": int(roi_mask.sum()),
        "frozen_gaussian_count": int((~roi_mask).sum()) if request.mode == "masked_finetune" else 0,
        "excluded_frame_count": int(sum(1 for f in rebuilt_manifest.frames if f.excluded)),
        "max_pose_drift": max_pose_drift, "max_unmasked_gaussian_drift": max_unmasked_drift,
        "parent_version_id": request.parent_version_id, "version_id": request.version_id,
        "rebuild_safeguards": request.rebuild_safeguards, "near_camera_floaters_pruned": pruned,
    }
    (request.output_dir / "rebuild_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
