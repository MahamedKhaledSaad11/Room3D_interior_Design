from __future__ import annotations

import importlib
import inspect
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.spatial import cKDTree

from room3d_worker.common.errors import GeometryError, StageExecutionError

LOGGER = logging.getLogger(__name__)
SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class TrainConfig:
    output_dir: Path
    device: str = "cuda:0"
    seed: int = 0
    iterations: int = 30_000
    test_every: int = 0
    holdout_every: int | None = None
    cap_max: int = 800_000
    min_gaussians: int | None = None
    max_gaussians: int | None = None
    export_depths: bool = True
    use_ppisp: bool = True
    allow_synthetic_fallback: bool = False
    lr_means: float = 1.6e-4
    scene_extent: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.test_every < 0 or self.iterations < 1:
            raise ValueError("invalid evaluation interval or iteration count")
        cap = self.max_gaussians if self.max_gaussians is not None else self.cap_max
        if cap < 1:
            raise ValueError("cap_max must be >= 1")
        cap = min(cap, 1_500_000)
        object.__setattr__(self, "cap_max", cap)
        object.__setattr__(self, "max_gaussians", cap)
        if self.min_gaussians is not None and (self.min_gaussians < 1 or self.min_gaussians > cap):
            raise ValueError("min_gaussians must stay within [1, cap_max]")


@dataclass(frozen=True)
class TrainTelemetry:
    psnr: float
    ssim: float
    gaussian_count: int
    peak_vram_mb: float
    runtime_sec: float


def _normalize_rgb(rgb: NDArray[Any] | None, count: int) -> NDArray[np.float32]:
    if rgb is None:
        return np.full((count, 3), 0.5, dtype=np.float32)
    arr = np.asarray(rgb)
    if np.issubdtype(arr.dtype, np.integer) or float(arr.max(initial=0.0)) > 1.0:
        return np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0)
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


@dataclass
class GaussianModelState:
    """Mutable 3DGS model state carrying full anisotropic parameters."""

    xyz: NDArray[np.float32]
    scales: NDArray[np.float32]
    quats: NDArray[np.float32]
    opacities: NDArray[np.float32]
    sh0: NDArray[np.float32]
    sh_rest: NDArray[np.float32] | None = None

    @classmethod
    def from_sparse(cls, points_xyz: NDArray[Any], rgb: NDArray[Any] | None = None) -> GaussianModelState:
        pts = np.asarray(points_xyz, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            raise GeometryError("points_xyz must have shape (N, 3) with N > 0")
        count = len(pts)
        norm_rgb = _normalize_rgb(rgb, count)
        try:
            from room3d_worker.common.gaussian_state import from_sparse_points

            state = from_sparse_points(pts, norm_rgb)
            return cls(state.xyz, state.scales, state.quats, state.opacities, state.sh0, state.sh_rest)
        except ImportError:
            iso = np.exp(initialize_scales_knn(pts, k=min(3, count - 1))) if count > 1 else np.full(count, 0.01, np.float32)
            scales = np.repeat(iso[:, None], 3, axis=1).astype(np.float32)
            quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1))
            opacities = np.full(count, 0.5, dtype=np.float32)
            return cls(pts, scales, quats, opacities, norm_rgb, None)

    def to_common_state(self) -> Any:
        from room3d_worker.common.gaussian_state import GaussianState

        return GaussianState(
            xyz=self.xyz, scales=self.scales, quats=self.quats,
            opacities=self.opacities, sh0=self.sh0, sh_rest=self.sh_rest,
        )


def build_train_config(output_dir: str | Path, allow_synthetic_fallback: bool = False, **overrides: Any) -> TrainConfig:
    """Build the production-safe config; evaluation is opt-in."""
    return TrainConfig(output_dir=Path(output_dir), allow_synthetic_fallback=allow_synthetic_fallback, **overrides)


def initialize_scales_knn(points_xyz: NDArray[Any], k: int = 3) -> NDArray[np.float32]:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) <= k:
        raise GeometryError("at least k+1 XYZ points are required for scale initialization")
    if not np.isfinite(points).all():
        raise GeometryError("points_xyz contains non-finite values")
    distances = cKDTree(points).query(points, k=k + 1, workers=1)[0][:, 1:]
    scale = np.median(distances, axis=1).clip(min=1e-6)
    return np.log(scale).astype(np.float32)


def radial_outlier_mask(points_xyz: NDArray[Any], sigma: float = 6.0) -> NDArray[np.bool_]:
    """Reject isolated statistical outliers while retaining dense, extreme room walls."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise GeometryError("at least four XYZ points are required for outlier pruning")
    center = np.median(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    q1, q3 = np.percentile(radii, [25.0, 75.0])
    radial_limit = q3 + 3.0 * max(q3 - q1, 1e-9)
    nn = cKDTree(points).query(points, k=2, workers=1)[0][:, 1]
    nn_median = float(np.median(nn))
    nn_mad = float(np.median(np.abs(nn - nn_median)))
    density_limit = nn_median + sigma * max(1.4826 * nn_mad, 1e-9)
    # A dense wall can be far from the robust center; local support protects it.
    return (radii <= radial_limit) | (nn <= density_limit)


def _peak_vram_mb(torch: Any) -> float:
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else 0.0


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str, torch: Any) -> str:
    if not requested.startswith("cuda"):
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    count = int(torch.cuda.device_count())
    if ":" in requested:
        try:
            ordinal = int(requested.split(":", 1)[1])
        except ValueError:
            return "cuda:0"
        if ordinal < 0 or ordinal >= count:
            LOGGER.warning("Requested device %s unavailable (device_count=%d); falling back to cuda:0", requested, count)
            return "cuda:0"
    return requested


def _optional_ppisp() -> Any | None:
    try:
        return importlib.import_module("ppisp")
    except ImportError:
        return None


def save_depth_maps(depth_maps: Any, output_dir: str | Path) -> list[Path]:
    """Persist renderer depth outputs in a format usable by Week 2 Lift-to-3D."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    array = np.asarray(depth_maps)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or not np.isfinite(array).all():
        raise GeometryError("depth_maps must be finite arrays shaped (N, H, W)")
    paths: list[Path] = []
    for index, depth in enumerate(array):
        path = target / f"{index:05d}.npy"
        np.save(path, depth.astype(np.float32), allow_pickle=False)
        paths.append(path)
    return paths


def _render_view_depths(gsplat: Any, torch: Any, params: dict[str, Any], views: list[dict[str, Any]], device: str) -> dict[int, np.ndarray]:
    maps: dict[int, np.ndarray] = {}
    with torch.no_grad():
        q = params["quats"] / params["quats"].norm(dim=-1, keepdim=True).clamp(min=1e-8)
        for view in views:
            depth = rasterize_depths(
                gsplat,
                means=params["means"],
                quats=q,
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=params["sh0"].clamp(0.0, 1.0),
                viewmats=torch.as_tensor(view["viewmat"], dtype=torch.float32, device=device).unsqueeze(0),
                Ks=torch.as_tensor(view["K"], dtype=torch.float32, device=device).unsqueeze(0),
                width=view["width"],
                height=view["height"],
            )
            maps[int(view["frame_idx"])] = depth[0, ..., 0].detach().cpu().numpy().astype(np.float32)
    return maps


def rasterize_depths(gsplat: Any, **render_kwargs: Any) -> Any:
    """Use gsplat's expected-depth mode; ED is depth, not accumulated alpha."""
    rasterization = getattr(gsplat, "rasterization", None)
    if rasterization is None:
        raise StageExecutionError("installed gsplat does not expose rasterization")
    rendered = rasterization(render_mode="ED", **render_kwargs)
    return rendered[0] if isinstance(rendered, tuple) else rendered


def _load_manifest(work_dir: Path, manifest: Any | None) -> Any | None:
    if manifest is not None:
        return manifest
    manifest_path = work_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    from room3d_worker.common.manifest import load_manifest

    return load_manifest(manifest_path)


def _resolved_intrinsics(manifest: Any, actual_w: int, actual_h: int) -> tuple[float, float, float, float]:
    """Scale manifest PINHOLE intrinsics when training images differ from manifest dimensions."""
    intr = manifest.intrinsics
    if actual_w <= 0 or actual_h <= 0:
        raise GeometryError("Training image dimensions must be positive")
    if actual_w == int(manifest.width) and actual_h == int(manifest.height):
        return float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy)
    sx = float(actual_w) / float(manifest.width)
    sy = float(actual_h) / float(manifest.height)
    return float(intr.fx) * sx, float(intr.fy) * sy, float(intr.cx) * sx, float(intr.cy) * sy


def _extract_training_views(work_dir: Path, manifest: Any | None, images: Any) -> list[dict[str, Any]]:
    if manifest is None:
        return []
    views: list[dict[str, Any]] = []
    img_arr = np.asarray(images) if images is not None else None
    fallback_hw = (int(img_arr.shape[1]), int(img_arr.shape[2])) if img_arr is not None and img_arr.ndim == 4 and len(img_arr) > 0 else None
    frames = manifest.frames
    for idx, frame in enumerate(frames):
        if getattr(frame, "excluded", False) or getattr(frame, "cam_from_world", None) is None:
            continue
        rel = Path(frame.image_relpath)
        candidates = [work_dir / rel, work_dir / "images" / rel.name, work_dir / "frames" / rel.name]
        img_path = next((p for p in candidates if p.exists()), None)
        if img_path is not None:
            with Image.open(img_path) as img:
                rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        elif img_arr is not None and img_arr.ndim == 4 and idx < len(img_arr):
            arr = img_arr[idx].astype(np.float32)
            rgb = arr / 255.0 if arr.max(initial=0.0) > 1.5 else arr
        else:
            continue
        actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
        if fallback_hw is not None and (actual_h, actual_w) != fallback_hw and img_path is None:
            actual_h, actual_w = fallback_hw
        fx, fy, cx, cy = _resolved_intrinsics(manifest, actual_w, actual_h)
        k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        views.append({
            "frame_idx": int(frame.frame_idx),
            "viewmat": np.asarray(frame.cam_from_world, dtype=np.float32),
            "K": k_mat,
            "rgb": rgb,
            "width": actual_w,
            "height": actual_h,
        })
    return views


def _build_mcmc_strategy(gsplat: Any, config: TrainConfig) -> Any | None:
    strategy_cls = getattr(getattr(gsplat, "strategy", gsplat), "MCMCStrategy", None)
    if strategy_cls is None:
        return None
    sig = inspect.signature(strategy_cls)
    kwargs: dict[str, Any] = {}
    if "cap_max" in sig.parameters:
        kwargs["cap_max"] = config.cap_max
    if "max_steps" in sig.parameters:
        kwargs["max_steps"] = config.iterations
    if "refine_start_iter" in sig.parameters:
        kwargs["refine_start_iter"] = min(500, max(1, config.iterations // 4))
    if "refine_stop_iter" in sig.parameters:
        kwargs["refine_stop_iter"] = max(1, int(config.iterations * 0.85))
    return strategy_cls(**kwargs)


def _compute_ssim(torch: Any, gsplat: Any, pred: Any, target: Any) -> Any:
    fused_ssim = getattr(gsplat, "fused_ssim", None)
    if callable(fused_ssim):
        # fused_ssim expects NCHW tensors in [0, 1].
        return fused_ssim(pred.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0))
    mu_x, mu_y = pred.mean(), target.mean()
    vx, vy = pred.var(unbiased=False), target.var(unbiased=False)
    cov = ((pred - mu_x) * (target - mu_y)).mean()
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (vx + vy + c2) + 1e-8)


def _make_optimizer(opt_cls: Any, torch: Any, param: Any, lr: float, name: str) -> Any:
    param_groups = [{"params": [param], "lr": lr, "name": name}]
    sig = inspect.signature(opt_cls.__init__ if inspect.isclass(opt_cls) else opt_cls).parameters
    if opt_cls is not torch.optim.Adam and "lr" not in sig:
        kwargs: dict[str, Any] = {}
        if "eps" in sig:
            kwargs["eps"] = 1e-15
        if "betas" in sig:
            kwargs["betas"] = (0.9, 0.999)
        return opt_cls(param_groups, **kwargs)
    if "eps" in sig:
        return opt_cls(param_groups, eps=1e-15)
    return opt_cls(param_groups)


def _visible_gaussian_mask(info: Any) -> Any | None:
    if not isinstance(info, dict) or "radii" not in info:
        return None
    radii = info["radii"]
    visible = radii > 0
    if visible.ndim == 3:
        visible = visible.all(-1)
    if visible.ndim >= 2:
        visible = visible.any(0)
    return visible


def _optimize_with_gsplat(
    config: TrainConfig,
    state: GaussianModelState,
    views: list[dict[str, Any]],
    torch: Any,
    gsplat: Any,
    device: str,
) -> tuple[GaussianModelState, float, float, dict[int, np.ndarray] | None]:
    rasterize = getattr(gsplat, "rasterization", None)
    if rasterize is None:
        raise StageExecutionError("installed gsplat does not expose rasterization")
    means = torch.nn.Parameter(torch.as_tensor(state.xyz, dtype=torch.float32, device=device))
    log_scales = torch.nn.Parameter(torch.log(torch.as_tensor(state.scales, dtype=torch.float32, device=device).clamp(min=1e-6)))
    quats = torch.nn.Parameter(torch.as_tensor(state.quats, dtype=torch.float32, device=device))
    logit_op = torch.nn.Parameter(torch.logit(torch.as_tensor(state.opacities, dtype=torch.float32, device=device).clamp(1e-4, 1.0 - 1e-4)))
    colors_p = torch.nn.Parameter(torch.as_tensor(state.sh0, dtype=torch.float32, device=device))
    params = torch.nn.ParameterDict({"means": means, "scales": log_scales, "quats": quats, "opacities": logit_op, "sh0": colors_p})
    selective_cls = getattr(getattr(gsplat, "optimizers", gsplat), "SelectiveAdam", None) or getattr(
        getattr(gsplat, "strategy", gsplat), "SelectiveAdam", None
    )
    opt_cls = selective_cls if selective_cls is not None else torch.optim.Adam
    optimizers = {
        "means": _make_optimizer(opt_cls, torch, means, config.lr_means, "means"),
        "scales": _make_optimizer(opt_cls, torch, log_scales, 5e-3, "scales"),
        "quats": _make_optimizer(opt_cls, torch, quats, 1e-3, "quats"),
        "opacities": _make_optimizer(opt_cls, torch, logit_op, 5e-2, "opacities"),
        "sh0": _make_optimizer(opt_cls, torch, colors_p, 2.5e-3, "sh0"),
    }
    strategy = _build_mcmc_strategy(gsplat, config)
    if strategy is not None and hasattr(strategy, "check_sanity"):
        strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state() if strategy is not None and hasattr(strategy, "initialize_state") else None
    pre_sig = inspect.signature(strategy.step_pre_backward).parameters if strategy is not None and hasattr(strategy, "step_pre_backward") else {}
    post_sig = inspect.signature(strategy.step_post_backward).parameters if strategy is not None and hasattr(strategy, "step_post_backward") else {}
    use_absgrad = bool(getattr(strategy, "absgrad", False))
    last_psnr, last_ssim = 0.0, 0.0
    for step in range(config.iterations):
        view = views[step % len(views)]
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        means, log_scales, quats = params["means"], params["scales"], params["quats"]
        logit_op, colors_p = params["opacities"], params["sh0"]
        viewmat = torch.as_tensor(view["viewmat"], dtype=torch.float32, device=device).unsqueeze(0)
        k_mat = torch.as_tensor(view["K"], dtype=torch.float32, device=device).unsqueeze(0)
        target = torch.as_tensor(view["rgb"], dtype=torch.float32, device=device)
        raster_kwargs: dict[str, Any] = {
            "means": means, "quats": quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8),
            "scales": torch.exp(log_scales), "opacities": torch.sigmoid(logit_op),
            "colors": colors_p.clamp(0.0, 1.0), "viewmats": viewmat, "Ks": k_mat,
            "width": view["width"], "height": view["height"], "packed": False,
        }
        if use_absgrad:
            raster_kwargs["absgrad"] = True
        out = rasterize(**raster_kwargs)
        rendered = out[0] if isinstance(out, tuple) else out
        info = out[2] if isinstance(out, tuple) and len(out) > 2 else {}
        pred = rendered[0].clamp(0.0, 1.0)
        l1 = (pred - target).abs().mean()
        ssim_val = _compute_ssim(torch, gsplat, pred, target)
        loss = 0.8 * l1 + 0.2 * (1.0 - ssim_val)
        if strategy is not None and strategy_state is not None and hasattr(strategy, "step_pre_backward"):
            pre_args: dict[str, Any] = {}
            if "packed" in pre_sig:
                pre_args["packed"] = False
            strategy.step_pre_backward(params, optimizers, strategy_state, step, info, **pre_args)
        loss.backward()
        visibility = _visible_gaussian_mask(info)
        radii_ok = visibility is not None
        for opt in optimizers.values():
            if opt_cls is selective_cls and hasattr(opt, "step") and "visibility" in inspect.signature(opt.step).parameters:
                if not radii_ok:
                    raise StageExecutionError("SelectiveAdam requires visibility radii from rasterization info")
                opt.step(visibility=visibility)
            else:
                opt.step()
        if strategy is not None and strategy_state is not None and hasattr(strategy, "step_post_backward"):
            post_kwargs: dict[str, Any] = {}
            if "packed" in post_sig:
                post_kwargs["packed"] = False
            if "lr" in post_sig:
                post_kwargs["lr"] = float(optimizers["means"].param_groups[0]["lr"]) if optimizers["means"].param_groups else config.lr_means
            strategy.step_post_backward(params, optimizers, strategy_state, step, info, **post_kwargs)
        if step == config.iterations - 1:
            mse = float(((pred - target) ** 2).mean().item())
            last_psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-10)))
            last_ssim = float(ssim_val.item())
    depths = _render_view_depths(gsplat, torch, dict(params.items()), views, device) if config.export_depths else None
    with torch.no_grad():
        means, log_scales, quats = params["means"], params["scales"], params["quats"]
        logit_op, colors_p = params["opacities"], params["sh0"]
        q_norm = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        refined = GaussianModelState(
            xyz=means.detach().cpu().numpy().astype(np.float32),
            scales=torch.exp(log_scales).detach().cpu().numpy().astype(np.float32),
            quats=q_norm.detach().cpu().numpy().astype(np.float32),
            opacities=torch.sigmoid(logit_op).detach().cpu().numpy().astype(np.float32),
            sh0=colors_p.clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32),
        )
    return refined, last_psnr, last_ssim, depths


def _save_state_artifacts(output_dir: Path, state: GaussianModelState, config: TrainConfig, frames_shape: tuple[int, ...]) -> None:
    try:
        from room3d_worker.common.gaussian_state import GaussianState

        common = GaussianState(state.xyz, state.scales, state.quats, state.opacities, state.sh0, state.sh_rest)
        common.save_npz(output_dir / "gs_state.npz")
        common.save_pt(output_dir / "gs_ckpt.pt")
        return
    except ImportError:
        pass
    np.savez_compressed(
        output_dir / "gs_state.npz",
        xyz=state.xyz.astype(np.float32),
        scales=state.scales.astype(np.float32),
        quats=state.quats.astype(np.float32),
        opacities=state.opacities.astype(np.float32),
        sh0=state.sh0.astype(np.float32),
    )
    try:
        torch = importlib.import_module("torch")
        torch.save(
            {
                "xyz": torch.as_tensor(state.xyz),
                "scales": torch.as_tensor(state.scales),
                "quats": torch.as_tensor(state.quats),
                "opacities": torch.as_tensor(state.opacities),
                "sh0": torch.as_tensor(state.sh0),
                "frames_shape": frames_shape,
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()},
            },
            output_dir / "gs_ckpt.pt",
        )
    except ImportError:
        pass


def _synthetic_fallback(
    config: TrainConfig, state: GaussianModelState, images: Any, depth_maps: Any | None, started: float, reason: str
) -> TrainTelemetry:
    allow_env = os.getenv("ROOM3D_ALLOW_SYNTHETIC_FALLBACK") == "1"
    if (not config.allow_synthetic_fallback and not allow_env) or os.getenv("ROOM3D_STRICT_PRODUCTION") == "1":
        raise StageExecutionError(f"s03_train strict mode disallows synthetic fallback: {reason}")
    # Deterministic lightweight refinement for CPU/headless test environments.
    _save_state_artifacts(config.output_dir, state, config, tuple(np.asarray(images).shape))
    if depth_maps is not None:
        save_depth_maps(depth_maps, config.output_dir / "depths")
    elif config.export_depths:
        (config.output_dir / "depths").mkdir(parents=True, exist_ok=True)
    telemetry = TrainTelemetry(0.0, 0.0, int(len(state.xyz)), 0.0, time.perf_counter() - started)
    LOGGER.info("s03_train synthetic fallback (%s) metrics=%s", reason, telemetry)
    return telemetry


def train_scene(
    config: TrainConfig,
    images: NDArray[Any],
    points_xyz: NDArray[Any],
    points_rgb: NDArray[Any] | None = None,
    manifest: Any | None = None,
    **kwargs: Any,
) -> TrainTelemetry:
    """Train 3DGS on GPU when gsplat + rasterizable views exist; fall back cleanly on CPU/synthetic tests."""
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points_xyz, dtype=np.float32)
    keep = radial_outlier_mask(points) if len(points) >= 4 else np.ones(len(points), dtype=bool)
    kept_rgb = np.asarray(points_rgb)[keep] if points_rgb is not None else None
    state = GaussianModelState.from_sparse(points[keep], kept_rgb)
    resolved_manifest = _load_manifest(config.output_dir, manifest)
    views = _extract_training_views(config.output_dir, resolved_manifest, images)
    depth_maps = kwargs.get("depth_maps")
    try:
        torch = importlib.import_module("torch")
        gsplat = importlib.import_module("gsplat")
    except ImportError:
        return _synthetic_fallback(config, state, images, depth_maps, started, "torch or gsplat unavailable")

    _seed_everything(config.seed, torch)
    device = _resolve_device(config.device, torch)
    _ = _optional_ppisp() if config.use_ppisp else None
    if not views or (device.startswith("cuda") and not torch.cuda.is_available()):
        reason = "no rasterizable training views" if not views else f"CUDA unavailable for {device}"
        return _synthetic_fallback(config, state, images, depth_maps, started, reason)
    refined, psnr, ssim, rendered_depths = _optimize_with_gsplat(config, state, views, torch, gsplat, device)
    _save_state_artifacts(config.output_dir, refined, config, tuple(np.asarray(images).shape))
    if depth_maps is not None:
        save_depth_maps(depth_maps, config.output_dir / "depths")
    elif rendered_depths:
        depths_dir = config.output_dir / "depths"
        depths_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, depth in rendered_depths.items():
            np.save(depths_dir / f"{frame_idx:05d}.npy", depth.astype(np.float32), allow_pickle=False)
    elif config.export_depths:
        (config.output_dir / "depths").mkdir(parents=True, exist_ok=True)
    telemetry = TrainTelemetry(psnr, ssim, int(len(refined.xyz)), _peak_vram_mb(torch), time.perf_counter() - started)
    LOGGER.info("s03_train metrics=%s", telemetry)
    return telemetry
