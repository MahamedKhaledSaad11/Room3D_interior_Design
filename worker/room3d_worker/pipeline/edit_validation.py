from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage import binary_dilation

from room3d_worker.common.errors import StageExecutionError
from room3d_worker.common.manifest import FrameRecord, SceneManifest
from room3d_worker.common.progress import report_progress
from room3d_worker.pipeline.recolor import diverse_frames, recolor_rgb
from room3d_worker.pipeline.s04_export import filter_floaters

if TYPE_CHECKING:
    from room3d_worker.pipeline.s07_rebuild import GaussianState, RebuildRequest


class EditRenderer:
    def __init__(self, manifest: SceneManifest, device: str) -> None:
        self.torch = importlib.import_module("torch")
        self.rasterize = importlib.import_module("gsplat").rasterization
        self.manifest, self.device = manifest, device
        self._states: dict[int, dict[str, Any]] = {}
        intr = manifest.intrinsics
        self.ks = self.tensor([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]])[None]

    def tensor(self, value: object) -> Any:
        return self.torch.as_tensor(value, dtype=self.torch.float32, device=self.device)

    def render(self, state: GaussianState, frame: FrameRecord) -> tuple[NDArray[np.float32], ...]:
        identity = id(state)
        if identity not in self._states:
            keep = filter_floaters(state.points_xyz, state.opacity, state.scales)
            self._states[identity] = {
                "means": self.tensor(state.points_xyz[keep]), "quats": self.tensor(state.normalized_quats()[keep]),
                "scales": self.tensor(state.scales[keep]), "opacities": self.tensor(state.opacity[keep]),
                "colors": self.tensor(state.colors[keep] / 255.0),
            }
        with self.torch.no_grad():
            rgbd, alpha, _ = self.rasterize(
                **self._states[identity], viewmats=self.tensor(frame.cam_from_world)[None],
                Ks=self.ks, width=self.manifest.width, height=self.manifest.height, render_mode="RGB+ED",
            )
        values = rgbd[0].detach().cpu().numpy().astype(np.float32)
        return values[..., :3], values[..., 3], alpha[0, ..., 0].detach().cpu().numpy().astype(np.float32)


def prepare_surfaces(request: RebuildRequest, manifest: SceneManifest, state: GaussianState) -> Path:
    folder = request.output_dir / "surface_depths"
    frames = [frame for frame in manifest.frames if not frame.excluded and frame.cam_from_world is not None
              and frame.frame_idx not in request.exclude_frames
              and (request.masks_dir / f"{frame.frame_idx:06d}.png").is_file()]
    render_surface_depths(manifest, state, request.device, folder, frames)
    return folder


def render_surface_depths(manifest: SceneManifest, state: GaussianState, device: str, folder: Path,
                          frames: list[FrameRecord], stage: str = "07_rebuild") -> None:
    renderer = EditRenderer(manifest, device)
    folder.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        _, depth, alpha = renderer.render(state, frame)
        np.save(folder / f"{frame.frame_idx:06d}.npy", depth, allow_pickle=False)
        np.save(folder / f"{frame.frame_idx:06d}_alpha.npy", alpha, allow_pickle=False)
        report_progress(stage, 10 + 15 * (index + 1) / max(len(frames), 1),
                        f"Checking visible surfaces {index + 1}/{len(frames)}")


def image_acceptance(before: NDArray[np.float32], after: NDArray[np.float32],
                     target: NDArray[np.float32], mask: NDArray[np.bool_]) -> dict[str, float | bool]:
    if not (np.isfinite(before).all() and np.isfinite(after).all() and np.isfinite(target).all()):
        raise StageExecutionError("Nonfinite pixels in edit validation renders")
    background = ~binary_dilation(mask, iterations=4)
    delta = np.abs(before - after)
    bg_mae = float(delta[background].mean()) if background.any() else 0.0
    bg_p99 = float(np.quantile(delta[background], 0.99)) if background.any() else 0.0
    initial = float(np.abs(before[mask] - target[mask]).mean()) if mask.any() else 0.0
    final = float(np.abs(after[mask] - target[mask]).mean()) if mask.any() else 0.0
    improved = not mask.any() or (initial > 0.01 and final <= initial * 0.9)
    return {"background_mae": bg_mae, "background_p99": bg_p99, "target_error_before": initial,
            "target_error_after": final, "passed": bg_mae <= 0.015 and bg_p99 <= 0.08 and improved}


def validate_candidate(request: RebuildRequest, manifest: SceneManifest,
                       base: GaussianState, updated: GaussianState) -> dict[str, object]:
    from room3d_worker.pipeline.s07_rebuild import _resolve_frame_file

    if request.device == "cpu":
        return {"status": "review_required", "reason": "Real GPU matched-camera validation has not run"}
    renderer = EditRenderer(manifest, request.device)
    active = [frame for frame in manifest.frames if frame.frame_idx not in request.exclude_frames
              and (request.masks_dir / f"{frame.frame_idx:06d}.png").is_file()]
    reference = request.anchor_frames[0] if request.anchor_frames else 0
    frames = {frame.frame_idx: frame for frame in [*diverse_frames(active, reference, 4),
                                                  *diverse_frames(manifest.frames, reference, 12)]}
    reports: dict[str, dict[str, float | bool]] = {}
    folder = request.output_dir / "quality_previews"
    folder.mkdir(parents=True, exist_ok=True)
    for idx, frame in frames.items():
        before, _, _ = renderer.render(base, frame)
        after, _, _ = renderer.render(updated, frame)
        mask_path = request.masks_dir / f"{idx:06d}.png"
        if mask_path.is_file() and idx not in request.exclude_frames:
            with Image.open(mask_path) as image:
                mask = np.asarray(image, dtype=np.uint8) == 255
        else:
            mask = np.zeros(before.shape[:2], dtype=bool)
        before_u8 = np.rint(before.clip(0, 1) * 255).astype(np.uint8)
        target = before.copy()
        if request.operation == "recolor":
            target[mask] = recolor_rgb(before_u8[mask], request.color_hex, request.color_strength) / 255.0
        else:
            target_path = _resolve_frame_file(request.edited_frames_dir, idx, frame.image_relpath)
            if target_path is not None:
                with Image.open(target_path) as image:
                    target = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        reports[str(idx)] = image_acceptance(before, after, target, mask)
        Image.fromarray(before_u8).save(folder / f"{idx:06d}_before.png")
        Image.fromarray(np.rint(after.clip(0, 1) * 255).astype(np.uint8)).save(folder / f"{idx:06d}_after.png")
    passed = bool(reports) and all(report["passed"] for report in reports.values())
    return {"status": "passed" if passed else "failed", "views": reports,
            "protocol": "calibrated capture-camera sweep; manual novel-view review required",
            "thresholds": {"background_mae": 0.015, "background_p99": 0.08, "minimum_target_improvement": 0.1}}
