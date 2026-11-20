from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image

from room3d_worker.pipeline.s06_diffusion import DiffusionConfig


@dataclass(frozen=True)
class StageBudget:
    absolute_sec: int
    stall_sec: int
    operations: int = 0


def positive_setting(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def stage_budget(stage: str, spec: Mapping[str, Any], default_sec: int, stall_sec: int) -> StageBudget:
    if not stage.startswith("diffusion_"):
        return StageBudget(default_sec, stall_sec)
    cfg = DiffusionConfig(model_variant=spec.get("model_variant", "flux2_klein_4b"),
                          quality_preset=spec.get("quality_preset", "fast"))
    operations = _operation_count(stage, spec)
    setup = positive_setting("ROOM3D_DIFFUSION_SETUP_BUDGET_SEC", 1800)
    per_step = positive_setting("ROOM3D_DIFFUSION_STEP_BUDGET_SEC", 30)
    cap = positive_setting("ROOM3D_DIFFUSION_MAX_BUDGET_SEC", 36000)
    absolute = min(cap, setup + max(1, operations) * cfg.num_inference_steps * per_step)
    return StageBudget(absolute, stall_sec, operations)


def _operation_count(stage: str, spec: Mapping[str, Any]) -> int:
    if stage == "diffusion_preview":
        seeds = spec.get("seeds")
        return len(seeds) if isinstance(seeds, list) else 1
    masks, frames = spec.get("mask_paths"), spec.get("frame_paths")
    if not isinstance(masks, dict) or not isinstance(frames, dict):
        return 0
    excluded = {int(idx) for idx in spec.get("skip_frames", [])}
    reference = int(spec["ref_frame_idx"])
    count = 0
    for idx, path in masks.items():
        if idx not in frames or int(idx) in excluded or int(idx) == reference:
            continue
        with Image.open(str(path)) as mask:
            count += int(mask.getbbox() is not None)
    # The production apply path is sequential; each nonempty view is one inference.
    return count
