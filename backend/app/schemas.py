from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .services.edit_coverage import EditCoverage


class ResolveClickRequest(BaseModel):
    screenX: float = 0.0
    screenY: float = 0.0
    worldHitColmap: tuple[float, float, float] | None = None
    rayOriginColmap: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rayDirColmap: tuple[float, float, float] = (0.0, 0.0, -1.0)
    nearestFrameIdx: int | None = None
    point_3d: tuple[float, float, float] | None = None
    ray_origin: tuple[float, float, float] | None = None
    ray_dir: tuple[float, float, float] | None = None
    u_norm: float | None = Field(default=None, ge=0.0, le=1.0)
    v_norm: float | None = Field(default=None, ge=0.0, le=1.0)
    frame_idx: int | None = Field(default=None, ge=0)
    base_version_id: str | None = None


class ClickResolveResponse(BaseModel):
    edit_id: str
    ref_frame_idx: int
    u: float
    v: float
    u_norm: float
    v_norm: float
    score: float
    method: Literal["projection", "nearest_snap", "frame_picker"]
    task_id: str
    base_version_id: str | None = None


class PromptPoint(BaseModel):
    x: float | None = Field(default=None, ge=0)
    y: float | None = Field(default=None, ge=0)
    u_norm: float | None = Field(default=None, ge=0.0, le=1.0)
    v_norm: float | None = Field(default=None, ge=0.0, le=1.0)
    label: Literal[0, 1]

    @model_validator(mode="after")
    def _require_coords(self) -> PromptPoint:
        has_px = self.x is not None and self.y is not None
        has_norm = self.u_norm is not None and self.v_norm is not None
        if not has_px and not has_norm:
            raise ValueError("PromptPoint requires either (x, y) or (u_norm, v_norm)")
        return self


class RefineRequest(BaseModel):
    points: list[PromptPoint] = Field(min_length=1)
    box_norm: tuple[float, float, float, float] | None = None
    frame_idx: int | None = Field(default=None, ge=0)


class PreviewSeedsRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    negative_prompt: str | None = Field(default=None, max_length=1000)
    seeds: list[int] = Field(default_factory=lambda: [101, 202, 303, 404], min_length=1, max_length=8)
    model_variant: Literal["flux2_klein_4b", "sdxl_inpaint_t4"] = "flux2_klein_4b"
    operation: Literal["recolor", "generative"] = "generative"
    color_hex: str = Field(default="#2563eb", pattern=r"^#[0-9a-fA-F]{6}$")
    color_strength: float = Field(default=1.0, ge=0.05, le=1.0, allow_inf_nan=False)
    quality_preset: Literal["fast", "quality"] = "fast"


class ApplyEditRequest(BaseModel):
    chosen_seed_id: str = Field(min_length=1)
    rebuild_mode: Literal["masked_finetune", "from_scratch"] = "masked_finetune"
    prompt: str | None = Field(default=None, max_length=1000)
    negative_prompt: str | None = Field(default=None, max_length=1000)
    model_variant: Literal["flux2_klein_4b", "sdxl_inpaint_t4"] | None = None
    rebuild_safeguards: bool = False
    unmasked_weight: float = Field(default=1.0, ge=1.0, allow_inf_nan=False)
    operation: Literal["recolor", "generative"] = "generative"
    color_hex: str = Field(default="#2563eb", pattern=r"^#[0-9a-fA-F]{6}$")
    color_strength: float = Field(default=1.0, ge=0.05, le=1.0, allow_inf_nan=False)
    quality_preset: Literal["fast", "quality"] = "fast"
    previews_reviewed: bool = False

    @model_validator(mode="after")
    def check_safeguard_mode(self) -> ApplyEditRequest:
        if self.rebuild_safeguards and self.rebuild_mode != "masked_finetune":
            raise ValueError("Background-preserving safeguards require masked_finetune mode")
        return self


class TaskViewResponse(BaseModel):
    id: str
    type: str
    status: str
    progress_pct: float
    stage: str | None
    message: str | None


class ApplyEditResponse(TaskViewResponse):
    edit_id: str
    target_version_id: str
    diffusion_task_id: str | None


class EditFrameItem(BaseModel):
    frame_idx: int
    image_relpath: str | None
    mask_key: str | None
    mask_status: str
    skip_frame: bool
    area_px: int | None = None
    visibility: str = "unresolved"
    confidence: float | None = None
    rejection_reason: str | None = None
    reviewed: bool = False


class EditFramesResponse(BaseModel):
    edit_id: str
    page: int
    page_size: int
    total: int
    frames: list[EditFrameItem]
    coverage: EditCoverage | None = None


class MaskUpdateResponse(BaseModel):
    edit_id: str
    frame_idx: int
    status: str
    skip_frame: bool
    mask_key: str | None = None


class EditSessionOut(BaseModel):
    id: str
    project_id: str
    base_version_id: str | None
    target_version_id: str | None
    output_version_id: str | None = None
    ref_frame_idx: int | None
    prompt: str | None
    chosen_seed_id: str | None
    status: str
    preview_seeds: dict[str, str] = Field(default_factory=dict)
    preview_gallery: dict[str, str] = Field(default_factory=dict)
    masks_summary: dict[str, Any] = Field(default_factory=dict)
    mask_revision: int = 0
    created_at: datetime | None
    updated_at: datetime | None


EditSummaryResponse = EditSessionOut
