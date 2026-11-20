from __future__ import annotations

import hashlib
import json
import sys
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..models import EditSession, Project, SceneVersion, Task
from ..schemas import (
    ApplyEditRequest,
    ApplyEditResponse,
    ClickResolveResponse,
    EditFrameItem,
    EditFramesResponse,
    EditSessionOut,
    EditPreviewSettings,
    MaskUpdateResponse,
    PreviewSeedsRequest,
    RefineRequest,
    ResolveClickRequest,
    TaskViewResponse,
)
from ..security import get_current_user
from ..services.celery_bridge import dispatch_task_if_configured
from ..services.edit_coverage import mask_coverage, mask_state
from ..services.project_control import lock_project
from ..services.queue import invalidate_edit_previews, masks_digest
from ..storage import get_storage
from .projects import _project

_WORKER_SRC = Path(__file__).resolve().parents[3] / "worker"
if str(_WORKER_SRC) not in sys.path:
    sys.path.insert(0, str(_WORKER_SRC))

from room3d_worker.common.click_resolve import resolve_viewer_click  # noqa: E402

router = APIRouter(tags=["edits"])
_MAX_MASK_BYTES = 16 * 1024 * 1024


def _load_version_manifest(
    project: Project, version_id: str | None, session: Session
) -> tuple[SceneVersion | None, dict[str, Any] | None]:
    target_id = version_id or project.active_version_id
    if not target_id:
        return None, None
    version = session.scalar(
        select(SceneVersion).where(
            SceneVersion.id == target_id,
            SceneVersion.project_id == project.id,
        )
    )
    if not version or not version.manifest_relpath:
        return version, None
    try:
        with get_storage(get_settings()).open_read(version.manifest_relpath) as stream:
            data = json.load(stream)
            return version, (data if isinstance(data, dict) else None)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return version, None


def _manifest_dims(manifest: dict[str, Any] | None) -> tuple[int, int]:
    if not manifest:
        return 640, 480
    return max(1, int(manifest.get("width") or 640)), max(1, int(manifest.get("height") or 480))


def _frame_relpath(manifest: dict[str, Any] | None, frame_idx: int | None) -> str | None:
    if manifest is None or frame_idx is None:
        return None
    for frame in manifest.get("frames", []):
        if int(frame.get("frame_idx", -1)) == frame_idx:
            rel = frame.get("image_relpath")
            return str(rel) if rel else None
    return None


def _image_key(project_id: str, version_id: str | None, relpath: str | None) -> str | None:
    if not version_id or not relpath:
        return None
    return f"{project_id}/{version_id}/{relpath.lstrip('/')}"


def _serialize_points(raw_points: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for pt in raw_points:
        if "u_norm" in pt and pt["u_norm"] is not None and "v_norm" in pt and pt["v_norm"] is not None:
            u_norm = float(pt["u_norm"])
            v_norm = float(pt["v_norm"])
            x = float(pt.get("x") if pt.get("x") is not None else u_norm * width)
            y = float(pt.get("y") if pt.get("y") is not None else v_norm * height)
        else:
            x = float(pt["x"])
            y = float(pt["y"])
            u_norm = min(1.0, max(0.0, x / max(width, 1)))
            v_norm = min(1.0, max(0.0, y / max(height, 1)))
        serialized.append({"x": x, "y": y, "u_norm": u_norm, "v_norm": v_norm, "label": int(pt["label"])})
    return serialized


def _resolve_target(
    manifest: dict[str, Any] | None, body: ResolveClickRequest
) -> tuple[int, float, float, float, Literal["projection", "nearest_snap", "frame_picker"]]:
    width, height = _manifest_dims(manifest)
    if body.frame_idx is not None:
        frames = (manifest or {}).get("frames") or []
        matched = next((f for f in frames if int(f.get("frame_idx", -1)) == body.frame_idx and not f.get("excluded")), None)
        if matched is None:
            raise HTTPException(status_code=422, detail=f"frame_idx {body.frame_idx} not found in manifest")
        u_norm = 0.5 if body.u_norm is None else float(body.u_norm)
        v_norm = 0.5 if body.v_norm is None else float(body.v_norm)
        return body.frame_idx, u_norm * width, v_norm * height, 1.0, "frame_picker"

    world_hit = body.worldHitColmap if body.worldHitColmap is not None else body.point_3d
    ray_origin = body.ray_origin if body.ray_origin is not None else body.rayOriginColmap
    ray_dir = body.ray_dir if body.ray_dir is not None else body.rayDirColmap
    frames = [f for f in (manifest or {}).get("frames", []) if not f.get("excluded")]
    if not frames or manifest is None:
        idx = body.nearestFrameIdx if body.nearestFrameIdx is not None else 0
        intr = (manifest or {}).get("intrinsics") or {}
        return idx, float(intr.get("cx", width * 0.5)), float(intr.get("cy", height * 0.5)), 0.0, "nearest_snap"
    res = resolve_viewer_click(
        manifest=manifest,
        world_hit_colmap=world_hit,
        ray_origin_colmap=ray_origin,
        ray_dir_colmap=ray_dir,
        nearest_frame_idx=body.nearestFrameIdx,
    )
    return res.ref_frame_idx, res.u, res.v, res.score, res.method


def _edit_for_user(session: Session, edit_id: str, user_id: str | None) -> tuple[EditSession, Project]:
    edit = session.get(EditSession, edit_id)
    if not edit:
        raise HTTPException(status_code=404, detail="edit session not found")
    return edit, _project(session, edit.project_id, user_id)


def _edit_to_schema(edit: EditSession) -> EditSessionOut:
    return EditSessionOut(
        id=edit.id,
        project_id=edit.project_id,
        base_version_id=edit.base_version_id,
        target_version_id=edit.target_version_id,
        output_version_id=edit.target_version_id,
        ref_frame_idx=edit.ref_frame_idx,
        prompt=edit.prompt,
        chosen_seed_id=edit.chosen_seed_id,
        status=edit.status,
        preview_seeds=dict(edit.preview_seeds_json or {}),
        masks_summary=json.loads(edit.masks_summary_json or "{}"),
        mask_revision=int(edit.mask_revision or 0),
        created_at=edit.created_at,
        updated_at=edit.updated_at,
        preview_gallery=json.loads(edit.preview_meta_json or "{}").get("preview_gallery", {}),
        preview_settings=EditPreviewSettings.model_validate_json(edit.preview_meta_json) if edit.preview_meta_json else None,
    )


def _task_view(task: Task) -> TaskViewResponse:
    return TaskViewResponse(
        id=task.id,
        type=task.task_type,
        status=task.status,
        progress_pct=task.progress_pct,
        stage=task.current_stage,
        message=task.message,
    )


def _queue_edit_task(session: Session, edit: EditSession, task_type: str, payload: dict[str, Any]) -> Task:
    project = lock_project(session, edit.project_id)
    if project is None or project.processing_status != "running":
        raise HTTPException(409, "Resume project processing before starting or changing an edit")
    task = Task(
        project_id=edit.project_id,
        edit_session_id=edit.id,
        task_type=task_type,
        status="queued",
        payload_json=json.dumps(payload),
    )
    session.add(task)
    session.commit()
    session.refresh(task)
    dispatch_task_if_configured(task.id, task.task_type, get_settings(), payload)
    return task


def _validate_binary_mask_png(raw: bytes, expected_size: tuple[int, int] | None) -> bytes:
    if not raw:
        raise HTTPException(status_code=422, detail="Mask PNG upload is empty")
    try:
        with Image.open(BytesIO(raw)) as img:
            img.load()
            if img.format != "PNG" or img.mode != "L":
                raise HTTPException(status_code=422, detail=f"Mask must be 1-channel uint8 PNG ('L'), got {img.format}/{img.mode}")
            if img.width < 64 or img.height < 64 or img.width % 16 != 0 or img.height % 16 != 0:
                raise HTTPException(status_code=422, detail=f"Mask dimensions {img.size} must be >= 64 and divisible by 16")
            if expected_size is not None and img.size != expected_size:
                raise HTTPException(status_code=422, detail=f"Mask dimensions {img.size} do not match scene {expected_size}")
            if img.tobytes().translate(None, b"\x00\xff"):
                raise HTTPException(status_code=422, detail="Mask pixel values must be strictly 0 or 255")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail="Invalid PNG image upload") from exc
    return raw


async def _read_bounded_upload(upload: Any, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(65536):
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="Mask upload exceeds size limit")
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/api/projects/{project_id}/edits/resolve-click", status_code=201, response_model=ClickResolveResponse)
def resolve_click(
    project_id: str,
    body: ResolveClickRequest,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> ClickResolveResponse:
    project = _project(session, project_id, user_id)
    target_version_id = body.base_version_id or project.active_version_id
    version, manifest = _load_version_manifest(project, target_version_id, session)
    if body.base_version_id is not None and version is None:
        raise HTTPException(status_code=404, detail=f"base version '{body.base_version_id}' not found")
    ref_frame_idx, u, v, score, method = _resolve_target(manifest, body)
    width, height = _manifest_dims(manifest)
    initial_points = _serialize_points([{"x": u, "y": v, "label": 1}], width, height)
    resolved_base_version_id = version.id if version else target_version_id
    edit = EditSession(
        project_id=project.id,
        base_version_id=resolved_base_version_id,
        ref_frame_idx=ref_frame_idx,
        points_json=json.dumps(initial_points),
        masks_summary_json=json.dumps({}),
        preview_seeds_json={},
        status="segmenting",
    )
    session.add(edit)
    session.flush()
    relpath = _frame_relpath(manifest, ref_frame_idx)
    task = _queue_edit_task(
        session,
        edit,
        "segment_init",
        {
            "project_id": project.id,
            "edit_id": edit.id,
            "base_version_id": edit.base_version_id,
            "manifest_key": version.manifest_relpath if version else None,
            "ref_frame_idx": ref_frame_idx,
            "image_relpath": relpath,
            "image_key": _image_key(project.id, edit.base_version_id, relpath),
            "points": initial_points,
            "box_norm": None,
            "mask_out_key": f"{project.id}/edits/{edit.id}/masks/{ref_frame_idx:06d}.png",
            "mask_revision": int(edit.mask_revision or 0),
        },
    )
    return ClickResolveResponse(
        edit_id=edit.id,
        ref_frame_idx=ref_frame_idx,
        u=u,
        v=v,
        u_norm=initial_points[0]["u_norm"],
        v_norm=initial_points[0]["v_norm"],
        score=score,
        method=method,
        task_id=task.id,
        base_version_id=edit.base_version_id,
    )


@router.post("/api/edits/{edit_id}/refine", response_model=TaskViewResponse)
def refine_edit(
    edit_id: str,
    body: RefineRequest,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> TaskViewResponse:
    edit, project = _edit_for_user(session, edit_id, user_id)
    version, manifest = _load_version_manifest(project, edit.base_version_id, session)
    width, height = _manifest_dims(manifest)
    target_frame_idx = body.frame_idx if body.frame_idx is not None else (edit.ref_frame_idx if edit.ref_frame_idx is not None else 0)
    if manifest is not None and not any(int(f.get("frame_idx", -1)) == target_frame_idx for f in manifest.get("frames", [])):
        raise HTTPException(status_code=404, detail=f"frame_idx {target_frame_idx} not found in manifest")
    existing = _serialize_points(json.loads(edit.points_json or "[]"), width, height)
    added = _serialize_points([p.model_dump() for p in body.points], width, height)
    base_ref_idx = edit.ref_frame_idx if edit.ref_frame_idx is not None else 0
    task_points = [*existing, *added] if target_frame_idx == base_ref_idx else added
    if target_frame_idx == base_ref_idx:
        edit.points_json = json.dumps(task_points)
    invalidate_edit_previews(edit)
    edit.status = "segmenting"
    relpath = _frame_relpath(manifest, target_frame_idx)
    task = _queue_edit_task(
        session,
        edit,
        "segment_refine",
        {
            "project_id": project.id,
            "edit_id": edit.id,
            "base_version_id": edit.base_version_id or (version.id if version else None),
            "manifest_key": version.manifest_relpath if version else None,
            "ref_frame_idx": target_frame_idx,
            "image_relpath": relpath,
            "image_key": _image_key(project.id, edit.base_version_id or (version.id if version else None), relpath),
            "points": task_points,
            "box_norm": list(body.box_norm) if body.box_norm is not None else None,
            "mask_out_key": f"{project.id}/edits/{edit.id}/masks/{target_frame_idx:06d}.png",
            "mask_revision": int(edit.mask_revision or 0),
        },
    )
    return _task_view(task)


@router.post("/api/edits/{edit_id}/track", response_model=TaskViewResponse)
def track_edit(
    edit_id: str,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> TaskViewResponse:
    edit, project = _edit_for_user(session, edit_id, user_id)
    version, manifest = _load_version_manifest(project, edit.base_version_id, session)
    width, height = _manifest_dims(manifest)
    points = _serialize_points(json.loads(edit.points_json or "[]"), width, height)
    summary: dict[str, Any] = json.loads(edit.masks_summary_json or "{}")
    corrections = {
        k: v
        for k, v in summary.items()
        if isinstance(v, dict) and (v.get("status") in {"manual", "skipped"}
                                   or v.get("skip_frame") and v.get("reviewed") is True)
    }
    invalidate_edit_previews(edit)
    edit.status = "tracking"
    task = _queue_edit_task(
        session,
        edit,
        "track_propagate",
        {
            "project_id": project.id,
            "edit_id": edit.id,
            "base_version_id": edit.base_version_id or (version.id if version else None),
            "manifest_key": version.manifest_relpath if version else None,
            "ref_frame_idx": edit.ref_frame_idx,
            "points": points,
            "keyframe_corrections": corrections,
            "masks_prefix": f"{project.id}/edits/{edit.id}/masks",
            "mask_revision": int(edit.mask_revision or 0),
        },
    )
    return _task_view(task)


@router.get("/api/edits/{edit_id}/frames", response_model=EditFramesResponse)
def list_edit_frames(
    edit_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=100),
    failures_first: bool = True,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> EditFramesResponse:
    edit, project = _edit_for_user(session, edit_id, user_id)
    _, manifest = _load_version_manifest(project, edit.base_version_id, session)
    summary: dict[str, Any] = json.loads(edit.masks_summary_json or "{}")
    frames_meta = (manifest or {}).get("frames") or [
        {"frame_idx": int(i), "image_relpath": f"images/frame_{int(i):06d}.jpg"} for i in sorted(summary, key=int)
    ]
    start = (page - 1) * page_size
    if failures_first:
        frames_meta = sorted(frames_meta, key=lambda frame: (
            mask_state(summary.get(str(frame["frame_idx"]))) != "unresolved", int(frame["frame_idx"])))
    items: list[EditFrameItem] = []
    for item in frames_meta[start : start + page_size]:
        idx = int(item["frame_idx"])
        entry = summary.get(str(idx)) or {}
        status = mask_state(entry)
        items.append(
            EditFrameItem(
                frame_idx=idx,
                image_relpath=item.get("image_relpath"),
                mask_key=entry.get("mask_key"),
                mask_status=status,
                skip_frame=bool(entry.get("skip_frame", False)),
                area_px=entry.get("area_px"), visibility=str(entry.get("visibility", "unresolved")),
                confidence=entry.get("confidence"), reviewed=bool(entry.get("reviewed")),
                rejection_reason=entry.get("rejection_reason") or ("legacy_mask_requires_review" if "area_px" not in entry else None),
            )
        )
    coverage = mask_coverage([int(frame["frame_idx"]) for frame in frames_meta if not frame.get("excluded")], summary)
    return EditFramesResponse(edit_id=edit.id, page=page, page_size=page_size, total=len(frames_meta), frames=items,
                              coverage=coverage)


def _apply_skip_flag(session: Session, edit: EditSession, frame_idx: int, payload: Any) -> MaskUpdateResponse:
    if not isinstance(payload, dict) or not isinstance(payload.get("skip_frame"), bool):
        raise HTTPException(status_code=422, detail="Expected JSON body {skip_frame: boolean}")
    summary: dict[str, Any] = json.loads(edit.masks_summary_json or "{}")
    entry = dict(summary.get(str(frame_idx)) or {})
    skip_flag = payload["skip_frame"]
    entry["skip_frame"] = skip_flag
    entry["reviewed"] = skip_flag
    entry["visibility"] = "absent" if skip_flag else ("visible" if entry.get("area_px", 0) > 0 else "unresolved")
    entry["rejection_reason"] = None if skip_flag else "review_required"
    entry["status"] = "skipped" if skip_flag else ("manual" if entry.get("mask_key") else "empty")
    summary[str(frame_idx)] = entry
    edit.masks_summary_json = json.dumps(summary)
    invalidate_edit_previews(edit)
    session.commit()
    return MaskUpdateResponse(edit_id=edit.id, frame_idx=frame_idx, status=entry["status"], skip_frame=skip_flag, mask_key=entry.get("mask_key"))


async def _apply_mask_upload(
    session: Session, edit: EditSession, project: Project, frame_idx: int, request: Request
) -> MaskUpdateResponse:
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=422, detail="Missing 'file' multipart field")
    settings = get_settings()
    raw = await _read_bounded_upload(upload, min(settings.max_upload_bytes, _MAX_MASK_BYTES))
    _, manifest = _load_version_manifest(project, edit.base_version_id, session)
    expected_size = (int(manifest["width"]), int(manifest["height"])) if manifest else None
    validated = _validate_binary_mask_png(raw, expected_size)
    mask_key = f"{project.id}/edits/{edit.id}/masks/{uuid.uuid4().hex}/{frame_idx:06d}.png"
    get_storage(settings).put_stream(mask_key, BytesIO(validated))
    summary: dict[str, Any] = json.loads(edit.masks_summary_json or "{}")
    entry = {"status": "manual", "mask_key": mask_key, "skip_frame": False}
    with Image.open(BytesIO(validated)) as image:
        area = image.tobytes().count(255)
    entry.update(area_px=area, content_sha256=hashlib.sha256(validated).hexdigest(),
                 visibility="visible" if area else "unresolved", reviewed=True,
                 rejection_reason=None if area else "empty_manual_mask")
    summary[str(frame_idx)] = entry
    edit.masks_summary_json = json.dumps(summary)
    invalidate_edit_previews(edit)
    session.commit()
    return MaskUpdateResponse(edit_id=edit.id, frame_idx=frame_idx, status="manual", skip_frame=False, mask_key=mask_key)


@router.put("/api/edits/{edit_id}/masks/{frame_idx}", response_model=MaskUpdateResponse)
async def update_edit_mask(
    edit_id: str,
    frame_idx: int,
    request: Request,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> MaskUpdateResponse:
    if frame_idx < 0:
        raise HTTPException(status_code=422, detail="frame_idx must be non-negative")
    edit, project = _edit_for_user(session, edit_id, user_id)
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        return _apply_skip_flag(session, edit, frame_idx, await request.json())
    if "multipart/form-data" in content_type:
        return await _apply_mask_upload(session, edit, project, frame_idx, request)
    raise HTTPException(status_code=415, detail="Use application/json or multipart/form-data")


@router.post("/api/edits/{edit_id}/preview-seeds", response_model=TaskViewResponse)
def preview_seeds(
    edit_id: str,
    body: PreviewSeedsRequest,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> TaskViewResponse:
    edit, project = _edit_for_user(session, edit_id, user_id)
    version, manifest = _load_version_manifest(project, edit.base_version_id, session)
    ref_idx = edit.ref_frame_idx if edit.ref_frame_idx is not None else 0
    summary: dict[str, Any] = json.loads(edit.masks_summary_json or "{}")
    ref_entry = summary.get(str(ref_idx)) or {}
    frame_ids = [int(frame["frame_idx"]) for frame in (manifest or {}).get("frames", []) if not frame.get("excluded")]
    if not mask_coverage(frame_ids, summary).ready:
        raise HTTPException(422, "Resolve missing/uncertain masks and confirm true absence before previewing")
    mask_key = ref_entry.get("mask_key") or f"{project.id}/edits/{edit.id}/masks/{ref_idx:06d}.png"
    clean_prompt = body.prompt.strip()
    edit.prompt = clean_prompt
    edit.preview_generation = int(edit.preview_generation or 0) + 1
    edit.preview_seeds_json = {}
    digest = masks_digest(edit.masks_summary_json, edit.points_json)
    mask_rev = int(edit.mask_revision or 0)
    edit.preview_meta_json = json.dumps(
        {
            "preview_generation": edit.preview_generation,
            "mask_revision": mask_rev,
            "masks_digest": digest,
            "prompt": clean_prompt,
            "negative_prompt": body.negative_prompt or "",
            "model_variant": body.model_variant,
            "completed": False,
            "operation": body.operation, "color_hex": body.color_hex, "color_strength": body.color_strength,
            "quality_preset": body.quality_preset,
        }
    )
    edit.status = "previewing"
    task = _queue_edit_task(
        session,
        edit,
        "diffusion_preview",
        {
            "project_id": project.id,
            "edit_id": edit.id,
            "base_version_id": edit.base_version_id or (version.id if version else None),
            "manifest_key": version.manifest_relpath if version else None,
            "ref_frame_idx": ref_idx,
            "prompt": clean_prompt,
            "negative_prompt": body.negative_prompt,
            "model_variant": body.model_variant,
            "seeds": body.seeds,
            "mask_key": mask_key,
            "previews_prefix": f"{project.id}/edits/{edit.id}/seed_previews/g{edit.preview_generation}",
            "preview_generation": edit.preview_generation,
            "mask_revision": mask_rev,
            "masks_digest": digest,
            "operation": body.operation, "color_hex": body.color_hex, "color_strength": body.color_strength,
            "quality_preset": body.quality_preset,
            "masks_summary": summary,
        },
    )
    return _task_view(task)


@router.post("/api/edits/{edit_id}/apply", response_model=ApplyEditResponse)
def apply_edit(
    edit_id: str,
    body: ApplyEditRequest,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> ApplyEditResponse:
    edit, project = _edit_for_user(session, edit_id, user_id)
    active_task = session.scalar(
        select(Task).where(
            Task.edit_session_id == edit.id,
            Task.task_type == "rebuild_week2",
            Task.status.in_(("queued", "claimed", "running", "paused", "pausing", "stopping")),
        )
    )
    if active_task is not None:
        raise HTTPException(status_code=409, detail="An apply/rebuild operation is already in progress for this edit")
    preview_seeds = dict(edit.preview_seeds_json or {})
    try:
        meta = json.loads(edit.preview_meta_json or "{}")
    except json.JSONDecodeError:
        meta = {}
    current_digest = masks_digest(edit.masks_summary_json, edit.points_json)
    if (
        not preview_seeds
        or not isinstance(meta, dict)
        or meta.get("completed") is not True
        or meta.get("mask_revision") != int(edit.mask_revision or 0)
        or meta.get("masks_digest") != current_digest
    ):
        raise HTTPException(status_code=422, detail="Previews are missing or stale after mask modification")
    if (body.operation != meta.get("operation", "generative") or
            body.operation == "recolor" and (body.color_hex != meta.get("color_hex") or
                                            body.color_strength != meta.get("color_strength"))):
        raise HTTPException(422, "Operation/color changed; regenerate the diverse-view previews")
    if body.operation == "recolor" and body.rebuild_mode != "masked_finetune":
        raise HTTPException(422, "Recolor requires a frozen-geometry masked edit")
    if body.operation == "recolor" and (not body.previews_reviewed or len(meta.get("preview_gallery") or {}) < 2):
        raise HTTPException(422, "Review at least two diverse-view recolor previews before building")
    if body.quality_preset != meta.get("quality_preset", "fast"):
        raise HTTPException(422, "Sampling preset changed; regenerate previews")
    if body.prompt is not None and body.prompt.strip() != meta.get("prompt", ""):
        raise HTTPException(status_code=422, detail="Prompt does not match generated seed previews; regenerate previews first")
    if body.model_variant is not None and body.model_variant != meta.get("model_variant"):
        raise HTTPException(
            status_code=422,
            detail="model_variant does not match generated seed previews; regenerate previews first",
        )
    if body.negative_prompt is not None and body.negative_prompt != (meta.get("negative_prompt") or ""):
        raise HTTPException(
            status_code=422,
            detail="negative_prompt does not match generated seed previews; regenerate previews first",
        )
    prompt = (body.prompt.strip() if body.prompt is not None else str(meta.get("prompt") or edit.prompt or "")).strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Edit prompt must be set before applying diffusion edits")
    effective_model_variant = body.model_variant or meta.get("model_variant") or "flux2_klein_4b"
    effective_negative_prompt = (
        body.negative_prompt if body.negative_prompt is not None else meta.get("negative_prompt")
    )
    version, manifest = _load_version_manifest(project, edit.base_version_id, session)
    base_version_id = edit.base_version_id or (version.id if version else None)
    target_version_id = edit.target_version_id or str(uuid.uuid4())
    colliding = session.get(SceneVersion, target_version_id)
    if colliding is not None:
        target_version_id = str(uuid.uuid4())
    masks_summary: dict[str, Any] = json.loads(edit.masks_summary_json or "{}")
    frame_ids = [int(frame["frame_idx"]) for frame in (manifest or {}).get("frames", []) if not frame.get("excluded")]
    coverage = mask_coverage(frame_ids, masks_summary)
    if not coverage.ready:
        raise HTTPException(422, f"Track and review all views before applying: {coverage.masked} masked, "
                            f"{coverage.skipped} skipped, {coverage.missing} unreviewed. "
                            "Mark missing objects or explicitly skip views where the object is absent.")
    exclude_frames = sorted(
        int(k) for k, v in masks_summary.items() if isinstance(v, dict) and bool(v.get("skip_frame"))
    )
    seed_key = body.chosen_seed_id
    stripped_seed = seed_key.removeprefix("seed_")
    preview_key: str | None = None
    for candidate in (seed_key, stripped_seed, f"seed_{stripped_seed}"):
        if candidate in preview_seeds and isinstance(preview_seeds[candidate], str) and preview_seeds[candidate]:
            preview_key = preview_seeds[candidate]
            break
    if preview_key is None:
        raise HTTPException(
            status_code=422,
            detail=f"chosen_seed_id '{seed_key}' not found in generated previews ({sorted(preview_seeds)})",
        )
    meta_seeds = meta.get("preview_seeds")
    if isinstance(meta_seeds, dict):
        meta_preview_key = next(
            (meta_seeds[c] for c in (seed_key, stripped_seed, f"seed_{stripped_seed}") if c in meta_seeds),
            None,
        )
        if meta_preview_key != preview_key:
            raise HTTPException(
                status_code=422,
                detail="Preview metadata does not match active generated seed key; regenerate previews",
            )
    edit.prompt = prompt
    edit.chosen_seed_id = body.chosen_seed_id
    edit.target_version_id = target_version_id
    edit.status = "rebuilding"
    common_payload: dict[str, Any] = {
        "project_id": project.id,
        "edit_id": edit.id,
        "base_version_id": base_version_id,
        "manifest_key": version.manifest_relpath if version else None,
        "ref_frame_idx": edit.ref_frame_idx if edit.ref_frame_idx is not None else 0,
        "prompt": prompt,
        "negative_prompt": effective_negative_prompt,
        "model_variant": effective_model_variant,
        "selected_seed": seed_key,
        "chosen_seed_id": seed_key,
        "preview_key": preview_key,
        "masks_summary": masks_summary,
        "masks_prefix": f"{project.id}/edits/{edit.id}/masks",
        "edited_frames_prefix": f"{project.id}/edits/{edit.id}/edited_frames",
        "rebuild_mode": body.rebuild_mode,
        "exclude_frames": exclude_frames,
        "target_version_id": target_version_id,
        "new_version_id": target_version_id,
        "rebuild_safeguards": body.rebuild_safeguards,
        "unmasked_weight": body.unmasked_weight,
        "operation": body.operation, "color_hex": body.color_hex, "color_strength": body.color_strength,
        "mask_revision": int(edit.mask_revision or 0), "masks_digest": current_digest,
        "quality_preset": body.quality_preset,
    }
    diffusion_task = Task(
        project_id=edit.project_id,
        edit_session_id=edit.id,
        task_type="diffusion_apply",
        status="queued",
        payload_json=json.dumps(common_payload),
    )
    if body.operation != "recolor":
        session.add(diffusion_task)
        session.flush()
    rebuild_payload = dict(common_payload)
    if body.operation != "recolor":
        rebuild_payload["depends_on_task_id"] = diffusion_task.id
    rebuild_task = _queue_edit_task(session, edit, "rebuild_week2", rebuild_payload)
    if body.operation != "recolor":
        dispatch_task_if_configured(diffusion_task.id, diffusion_task.task_type, get_settings(), common_payload)
    return ApplyEditResponse(
        **_task_view(rebuild_task).model_dump(),
        edit_id=edit.id,
        target_version_id=target_version_id,
        diffusion_task_id=diffusion_task.id if body.operation != "recolor" else None,
    )


@router.get("/api/projects/{project_id}/edits", response_model=list[EditSessionOut])
def list_project_edits(
    project_id: str,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> list[EditSessionOut]:
    _project(session, project_id, user_id)
    edits = session.scalars(
        select(EditSession).where(EditSession.project_id == project_id).order_by(EditSession.created_at.desc())
    ).all()
    return [_edit_to_schema(e) for e in edits]


@router.get("/api/projects/{project_id}/edits/{edit_id}", response_model=EditSessionOut)
def get_project_edit(
    project_id: str,
    edit_id: str,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> EditSessionOut:
    _project(session, project_id, user_id)
    edit = session.get(EditSession, edit_id)
    if not edit or edit.project_id != project_id:
        raise HTTPException(status_code=404, detail="edit session not found")
    return _edit_to_schema(edit)


@router.get("/api/edits/{edit_id}", response_model=EditSessionOut)
def get_edit(
    edit_id: str,
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> EditSessionOut:
    edit, _ = _edit_for_user(session, edit_id, user_id)
    return _edit_to_schema(edit)
