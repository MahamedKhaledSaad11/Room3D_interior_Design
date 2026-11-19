from __future__ import annotations

import json
import uuid
import time
from typing import BinaryIO, Literal

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..models import EditSession, Project, SceneVersion, Task, TaskAttempt
from ..security import create_asset_token, get_current_user
from ..services.celery_bridge import dispatch_task_if_configured
from ..services.project_control import control_project, lock_project
from ..services.queue import _reclaim_stale_tasks
from ..storage import get_storage

router = APIRouter(prefix="/api/projects", tags=["projects"])
_EXTENSIONS = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class ProjectControl(BaseModel):
    action: Literal["pause", "resume", "stop"]


class UploadTiming(BaseModel):
    browser_transfer_sec: float = Field(ge=0, le=86400, allow_inf_nan=False)
    browser_roundtrip_sec: float = Field(ge=0, le=86400, allow_inf_nan=False)


@router.post("/{project_id}/tasks/{task_id}/upload-timing")
def upload_timing(project_id: str, task_id: str, body: UploadTiming,
                  session: Session = Depends(get_db), user_id: str | None = Depends(get_current_user)) -> dict[str, bool]:
    _project(session, project_id, user_id)
    task = session.get(Task, task_id)
    if task is None or task.project_id != project_id or task.task_type != "reconstruct_week1":
        raise HTTPException(404, "Upload task not found")
    payload = json.loads(task.payload_json or "{}")
    payload["upload_metrics"] = {**payload.get("upload_metrics", {}), **body.model_dump(), "browser_reported": True}
    task.payload_json = json.dumps(payload)
    session.commit()
    return {"recorded": True}


class BoundedStream:
    def __init__(self, source: BinaryIO, limit: int) -> None:
        self.source, self.limit, self.total = source, limit, 0

    def read(self, size: int = -1) -> bytes:
        remaining = self.limit + 1 - self.total
        chunk = self.source.read(min(size, remaining) if size >= 0 else remaining)
        self.total += len(chunk)
        if self.total > self.limit:
            raise ValueError("upload exceeds MAX_UPLOAD_BYTES")
        return chunk


def _project(session: Session, project_id: str, user_id: str | None) -> Project:
    project = session.get(Project, project_id)
    if not project or (user_id and project.owner_id != user_id):
        raise HTTPException(status_code=404, detail="project not found")
    return project


def _view(project: Project) -> dict[str, object]:
    return {"id": project.id, "name": project.name, "status": project.status,
            "processing_status": project.processing_status,
            "active_version_id": project.active_version_id, "created_at": project.created_at}


def _task_item(task: Task) -> dict[str, object]:
    try:
        payload = json.loads(task.payload_json or "{}")
    except json.JSONDecodeError:
        payload = {}
    telemetry = payload.get("telemetry") if isinstance(payload, dict) else None
    result = json.loads(task.result_json or "{}")
    return {
        "id": task.id,
        "type": task.task_type,
        "status": task.status,
        "stage": task.current_stage,
        "progress_pct": task.progress_pct,
        "message": task.message,
        "error_detail": task.message if task.status == "failed" else None,
        "telemetry": telemetry,
        "metrics": result,
        "upload_metrics": payload.get("upload_metrics", {}),
        "updated_at": task.updated_at,
    }


@router.get("")
def list_projects(session: Session = Depends(get_db), user_id: str | None = Depends(get_current_user)) -> list[dict[str, object]]:
    query = select(Project).order_by(Project.created_at.desc())
    if user_id:
        query = query.where(Project.owner_id == user_id)
    return [_view(project) for project in session.scalars(query)]


@router.post("", status_code=201)
def create_project(body: ProjectCreate, session: Session = Depends(get_db), user_id: str | None = Depends(get_current_user)) -> dict[str, object]:
    project = Project(name=body.name, owner_id=user_id, status="queued")
    session.add(project)
    session.commit()
    return _view(project)


@router.get("/{project_id}")
def get_project(project_id: str, session: Session = Depends(get_db), user_id: str | None = Depends(get_current_user)) -> dict[str, object]:
    project = _project(session, project_id, user_id)
    _reclaim_stale_tasks(session, project_id=project.id)
    versions = session.scalars(select(SceneVersion).where(SceneVersion.project_id == project.id)).all()
    tasks = session.scalars(select(Task).where(Task.project_id == project.id).order_by(Task.created_at.desc())).all()
    edits = session.scalars(select(EditSession).where(EditSession.project_id == project.id).order_by(EditSession.created_at.desc())).all()
    return {
        **_view(project),
        "asset_access": create_asset_token(user_id, project_id) if user_id else None,
        "versions": [{"id": v.id, "label": v.label} for v in versions],
        "tasks": [
            _task_item(t)
            for t in tasks
        ],
        "edit_sessions": [
            {
                "id": e.id,
                "status": e.status,
                "base_version_id": e.base_version_id,
                "target_version_id": e.target_version_id,
                "output_version_id": e.target_version_id,
                "ref_frame_idx": e.ref_frame_idx,
                "prompt": e.prompt,
                "chosen_seed_id": e.chosen_seed_id,
                "preview_seeds": e.preview_seeds_json or {},
                "mask_revision": int(e.mask_revision or 0),
                "preview_gallery": json.loads(e.preview_meta_json or "{}").get("preview_gallery", {}),
                "masks_summary": json.loads(e.masks_summary_json or "{}"),
            }
            for e in edits
        ],
    }


@router.post("/{project_id}/upload")
def upload_project(
    project_id: str,
    file: UploadFile,
    trim_start_sec: float = Form(0.0),
    trim_end_sec: float | None = Form(None),
    target_fps: float | None = Form(None),
    max_frames: int | None = Form(None),
    pose_backend: str = Form("colmap_global"),
    session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> dict[str, object]:
    project = _project(session, project_id, user_id)
    if project.processing_status != "running":
        raise HTTPException(409, "Resume project processing before uploading a video")
    filename = file.filename or "upload"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in _EXTENSIONS or (file.content_type and file.content_type != _EXTENSIONS[suffix]):
        raise HTTPException(status_code=415, detail="unsupported video type")
    prefix = file.file.read(12)
    valid_magic = (suffix == ".webm" and prefix.startswith(b"\x1a\x45\xdf\xa3")) or (suffix in {".mp4", ".mov"} and prefix[4:8] == b"ftyp")
    if not valid_magic:
        raise HTTPException(status_code=415, detail="invalid video container")
    file.file.seek(0)
    version_id = str(uuid.uuid4())
    key = f"{project.id}/input/{version_id}{suffix}"
    settings = get_settings()
    storage = get_storage(settings)
    started = time.perf_counter()
    try:
        written = storage.put_stream(key, BoundedStream(file.file, settings.max_upload_bytes))
    except ValueError as exc:
        try:
            storage.delete_prefix(key)
        except ValueError:
            pass
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    payload = {
        "input_key": key,
        "project_id": project.id,
        "version_id": version_id,
        "source_video_key": key,
        "trim_start_sec": trim_start_sec,
        "trim_end_sec": trim_end_sec,
        "target_fps": target_fps,
        "max_frames": max_frames,
        "pose_backend": pose_backend,
        "upload_metrics": {"server_storage_sec": round(time.perf_counter() - started, 4), "bytes": written},
    }
    project.video_relpath = key
    task = Task(project_id=project.id, task_type="reconstruct_week1", payload_json=json.dumps(payload))
    session.add(task)
    session.commit()
    dispatch_task_if_configured(task.id, task.task_type, settings, payload)
    return {"project_id": project.id, "video_key": key, "version_id": version_id, "bytes": written, "task_id": task.id}


@router.post("/{project_id}/processing")
def update_processing(
    project_id: str, body: ProjectControl, session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> dict[str, object]:
    project = _project(session, project_id, user_id)
    try:
        control_project(session, project, body.action)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _view(project)


@router.delete("/{project_id}")
def delete_project(
    project_id: str, session: Session = Depends(get_db),
    user_id: str | None = Depends(get_current_user),
) -> dict[str, bool]:
    project = _project(session, project_id, user_id)
    lock_project(session, project.id)
    _reclaim_stale_tasks(session, project_id=project.id)
    project.processing_status = "deleting"
    session.flush()
    active = session.scalar(select(Task.id).where(Task.project_id == project.id,
                            Task.status.in_(("claimed", "running", "pausing", "stopping"))))
    if active:
        raise HTTPException(409, "Stop processing and wait for the worker to acknowledge before deleting.")
    get_storage(get_settings()).delete_prefix(project.id)
    task_ids = select(Task.id).where(Task.project_id == project.id)
    session.execute(delete(TaskAttempt).where(TaskAttempt.task_id.in_(task_ids)))
    for model in (Task, EditSession, SceneVersion):
        session.execute(delete(model).where(model.project_id == project.id))
    session.delete(project)
    session.commit()
    return {"deleted": True}
