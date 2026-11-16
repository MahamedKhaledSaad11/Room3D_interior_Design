from __future__ import annotations

import json
import uuid
from typing import BinaryIO

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..models import EditSession, Project, SceneVersion, Task
from ..security import get_current_user
from ..services.celery_bridge import dispatch_task_if_configured
from ..storage import get_storage

router = APIRouter(prefix="/api/projects", tags=["projects"])
_EXTENSIONS = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"}


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)


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
            "active_version_id": project.active_version_id, "created_at": project.created_at}


def _task_item(task: Task) -> dict[str, object]:
    try:
        payload = json.loads(task.payload_json or "{}")
    except json.JSONDecodeError:
        payload = {}
    telemetry = payload.get("telemetry") if isinstance(payload, dict) else None
    return {
        "id": task.id,
        "type": task.task_type,
        "status": task.status,
        "stage": task.current_stage,
        "progress_pct": task.progress_pct,
        "message": task.message,
        "error_detail": task.message if task.status == "failed" else None,
        "telemetry": telemetry,
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
    versions = session.scalars(select(SceneVersion).where(SceneVersion.project_id == project.id)).all()
    tasks = session.scalars(select(Task).where(Task.project_id == project.id).order_by(Task.created_at.desc())).all()
    edits = session.scalars(select(EditSession).where(EditSession.project_id == project.id).order_by(EditSession.created_at.desc())).all()
    return {
        **_view(project),
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
    }
    project.video_relpath = key
    task = Task(project_id=project.id, task_type="reconstruct_week1", payload_json=json.dumps(payload))
    session.add(task)
    session.commit()
    dispatch_task_if_configured(task.id, task.task_type, settings, payload)
    return {"project_id": project.id, "video_key": key, "version_id": version_id, "bytes": written, "task_id": task.id}
