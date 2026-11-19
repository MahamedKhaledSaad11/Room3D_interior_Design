from __future__ import annotations

import mimetypes
from collections.abc import Iterator
from typing import BinaryIO

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..models import Project, SceneVersion
from ..security import get_current_user, get_storage_actor
from ..services.project_control import lock_project
from ..storage import get_storage
from ..storage.base import validate_key
from .projects import _project

router = APIRouter(tags=["scenes"])


@router.get("/api/projects/{project_id}/versions")
def versions(project_id: str, session: Session = Depends(get_db), user_id: str | None = Depends(get_current_user)) -> list[dict[str, object]]:
    _project(session, project_id, user_id)
    return [{"id": v.id, "label": v.label, "parent_version_id": v.parent_version_id, "created_at": v.created_at}
            for v in session.scalars(select(SceneVersion).where(SceneVersion.project_id == project_id))]


@router.get("/api/projects/{project_id}/versions/{version_id}/manifest")
def manifest(project_id: str, version_id: str, session: Session = Depends(get_db), user_id: str | None = Depends(get_current_user)) -> object:
    _project(session, project_id, user_id)
    version = session.scalar(select(SceneVersion).where(SceneVersion.id == version_id, SceneVersion.project_id == project_id))
    if not version or not version.manifest_relpath:
        raise HTTPException(status_code=404, detail="manifest not found")
    storage = get_storage(get_settings())
    try:
        import json
        with storage.open_read(version.manifest_relpath) as stream:
            return json.load(stream)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail="manifest not found") from exc


def _storage_project(session: Session, project_id: str, actor: tuple[bool, str | None], *, for_write: bool = False) -> Project:
    is_worker, user_id = actor
    if not is_worker:
        _project(session, project_id, user_id)
    project = lock_project(session, project_id) if for_write else session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")
    if for_write and project.processing_status != "running":
        raise HTTPException(409, "Project processing is paused or stopped")
    return project


@router.post("/api/storage/{project_id}/{version_id}/upload")
def storage_upload(
    project_id: str,
    version_id: str,
    file: UploadFile,
    session: Session = Depends(get_db),
    actor: tuple[bool, str | None] = Depends(get_storage_actor),
) -> dict[str, object]:
    _storage_project(session, project_id, actor, for_write=True)
    filename = file.filename or "upload"
    try:
        safe_version = validate_key(version_id)
        if "/" in safe_version:
            raise ValueError("version_id must be a single path segment")
        key = validate_key(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    storage = get_storage(get_settings())
    full_key = f"{project_id}/{safe_version}/{key}"
    written = storage.put_stream(full_key, file.file)
    return {"key": full_key, "bytes": written}


@router.get("/api/storage/{project_id}/{version_id}/{path:path}", response_model=None)
def storage_download(
    project_id: str,
    version_id: str,
    path: str,
    session: Session = Depends(get_db),
    actor: tuple[bool, str | None] = Depends(get_storage_actor),
) -> StreamingResponse | FileResponse:
    _storage_project(session, project_id, actor)
    try:
        key = validate_key(f"{project_id}/{version_id}/{path}")
        storage = get_storage(get_settings())
        stream = storage.open_read(key)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400 if isinstance(exc, ValueError) else 404, detail="invalid or missing storage object") from exc
    media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return StreamingResponse(_storage_chunks(stream), media_type=media_type)


def _storage_chunks(stream: BinaryIO) -> Iterator[bytes]:
    try:
        while chunk := stream.read(1024 * 1024):
            yield chunk
    finally:
        stream.close()
