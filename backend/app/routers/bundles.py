from __future__ import annotations

import tempfile
from collections.abc import Iterator
from typing import Annotated, BinaryIO, cast

from fastapi import APIRouter, Depends, Header, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from ..config import get_settings
from ..db import get_db
from ..security import get_storage_actor, require_worker_token
from ..storage import get_storage
from ..storage.bundles import (
    bundle_objects,
    bundle_path,
    import_bundle,
    object_payload,
    object_revision,
    write_bundle,
)
from .scenes import _storage_project

router = APIRouter(prefix="/api/storage", tags=["bundles"])
Paths = Annotated[list[str] | None, Query()]
SessionDep = Annotated[Session, Depends(get_db)]
ActorDep = Annotated[tuple[bool, str | None], Depends(get_storage_actor)]


def _prefix(project_id: str, scope: str, edit_id: str | None) -> str:
    clean = bundle_path(scope)
    if "/" in clean:
        raise ValueError("Storage scope must be a single path segment")
    if edit_id is None:
        return f"{project_id}/{clean}"
    edit = bundle_path(edit_id)
    if "/" in edit or scope != "edits":
        raise ValueError("Invalid edit scope")
    return f"{project_id}/edits/{edit}"


@router.get("/{project_id}/{scope}/bundle-index")
@router.get("/{project_id}/{scope}/{edit_id}/bundle-index")
def index(
    project_id: str, scope: str, session: SessionDep, actor: ActorDep,
    paths: Paths = None, edit_id: str | None = None,
) -> dict[str, object]:
    _storage_project(session, project_id, actor)
    try:
        storage = get_storage(get_settings())
        return {"objects": object_payload(bundle_objects(storage, _prefix(project_id, scope, edit_id), paths or []))}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _chunks(stream: BinaryIO) -> Iterator[bytes]:
    with stream:
        while chunk := stream.read(1024 * 1024):
            yield chunk


@router.get("/{project_id}/{scope}/bundle.zip")
@router.get("/{project_id}/{scope}/{edit_id}/bundle.zip")
def download_bundle(
    project_id: str, scope: str, session: SessionDep, actor: ActorDep,
    paths: Paths = None, edit_id: str | None = None,
    if_match: str | None = Header(default=None),
) -> StreamingResponse:
    _storage_project(session, project_id, actor)
    output = cast(BinaryIO, tempfile.TemporaryFile())  # noqa: SIM115 -- the streaming response owns this file.
    try:
        storage = get_storage(get_settings())
        objects = bundle_objects(storage, _prefix(project_id, scope, edit_id), paths or [])
        revision = object_revision(objects)
        if if_match is not None and if_match.strip('"') != revision:
            raise HTTPException(412, "Assets changed; refresh the bundle index")
        write_bundle(storage, objects, output)
        current = bundle_objects(storage, _prefix(project_id, scope, edit_id), paths or [])
        if object_revision(current) != revision:
            raise HTTPException(412, "Assets changed while building the bundle; retry")
        size = output.seek(0, 2)
        output.seek(0)
        return StreamingResponse(_chunks(output), media_type="application/zip", background=BackgroundTask(output.close), headers={
            "Content-Length": str(size), "Content-Disposition": 'attachment; filename="bundle.zip"',
            "Cache-Control": "no-store",
            "ETag": f'"{revision}"',
        })
    except ValueError as exc:
        output.close()
        raise HTTPException(400, str(exc)) from exc
    except BaseException:
        output.close()
        raise


@router.post("/{project_id}/{scope}/upload-bundle", dependencies=[Depends(require_worker_token)])
@router.post("/{project_id}/{scope}/{edit_id}/upload-bundle", dependencies=[Depends(require_worker_token)])
def upload_bundle(
    project_id: str, scope: str, file: UploadFile, session: SessionDep, edit_id: str | None = None,
) -> dict[str, object]:
    _storage_project(session, project_id, (True, None), for_write=True)
    try:
        storage = get_storage(get_settings())
        prefix = _prefix(project_id, scope, edit_id)
        count = import_bundle(storage, prefix, file.file)
        return {"files": count, "objects": object_payload(bundle_objects(storage, prefix, []))}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
