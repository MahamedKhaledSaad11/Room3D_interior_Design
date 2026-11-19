from __future__ import annotations

import hashlib
import io
import json
import tempfile
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..security import require_worker_token
from ..storage import get_storage
from ..storage.base import StorageDriver, validate_key
from ..storage.bundles import bundle_objects, bundle_path, import_bundle, object_payload
from .scenes import _storage_project

router = APIRouter(prefix="/api/storage", dependencies=[Depends(require_worker_token)])


class CompleteUpload(BaseModel):
    path: str
    part_count: int = Field(ge=1)
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    extract_to: str | None = None


def _scope(project_id: str, scope: str, upload_id: UUID, session: Session) -> tuple[StorageDriver, str]:
    _storage_project(session, project_id, (True, None), for_write=True)
    try:
        clean = validate_key(scope)
        if "/" in clean:
            raise ValueError("scope must be a single path segment")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return get_storage(get_settings()), f"{project_id}/{clean}/.uploads/{upload_id}"


@router.post("/{project_id}/{scope}/uploads/{upload_id}/parts/{index}")
def upload_part(
    project_id: str, scope: str, upload_id: UUID, index: int, file: UploadFile,
    session: Session = Depends(get_db),
) -> dict[str, int]:
    storage, prefix = _scope(project_id, scope, upload_id, session)
    if index < 0:
        raise HTTPException(422, "part index must be non-negative")
    return {"bytes": storage.put_stream(f"{prefix}/parts/{index:08d}", file.file)}


def _assemble(storage: StorageDriver, prefix: str, target: str, body: CompleteUpload) -> int:
    digest = hashlib.sha256()
    size = 0
    with tempfile.TemporaryFile() as assembled:
        for index in range(body.part_count):
            try:
                source = storage.open_read(f"{prefix}/parts/{index:08d}")
            except FileNotFoundError as exc:
                raise HTTPException(422, f"Missing upload part {index}") from exc
            with source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    assembled.write(chunk)
        if size != body.size or digest.hexdigest() != body.sha256:
            raise HTTPException(422, "Artifact size or SHA-256 mismatch; retry the upload")
        assembled.seek(0)
        if body.extract_to is not None:
            try:
                import_bundle(storage, target, assembled)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            return size
        return storage.put_stream(target, assembled)


@router.post("/{project_id}/{scope}/uploads/{upload_id}/complete")
def complete_upload(
    project_id: str, scope: str, upload_id: UUID, body: CompleteUpload,
    session: Session = Depends(get_db),
) -> dict[str, object]:
    storage, prefix = _scope(project_id, scope, upload_id, session)
    try:
        path = validate_key(body.path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    receipt_key = f"{prefix}/receipt.json"
    target = f"{project_id}/{scope}/{path}"
    if body.extract_to is not None:
        try:
            destination = bundle_path(body.extract_to) if body.extract_to else ""
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        target = f"{project_id}/{scope}" + (f"/{destination}" if destination else "")
    receipt = {"key": target, "bytes": body.size, "sha256": body.sha256}
    if storage.exists(receipt_key):
        with storage.open_read(receipt_key) as stream:
            if json.load(stream) != receipt:
                raise HTTPException(409, "Upload ID already committed to another artifact")
        return _result(storage, target, receipt, body.extract_to is not None)
    _assemble(storage, prefix, target, body)
    storage.put_stream(receipt_key, io.BytesIO(json.dumps(receipt).encode()))
    storage.delete_prefix(f"{prefix}/parts")
    return _result(storage, target, receipt, body.extract_to is not None)


def _result(
    storage: StorageDriver, target: str, receipt: dict[str, str | int], is_bundle: bool,
) -> dict[str, object]:
    if is_bundle:
        return {**receipt, "objects": object_payload(bundle_objects(storage, target, []))}
    return dict(receipt)
