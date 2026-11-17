from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import EditSession, Project, SceneVersion, Task, TaskAttempt
from .progress import compute_progress
from .task_types import canonical_task_type, expand_capabilities

MAX_TASK_RETRIES = 3


class ConflictError(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_payload(session: Session, task: Task) -> dict[str, Any] | None:
    try:
        parsed = json.loads(task.payload_json or "{}")
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    task.status = "failed"
    task.message = "Malformed task payload_json"
    _mark_edit_failed(session, task)
    session.commit()
    return None


def _supports(task: Task, payload: dict[str, Any], capabilities: set[str]) -> bool:
    req_raw = payload.get("required_capabilities")
    if req_raw is None:
        required: set[str] = set()
    elif isinstance(req_raw, list):
        required = {str(item) for item in req_raw}
    else:
        return False
    if not required.issubset(capabilities):
        return False
    return not capabilities or not task.task_type or task.task_type in capabilities


def _mark_edit_failed(session: Session, task: Task) -> None:
    if not task.edit_session_id:
        return
    edit = session.get(EditSession, task.edit_session_id)
    if edit is not None:
        edit.status = "failed"


def _dependency_ready(session: Session, task: Task, payload: dict[str, Any]) -> bool:
    dep_id = payload.get("depends_on_task_id")
    if not isinstance(dep_id, str):
        return True
    dep_task = session.get(Task, dep_id)
    if dep_task is None or dep_task.status in {"failed", "cancelled"}:
        reason = dep_task.status if dep_task is not None else "missing"
        task.status = "failed"
        task.message = f"Prerequisite task '{dep_id}' is {reason}"
        _mark_edit_failed(session, task)
        session.commit()
        return False
    return dep_task.status == "completed"


_INTERACTIVE_TASKS = frozenset({"segment_init", "segment_refine"})


def _is_interactive(task: Task, payload: dict[str, Any]) -> bool:
    return task.task_type in _INTERACTIVE_TASKS or bool(payload.get("interactive", False))


def _preflight_claim(
    session: Session, task_id: str, capabilities: list[str], interactive_only: bool
) -> tuple[Task, dict[str, Any]] | None:
    task = session.get(Task, task_id)
    if task is None or task.status != "queued":
        return None
    payload = _parse_payload(session, task)
    if payload is None:
        return None
    if (interactive_only and not _is_interactive(task, payload)) or not _supports(task, payload, set(capabilities)):
        return None
    if not _dependency_ready(session, task, payload):
        return None
    return task, payload


def _try_claim_sqlite(
    session: Session, task_id: str, worker_id: str, capabilities: list[str], interactive_only: bool
) -> dict[str, Any] | None:
    ready = _preflight_claim(session, task_id, capabilities, interactive_only)
    if ready is None:
        return None
    task, payload = ready
    attempt_id = str(uuid.uuid4())
    changed = session.execute(
        update(Task).where(Task.id == task.id, Task.status == "queued").values(
            status="claimed", active_attempt_id=attempt_id, worker_id=worker_id, updated_at=_now()
        )
    ).rowcount
    if changed != 1:
        session.rollback()
        return None
    session.add(TaskAttempt(id=attempt_id, task_id=task.id, worker_id=worker_id, status="claimed"))
    session.commit()
    return {"task_id": task.id, "attempt_id": attempt_id, "project_id": task.project_id,
            "task_type": task.task_type, "payload": payload}


def _try_claim_postgres(
    session: Session, task_id: str, worker_id: str, capabilities: list[str], interactive_only: bool
) -> dict[str, Any] | None:
    ready = _preflight_claim(session, task_id, capabilities, interactive_only)
    if ready is None:
        return None
    _, payload = ready
    stmt = (
        select(Task)
        .where(Task.id == task_id, Task.status == "queued")
        .with_for_update(skip_locked=True)
        .execution_options(populate_existing=True)
    )
    task = session.scalars(stmt).first()
    if task is None:
        return None
    attempt_id = str(uuid.uuid4())
    task.status = "claimed"
    task.active_attempt_id = attempt_id
    task.worker_id = worker_id
    task.updated_at = _now()
    session.add(TaskAttempt(id=attempt_id, task_id=task.id, worker_id=worker_id, status="claimed"))
    session.commit()
    return {"task_id": task.id, "attempt_id": attempt_id, "project_id": task.project_id,
            "task_type": task.task_type, "payload": payload}


def _as_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _reclaim_stale_tasks(session: Session, lease_timeout_sec: int | None = None) -> int:
    timeout = lease_timeout_sec if lease_timeout_sec is not None else get_settings().task_lease_timeout_sec
    if timeout <= 0:
        return 0
    now = _now()
    cutoff = now - timedelta(seconds=timeout)
    is_pg = bool(session.bind and session.bind.dialect.name != "sqlite")
    stmt = select(Task).where(Task.status.in_(("claimed", "running")))
    if is_pg:
        stmt = stmt.with_for_update(skip_locked=True)
    active_tasks = list(session.scalars(stmt).all())
    reclaimed = 0
    for task in active_tasks:
        attempt_id = task.active_attempt_id
        attempt = session.get(TaskAttempt, attempt_id) if attempt_id else None
        last_seen = _as_utc(attempt.heartbeat_at if attempt is not None else None) or _as_utc(task.updated_at)
        if last_seen is None or last_seen >= cutoff:
            continue
        requeue = task.retry_count < MAX_TASK_RETRIES
        next_status = "queued" if requeue else "failed"
        next_retry = task.retry_count + 1 if requeue else task.retry_count
        next_msg = "Re-queued after worker lease timeout" if requeue else f"LEASE_TIMEOUT: Worker lease exceeded {timeout}s"
        changed = session.execute(
            update(Task)
            .where(
                Task.id == task.id,
                Task.active_attempt_id == attempt_id,
                Task.status.in_(("claimed", "running")),
            )
            .values(
                status=next_status,
                retry_count=next_retry,
                worker_id=None,
                active_attempt_id=None,
                message=next_msg,
                updated_at=now,
            )
        ).rowcount
        if changed != 1:
            continue
        if attempt is not None:
            attempt.status = "failed"
            attempt.error_code = "LEASE_TIMEOUT"
            attempt.error_message = f"Worker lease exceeded {timeout}s without heartbeat"
            attempt.finished_at = now
        if not requeue:
            _mark_edit_failed(session, task)
        reclaimed += 1
    if reclaimed:
        session.commit()
    return reclaimed


def claim_next_task(session: Session, worker_id: str, capabilities: list[str], interactive_only: bool) -> dict[str, Any] | None:
    capabilities = expand_capabilities(capabilities)
    _reclaim_stale_tasks(session)
    stmt = select(Task.id).where(Task.status == "queued")
    if capabilities:
        stmt = stmt.where(Task.task_type.in_(capabilities))
    if interactive_only:
        stmt = stmt.where(Task.task_type.in_(_INTERACTIVE_TASKS) | Task.payload_json.contains("interactive"))
    candidate_ids = list(session.scalars(stmt.order_by(Task.created_at, Task.id)).all())
    is_pg = bool(session.bind and session.bind.dialect.name == "postgresql")
    try:
        for task_id in candidate_ids:
            claimed = (_try_claim_postgres if is_pg else _try_claim_sqlite)(
                session, task_id, worker_id, capabilities, interactive_only
            )
            if claimed is not None:
                return claimed
    except OperationalError:
        session.rollback()
        raise
    session.commit()
    return None


def _locked_task(session: Session, task_id: str) -> Task:
    is_pg = bool(session.bind and session.bind.dialect.name != "sqlite")
    stmt = select(Task).where(Task.id == task_id)
    if is_pg:
        stmt = stmt.with_for_update()
    task = session.scalars(stmt).first()
    if not task:
        raise LookupError(f"task '{task_id}' not found")
    return task


def _check_active_attempt(session: Session, task: Task, attempt_id: str, worker_id: str | None = None) -> TaskAttempt:
    if task.status not in {"claimed", "running"}:
        raise ConflictError("task is no longer active")
    if task.active_attempt_id != attempt_id:
        raise ConflictError("stale task attempt")
    attempt = session.get(TaskAttempt, attempt_id)
    if not attempt or attempt.task_id != task.id:
        raise ConflictError("invalid task attempt")
    if worker_id is not None and (task.worker_id != worker_id or attempt.worker_id != worker_id):
        raise ConflictError("worker does not own active task")
    return attempt


def heartbeat_task(session: Session, task_id: str, attempt_id: str, worker_id: str) -> None:
    task = _locked_task(session, task_id)
    attempt = _check_active_attempt(session, task, attempt_id, worker_id=worker_id)
    attempt.heartbeat_at = _now()
    if task.status == "claimed":
        task.status = "running"
    session.commit()


def update_progress(
    session: Session,
    task_id: str,
    attempt_id: str,
    stage: str,
    progress_pct: float,
    message: str | None = None,
    telemetry: dict[str, Any] | None = None,
    worker_id: str | None = None,
) -> float:
    task = _locked_task(session, task_id)
    _check_active_attempt(session, task, attempt_id, worker_id=worker_id)
    progress = compute_progress(task.task_type, stage, progress_pct)
    task.current_stage, task.progress_pct, task.message, task.status = stage, progress, message, "running"
    if telemetry:
        task.payload_json = json.dumps({**json.loads(task.payload_json or "{}"), "telemetry": telemetry})
    session.commit()
    return progress


report_progress = update_progress


def _register_completed_version(
    session: Session,
    task: Task,
    result: dict[str, Any],
    result_manifest_key: str | None,
) -> str | None:
    if canonical_task_type(task.task_type) not in {"reconstruct_week1", "rebuild_week2"}:
        return None
    version_id = str(result.get("version_id") or "")
    if not version_id and result_manifest_key:
        parts = result_manifest_key.split("/")
        if len(parts) >= 3:
            version_id = parts[1]
    if not version_id:
        return None
    existing = session.get(SceneVersion, version_id)
    if existing is not None and existing.project_id != task.project_id:
        version_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{task.project_id}:{version_id}"))
        existing = session.get(SceneVersion, version_id)
    asset_prefix = Path(result_manifest_key).parent.as_posix() if result_manifest_key else f"{task.project_id}/{version_id}"
    if existing is None:
        session.add(
            SceneVersion(
                id=version_id,
                project_id=task.project_id,
                parent_version_id=result.get("parent_version_id"),
                label=str(result.get("label") or version_id),
                manifest_relpath=result_manifest_key,
                spz_relpath=f"{asset_prefix}/scene.spz",
                ply_relpath=f"{asset_prefix}/scene.ply",
            )
        )
    else:
        existing.manifest_relpath = result_manifest_key or existing.manifest_relpath
    project = session.get(Project, task.project_id)
    if project is not None:
        project.active_version_id = version_id
        project.status = "completed"
    return version_id


def _canonical_json(raw: str | None, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def masks_digest(masks_summary_json: str | None, points_json: str | None) -> str:
    payload = {
        "masks_summary": _canonical_json(masks_summary_json, {}),
        "points": _canonical_json(points_json, []),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def invalidate_edit_previews(edit: EditSession) -> None:
    edit.mask_revision = int(edit.mask_revision or 0) + 1
    edit.preview_generation = int(edit.preview_generation or 0) + 1
    edit.preview_seeds_json = {}
    edit.preview_meta_json = None


def _apply_edit_completion(
    session: Session,
    task: Task,
    result: dict[str, Any],
    version_id: str | None,
) -> None:
    if not task.edit_session_id:
        return
    edit = session.get(EditSession, task.edit_session_id)
    if edit is None:
        return

    task_type = canonical_task_type(task.task_type)
    if task_type in {"segment_init", "segment_refine"}:
        summary: dict[str, Any] = json.loads(edit.masks_summary_json or "{}")
        mask_summary = result.get("mask_summary")
        if isinstance(mask_summary, dict):
            if "frame_idx" in mask_summary:
                summary[str(mask_summary["frame_idx"])] = mask_summary
            else:
                for key, entry in mask_summary.items():
                    summary[str(key)] = entry
        elif isinstance(result.get("mask_key"), str) and edit.ref_frame_idx is not None:
            summary[str(edit.ref_frame_idx)] = {
                "status": "segmented",
                "mask_key": result["mask_key"],
                "skip_frame": False,
            }
        edit.masks_summary_json = json.dumps(summary)
        invalidate_edit_previews(edit)
        edit.status = "segmented"
    elif task_type == "track_propagate":
        masks_summary = result.get("masks_summary")
        if isinstance(masks_summary, dict):
            edit.masks_summary_json = json.dumps({str(k): v for k, v in masks_summary.items()})
        invalidate_edit_previews(edit)
        edit.status = "tracked"
    elif task_type == "diffusion_preview":
        previews = result.get("preview_seeds")
        extracted: dict[str, str] = {}
        if isinstance(previews, dict):
            extracted = {str(k): str(v) for k, v in previews.items()}
        elif isinstance(result.get("seeds"), list):
            extracted = {
                str(item["seed_id"]): str(item["preview_key"])
                for item in result["seeds"]
                if isinstance(item, dict) and "seed_id" in item and "preview_key" in item
            }
        task_payload = _canonical_json(task.payload_json, {})
        if not isinstance(task_payload, dict):
            task_payload = {}
        matches_generation = (
            task_payload.get("preview_generation") == int(edit.preview_generation or 0)
            and task_payload.get("mask_revision") == int(edit.mask_revision or 0)
            and task_payload.get("masks_digest") == masks_digest(edit.masks_summary_json, edit.points_json)
        )
        if matches_generation and extracted:
            edit.preview_seeds_json = extracted
            meta = _canonical_json(edit.preview_meta_json, {})
            if not isinstance(meta, dict):
                meta = {}
            meta["completed"] = True
            meta["preview_seeds"] = dict(extracted)
            edit.preview_meta_json = json.dumps(meta)
            edit.status = "preview_ready"
    elif task_type == "diffusion_apply":
        edit.status = "Edited"
    elif task_type == "rebuild_week2":
        if version_id:
            edit.target_version_id = version_id
        edit.status = "completed"


def complete_task(
    session: Session,
    task_id: str,
    attempt_id: str,
    result_payload: dict[str, Any],
    result_manifest_key: str | None = None,
    worker_id: str | None = None,
) -> None:
    task = _locked_task(session, task_id)
    attempt = _check_active_attempt(session, task, attempt_id, worker_id=worker_id)
    result = dict(result_payload)
    if result_manifest_key:
        result["manifest_key"] = result_manifest_key
    changed = session.execute(
        update(Task)
        .where(
            Task.id == task.id,
            Task.active_attempt_id == attempt_id,
            Task.status.in_(("claimed", "running")),
        )
        .values(
            status="completed",
            progress_pct=100.0,
            result_json=json.dumps(result),
            active_attempt_id=None,
            updated_at=_now(),
        )
    ).rowcount
    if changed != 1:
        session.rollback()
        raise ConflictError("stale task attempt")
    attempt.status, attempt.finished_at = "completed", _now()
    version_id = _register_completed_version(session, task, result, result_manifest_key)
    _apply_edit_completion(session, task, result, version_id)
    session.commit()


def fail_task(
    session: Session,
    task_id: str,
    attempt_id: str,
    error_code: str,
    error_message: str,
    retryable: bool,
    worker_id: str | None = None,
) -> None:
    task = _locked_task(session, task_id)
    attempt = _check_active_attempt(session, task, attempt_id, worker_id=worker_id)
    should_retry = retryable and task.retry_count < MAX_TASK_RETRIES
    detail = f"{error_code}: {error_message}" if error_code else error_message
    next_status = "queued" if should_retry else "failed"
    changed = session.execute(
        update(Task)
        .where(
            Task.id == task.id,
            Task.active_attempt_id == attempt_id,
            Task.status.in_(("claimed", "running")),
        )
        .values(
            status=next_status,
            message=detail[:1000],
            retry_count=task.retry_count + 1 if should_retry else task.retry_count,
            worker_id=None if should_retry else task.worker_id,
            active_attempt_id=None,
            updated_at=_now(),
        )
    ).rowcount
    if changed != 1:
        session.rollback()
        raise ConflictError("stale task attempt")
    if not should_retry:
        _mark_edit_failed(session, task)
    attempt.status, attempt.finished_at = "failed", _now()
    attempt.error_code, attempt.error_message = error_code, error_message
    session.commit()
