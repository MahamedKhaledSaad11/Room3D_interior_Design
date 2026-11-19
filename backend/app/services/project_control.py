from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..models import EditSession, Project, Task, TaskAttempt

ControlAction = Literal["pause", "resume", "stop"]
UNFINISHED = ("queued", "claimed", "running", "paused", "pausing", "stopping")


def lock_project(session: Session, project_id: str) -> Project | None:
    # Serialize claims and controls on the project before locking individual tasks.
    if session.bind is not None and session.bind.dialect.name == "sqlite":
        session.execute(update(Project).where(Project.id == project_id)
                        .values(processing_status=Project.processing_status))
    return session.scalar(select(Project).where(Project.id == project_id).with_for_update()
                          .execution_options(populate_existing=True))


def control_project(session: Session, project: Project, action: ControlAction) -> None:
    if lock_project(session, project.id) is None:
        raise LookupError("Project no longer exists")
    tasks = list(session.scalars(select(Task).where(Task.project_id == project.id, Task.status.in_(UNFINISHED)).with_for_update()))
    if action == "resume" and any(task.status in {"pausing", "stopping"} for task in tasks):
        raise ValueError("Wait for the worker to finish stopping before resuming.")
    for task in tasks:
        if action == "resume":
            if task.status == "paused":
                task.status, task.worker_id, task.active_attempt_id = "queued", None, None
                task.progress_pct, task.current_stage = 0.0, None
                task.message = "Resumed; restarting the interrupted stage"
            continue
        if task.status in {"claimed", "running", "pausing", "stopping"}:
            task.status = "pausing" if action == "pause" and task.status != "stopping" else "stopping"
        else:
            task.status = "paused" if action == "pause" else "cancelled"
        task.message = "Pause requested" if action == "pause" else "Stop requested"
    project.processing_status = {"pause": "paused", "resume": "running", "stop": "stopped"}[action]
    if action == "stop":
        edits = session.scalars(select(EditSession).where(EditSession.project_id == project.id))
        for edit in edits:
            if edit.status not in {"completed", "failed", "cancelled"}:
                edit.status = "cancelled"
    session.commit()


def finish_interruption(session: Session, task: Task) -> None:
    target = "paused" if task.status == "pausing" else "cancelled"
    attempt = session.get(TaskAttempt, task.active_attempt_id) if task.active_attempt_id else None
    if attempt is not None:
        attempt.status, attempt.finished_at = target, datetime.now(UTC)
    task.status, task.active_attempt_id, task.worker_id = target, None, None
    task.message = "Paused; resume restarts this stage" if target == "paused" else "Stopped by user"
