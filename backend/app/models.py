from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, event, func, inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def new_id() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    projects: Mapped[list[Project]] = relationship(back_populates="owner")


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    owner_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), index=True, nullable=True)
    name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="queued")
    active_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    video_relpath: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    owner: Mapped[User] = relationship(back_populates="projects")
    versions: Mapped[list[SceneVersion]] = relationship(back_populates="project", foreign_keys="SceneVersion.project_id")


class SceneVersion(Base):
    __tablename__ = "scene_versions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    parent_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    label: Mapped[str] = mapped_column(String(255))
    manifest_relpath: Mapped[str | None] = mapped_column(Text, nullable=True)
    spz_relpath: Mapped[str | None] = mapped_column(Text, nullable=True)
    ply_relpath: Mapped[str | None] = mapped_column(Text, nullable=True)
    collision_glb_relpath: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    project: Mapped[Project] = relationship(back_populates="versions", foreign_keys=[project_id])


class EditSession(Base):
    __tablename__ = "edit_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    base_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    target_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    ref_frame_idx: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    points_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    masks_summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    preview_seeds_json: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    mask_revision: Mapped[int] = mapped_column(Integer, default=0)
    preview_generation: Mapped[int] = mapped_column(Integer, default=0)
    preview_meta_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    chosen_seed_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    @property
    def output_version_id(self) -> str | None:
        return self.target_version_id

    @output_version_id.setter
    def output_version_id(self, value: str | None) -> None:
        self.target_version_id = value


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    edit_session_id: Mapped[str | None] = mapped_column(ForeignKey("edit_sessions.id"), nullable=True)
    task_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    current_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    progress_pct: Mapped[float] = mapped_column(default=0.0)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    active_attempt_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    @property
    def edit_id(self) -> str | None:
        return self.edit_session_id
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class TaskAttempt(Base):
    __tablename__ = "task_attempts"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    worker_id: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default="claimed")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


@event.listens_for(Base.metadata, "after_create")
def upgrade_existing_schema(target: Any, connection: Connection, **_: Any) -> None:
    """Backfill additive columns on pre-existing SQLite/SQL databases during create_all."""
    del target
    inspector = inspect(connection)
    tables = set(inspector.get_table_names())
    if "edit_sessions" in tables:
        cols = {col["name"] for col in inspector.get_columns("edit_sessions")}
        if "preview_seeds_json" not in cols:
            connection.execute(text("ALTER TABLE edit_sessions ADD COLUMN preview_seeds_json JSON"))
        if "mask_revision" not in cols:
            connection.execute(text("ALTER TABLE edit_sessions ADD COLUMN mask_revision INTEGER NOT NULL DEFAULT 0"))
        if "preview_generation" not in cols:
            connection.execute(text("ALTER TABLE edit_sessions ADD COLUMN preview_generation INTEGER NOT NULL DEFAULT 0"))
        if "preview_meta_json" not in cols:
            connection.execute(text("ALTER TABLE edit_sessions ADD COLUMN preview_meta_json TEXT"))
    if "tasks" in tables:
        task_cols = {col["name"] for col in inspector.get_columns("tasks")}
        if "worker_id" not in task_cols:
            connection.execute(text("ALTER TABLE tasks ADD COLUMN worker_id VARCHAR(255)"))
