"""Add edit_sessions preview generation and mask revision columns."""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_edit_preview_generation"
down_revision = "0001_foundation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "edit_sessions" in tables:
        edit_cols = {col["name"] for col in inspector.get_columns("edit_sessions")}
        with op.batch_alter_table("edit_sessions") as batch_op:
            if "preview_seeds_json" not in edit_cols:
                batch_op.add_column(sa.Column("preview_seeds_json", sa.JSON(), nullable=True))
            if "mask_revision" not in edit_cols:
                batch_op.add_column(sa.Column("mask_revision", sa.Integer(), nullable=False, server_default="0"))
            if "preview_generation" not in edit_cols:
                batch_op.add_column(sa.Column("preview_generation", sa.Integer(), nullable=False, server_default="0"))
            if "preview_meta_json" not in edit_cols:
                batch_op.add_column(sa.Column("preview_meta_json", sa.Text(), nullable=True))

    if "tasks" in tables:
        task_cols = {col["name"] for col in inspector.get_columns("tasks")}
        if "worker_id" not in task_cols:
            with op.batch_alter_table("tasks") as batch_op:
                batch_op.add_column(sa.Column("worker_id", sa.String(length=255), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "edit_sessions" in inspector.get_table_names():
        edit_cols = {col["name"] for col in inspector.get_columns("edit_sessions")}
        with op.batch_alter_table("edit_sessions") as batch_op:
            for col_name in ("preview_meta_json", "preview_generation", "mask_revision", "preview_seeds_json"):
                if col_name in edit_cols:
                    batch_op.drop_column(col_name)
