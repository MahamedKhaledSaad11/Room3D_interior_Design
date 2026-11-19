"""Add persistent project processing controls."""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_project_processing"
down_revision = "0002_edit_preview_generation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("projects")}
    if "processing_status" not in columns:
        op.add_column("projects", sa.Column("processing_status", sa.String(32), nullable=False, server_default="running"))


def downgrade() -> None:
    with op.batch_alter_table("projects") as batch:
        batch.drop_column("processing_status")
