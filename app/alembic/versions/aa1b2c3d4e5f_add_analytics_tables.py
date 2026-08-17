"""Add analytics tables chat_summary and backend_insight

Revision ID: aa1b2c3d4e5f
Revises: 1475dcf15094
Create Date: 2026-01-21 17:29:12

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "aa1b2c3d4e5f"
down_revision: Union[str, None] = "1475dcf15094"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema by creating analytics tables."""

    # chat_summary: per-chat conversation summary and analytics signal.
    op.create_table(
        "chat_summary",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("chat_id", sa.String(), nullable=False),
        sa.Column("owner_email", sa.String(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=False, server_default="unknown"),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("metrics", postgresql.JSONB, nullable=False),
        sa.Column("backend_enhancement_ideas", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "idx_chat_summary_chat_owner",
        "chat_summary",
        ["chat_id", "owner_email"],
        unique=False,
    )

    # backend_insight: aggregated suggestions for improving the backend.
    op.create_table(
        "backend_insight",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("source", sa.String(), nullable=False, server_default="chat_summary"),
        sa.Column("payload", postgresql.JSONB, nullable=False),
        sa.Column("applied", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    """Downgrade schema by dropping analytics tables."""
    op.drop_table("backend_insight")
    op.drop_index("idx_chat_summary_chat_owner", table_name="chat_summary")
    op.drop_table("chat_summary")
