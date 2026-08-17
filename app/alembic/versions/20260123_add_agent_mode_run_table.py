"""add_agent_mode_run_table

Revision ID: 20260123_add_agent_mode_run
Revises: PUT_PREVIOUS_REVISION_HERE
Create Date: 2026-01-23 00:00:00.000000

NOTE:
- Replace `PUT_PREVIOUS_REVISION_HERE` with your actual previous Alembic revision id
  before running `alembic upgrade head`.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260123_add_agent_mode_run"
down_revision = "e8e4dad5b79c"
branch_labels = None
depends_on = None


def upgrade():
    """Upgrade schema - create agent_mode_run table."""
    
    # Create enum type if it doesn't exist (idempotent)
    # Check if the enum type already exists
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'agent_mode_run_status')"
    ))
    enum_exists = result.scalar()
    
    if not enum_exists:
        # Create the enum type
        agent_mode_run_status = postgresql.ENUM(
            'queued', 'running', 'succeeded', 'failed', 'cancelled', 'timeout',
            name='agent_mode_run_status',
            create_type=True
        )
        agent_mode_run_status.create(op.get_bind(), checkfirst=True)
    
    # Create the table
    op.create_table(
        "agent_mode_run",
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("chat_id", sa.UUID(), nullable=False),
        sa.Column("message_id", sa.UUID(), nullable=False),
        sa.Column("owner_id", sa.String(), nullable=False),
        sa.Column("status", postgresql.ENUM(
            'queued', 'running', 'succeeded', 'failed', 'cancelled', 'timeout',
            name='agent_mode_run_status',
            create_type=False  # Don't try to create it again
        ), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("input_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("output_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("run_id"),
    )
    
    # Create indexes
    op.create_index(
        op.f("ix_agent_mode_run_chat_id"),
        "agent_mode_run",
        ["chat_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_mode_run_message_id"),
        "agent_mode_run",
        ["message_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_mode_run_owner_id"),
        "agent_mode_run",
        ["owner_id"],
        unique=False,
    )


def downgrade():
    """Downgrade schema - drop agent_mode_run table."""
    op.drop_index(op.f("ix_agent_mode_run_owner_id"), table_name="agent_mode_run")
    op.drop_index(op.f("ix_agent_mode_run_message_id"), table_name="agent_mode_run")
    op.drop_index(op.f("ix_agent_mode_run_chat_id"), table_name="agent_mode_run")
    op.drop_table("agent_mode_run")
    
    # Drop the enum type
    agent_mode_run_status = postgresql.ENUM(
        'queued', 'running', 'succeeded', 'failed', 'cancelled', 'timeout',
        name='agent_mode_run_status'
    )
    agent_mode_run_status.drop(op.get_bind(), checkfirst=True)
