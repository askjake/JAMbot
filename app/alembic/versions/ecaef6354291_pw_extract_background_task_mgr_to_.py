"""pw: extract background task mgr to standalone module

Revision ID: ecaef6354291
Revises: 50e3ad568b2a
Create Date: 2025-06-03 02:38:08.499394

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'ecaef6354291'
down_revision: Union[str, None] = '50e3ad568b2a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    
    op.create_table('bg_tasks',
    sa.Column('task_id', sa.Uuid(), nullable=False),
    sa.Column('task_type', sa.String(), nullable=False),
    sa.Column('status', postgresql.ENUM('queued', 'processing', 'ready', 'failed', name='progressstatusenum',  create_type=False), nullable=False),
    sa.Column('progress', sa.Integer(), nullable=False),
    sa.Column('total', sa.Integer(), nullable=False),
    sa.Column('message', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('task_id')
    )

    # Handle attachment table changes
    # Add preprocessed column as nullable first, then update and make NOT NULL
    op.add_column('attachment', sa.Column('preprocessed', sa.Boolean(), nullable=True))
    op.execute("UPDATE attachment SET preprocessed = FALSE WHERE status != 'ready'")
    op.execute("UPDATE attachment SET preprocessed = TRUE WHERE status = 'ready'")
    op.alter_column('attachment', 'preprocessed', nullable=False)

    op.drop_column('attachment', 'progress')
    op.drop_column('attachment', 'progress_msg')
    op.drop_column('attachment', 'status')
    
    # Add updated_at columns with default values for existing rows
    op.add_column('chat', sa.Column('updated_at', sa.DateTime(), nullable=True))
    op.execute("UPDATE chat SET updated_at = COALESCE(created_at, NOW()) WHERE updated_at IS NULL")
    op.alter_column('chat', 'updated_at', nullable=False)
    
    op.add_column('vault_session', sa.Column('updated_at', sa.DateTime(), nullable=True))
    op.execute("UPDATE vault_session SET updated_at = COALESCE(created_at, NOW()) WHERE updated_at IS NULL")
    op.alter_column('vault_session', 'updated_at', nullable=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('vault_session', 'updated_at')
    op.drop_column('chat', 'updated_at')
    op.add_column('attachment', sa.Column('status', postgresql.ENUM('queued', 'processing', 'ready', 'failed', name='progressstatusenum'), autoincrement=False, nullable=False))
    op.add_column('attachment', sa.Column('progress_msg', sa.VARCHAR(), autoincrement=False, nullable=False))
    op.add_column('attachment', sa.Column('progress', sa.INTEGER(), autoincrement=False, nullable=False))
    op.drop_column('attachment', 'preprocessed')
    op.drop_table('bg_tasks')
    # ### end Alembic commands ###
