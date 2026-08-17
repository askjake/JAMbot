"""Add idle_checked column to chat table

Revision ID: 20260508_add_idle_checked
Revises: 20260429_mem_enhance
Create Date: 2026-05-08 08:53:48

This migration adds an idle_checked boolean flag to the chat table
to prevent re-scanning chats that have already been evaluated by
the idle chat checker.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20260508_add_idle_checked'
down_revision = "56ae7a84b80b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Add idle_checked column to chat table with index for query optimization.
    """
    # Add idle_checked column with default False
    op.add_column(
        'chat',
        sa.Column(
            'idle_checked',
            sa.Boolean(),
            server_default='false',
            nullable=False,
            comment='True once idle checker has evaluated this chat; prevents re-checks',
        )
    )
    
    # Create index for efficient querying of unchecked chats
    op.create_index(
        'idx_chat_idle_checked',
        'chat',
        ['idle_checked']
    )
    
    print("✅ Added idle_checked column and index to chat table")


def downgrade() -> None:
    """
    Remove idle_checked column and its index from chat table.
    """
    op.drop_index('idx_chat_idle_checked', table_name='chat')
    op.drop_column('chat', 'idle_checked')
    
    print("✅ Removed idle_checked column and index from chat table")
