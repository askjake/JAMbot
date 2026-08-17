"""pw: add active_checkpoint field to chat

Revision ID: 863206563d57
Revises: d9834ecac7fc
Create Date: 2025-05-16 17:44:48.746898

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '863206563d57'
down_revision: Union[str, None] = 'd9834ecac7fc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('chat', sa.Column('active_checkpoint', sa.Uuid(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('chat', 'active_checkpoint')
