"""Add model config field to message metadata

Revision ID: 1475dcf15094
Revises: e8e4dad5b79c
Create Date: 2025-07-11 17:44:54.293781

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '1475dcf15094'
down_revision: Union[str, None] = 'e8e4dad5b79c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('message_metadata', sa.Column('message_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True, comment='The additional model kwargs used to generate the response message.'))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('message_metadata', 'message_config')
