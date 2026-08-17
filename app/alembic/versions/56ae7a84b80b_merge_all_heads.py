"""merge_all_heads

Revision ID: 56ae7a84b80b
Revises: merge_heads_001, 20260216_timestamp_fix, 9197c9b549a2, aa1b2c3d4e5f
Create Date: 2026-02-27 13:04:03.868068

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '56ae7a84b80b'
down_revision: Union[str, None] = ('merge_heads_001', '20260216_timestamp_fix', '9197c9b549a2', 'aa1b2c3d4e5f')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
