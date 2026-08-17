"""Fix usage_tracking timestamp default

Revision ID: 20260216_timestamp_fix
Revises: 65a3145b60d5
Create Date: 2026-02-16 13:58:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20260216_timestamp_fix'
down_revision: Union[str, None] = '65a3145b60d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add server default for timestamp column."""
    op.execute("ALTER TABLE usage_tracking ALTER COLUMN timestamp SET DEFAULT now();")


def downgrade() -> None:
    """Remove server default for timestamp column."""
    op.execute("ALTER TABLE usage_tracking ALTER COLUMN timestamp DROP DEFAULT;")
