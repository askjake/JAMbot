"""merge heads

Revision ID: merge_heads_001
Revises: 20260123_add_agent_mode_run, journal_001
Create Date: 2026-01-26

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'merge_heads_001'
down_revision = ('20260123_add_agent_mode_run', 'journal_001')
branch_labels = None
depends_on = None


def upgrade():
    # This is a merge migration, no changes needed
    pass


def downgrade():
    # This is a merge migration, no changes needed
    pass
