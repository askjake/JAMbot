"""pw: update vault table email column names

Revision ID: b31e7fa889c1
Revises: ec127df018c6
Create Date: 2025-09-09 16:53:06.179347

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b31e7fa889c1"
down_revision: Union[str, None] = "ec127df018c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column("vault_credential", "user_email", new_column_name="owner_id")
    op.alter_column("vault_session", "user_email", new_column_name="owner_id")
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column("vault_credential", "owner_id", new_column_name="user_email")
    op.alter_column("vault_session", "owner_id", new_column_name="user_email")
    # ### end Alembic commands ###
