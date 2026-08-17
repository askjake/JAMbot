"""pw: add namespace to chats

Revision ID: 9197c9b549a2
Revises: b31e7fa889c1
Create Date: 2025-11-04 23:54:48.761727

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "9197c9b549a2"
down_revision: Union[str, None] = "b31e7fa889c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "chat",
        sa.Column(
            "namespace",
            sa.String(),
            server_default="generic",
            nullable=False,
            comment="Namespace for the chat (i.e. belongs to a certain app)",
        ),
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("chat", "namespace")
    # ### end Alembic commands ###
