"""pw: update attachment table

Revision ID: 50e3ad568b2a
Revises: 863206563d57
Create Date: 2025-05-29 01:29:31.653343

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '50e3ad568b2a'
down_revision: Union[str, None] = '863206563d57'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


enum_values = ('queued', 'processing', 'ready', 'failed')
old_enum_type = postgresql.ENUM(*enum_values, name="attachmentstatusenum")
new_enum_type = postgresql.ENUM(*enum_values, name="progressstatusenum")

def upgrade() -> None:
    """Upgrade schema."""
    new_enum_type.create(op.get_bind(), checkfirst=False)
    op.alter_column('attachment', 'status',
               existing_type=old_enum_type,
               type_=new_enum_type,
               existing_nullable=False,
               postgresql_using='status::text::progressstatusenum'
    )
    old_enum_type.drop(op.get_bind(), checkfirst=False) # Set checkfirst=True if unsure

def downgrade() -> None:
    """Downgrade schema."""
    # 1. Re-create the old ENUM type
    old_enum_type.create(op.get_bind(), checkfirst=False)

    # 2. Alter the column back to the old ENUM type
    op.alter_column('attachment', 'status',
        existing_type=new_enum_type,
        type_=old_enum_type,
        existing_nullable=False,
        postgresql_using='status::text::attachmentstatusenum'
    )

    # 3. Drop the new ENUM type
    new_enum_type.drop(op.get_bind(), checkfirst=False)
 