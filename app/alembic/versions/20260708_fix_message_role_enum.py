"""fix message_role_enum values to lowercase

Revision ID: 20260708_fix_message_role_enum
Revises: 56ae7a84b80b
Create Date: 2026-07-08
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '20260708_fix_role_enum'
down_revision: str = '56ae7a84b80b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Fix the message_role_enum type to use lowercase values matching
    the Python enum's .value attributes: 'user', 'assistant', 'tool'.
    
    This handles the case where the enum was created with uppercase names
    (AI, USER, TOOL) but the SQLAlchemy model now sends lowercase values.
    """
    # Rename existing enum values to lowercase equivalents
    # PostgreSQL supports ALTER TYPE ... RENAME VALUE in PG 10+
    # But since we can't conditionally check, we'll recreate the type safely.
    
    # Strategy: Create new enum, migrate column, drop old enum
    op.execute("DO $$ BEGIN "
               "  IF EXISTS (SELECT 1 FROM pg_enum WHERE enumlabel = 'USER' "
               "             AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'message_role_enum')) THEN "
               "    ALTER TABLE message_metadata ALTER COLUMN role TYPE VARCHAR; "
               "    UPDATE message_metadata SET role = LOWER(role) WHERE role IN ('USER', 'TOOL'); "
               "    UPDATE message_metadata SET role = 'assistant' WHERE role = 'AI'; "
               "    DROP TYPE message_role_enum; "
               "    CREATE TYPE message_role_enum AS ENUM ('user', 'assistant', 'tool'); "
               "    ALTER TABLE message_metadata ALTER COLUMN role TYPE message_role_enum USING role::message_role_enum; "
               "  END IF; "
               "END $$;")


def downgrade() -> None:
    """Revert to uppercase enum values."""
    op.execute("DO $$ BEGIN "
               "  IF EXISTS (SELECT 1 FROM pg_enum WHERE enumlabel = 'user' "
               "             AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'message_role_enum')) THEN "
               "    ALTER TABLE message_metadata ALTER COLUMN role TYPE VARCHAR; "
               "    UPDATE message_metadata SET role = UPPER(role) WHERE role IN ('user', 'tool'); "
               "    UPDATE message_metadata SET role = 'AI' WHERE role = 'assistant'; "
               "    DROP TYPE message_role_enum; "
               "    CREATE TYPE message_role_enum AS ENUM ('AI', 'USER', 'TOOL'); "
               "    ALTER TABLE message_metadata ALTER COLUMN role TYPE message_role_enum USING role::message_role_enum; "
               "  END IF; "
               "END $$;")
