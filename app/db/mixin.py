from uuid import uuid4, UUID
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column, validates

class UUIDMixin:
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)

class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(insert_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(insert_default=func.now())

class LowerCaseEmailMixin:
    @validates("owner_id")
    def convert_lower(self, key, value):
        return value.lower()