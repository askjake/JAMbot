from datetime import date as date_type

from sqlalchemy import func, ARRAY, CheckConstraint, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from uuid import UUID, uuid4

from app.db import Base
from app.db.mixin import TimestampMixin


class Release(Base, TimestampMixin):
    __tablename__ = "releases"

    release_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    date: Mapped[date_type] = mapped_column(insert_default=func.current_date(), index=True)  # Use date_type
    title: Mapped[str] = mapped_column(String(100))
    changes: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)

    # --- Table Arguments (Indexes, Constraints) ---
    __table_args__ = (
        CheckConstraint('char_length(title) > 0', name='release_title_min_length'),
        CheckConstraint('array_length(changes, 1) <= 50', name='changes_max_length'),
        CheckConstraint('array_length(changes, 1) >= 1', name='changes_min_length'),
        # Move the index inside __table_args__
        Index("idx_release_title_lower", func.lower("title")),  # Use string reference
    )
