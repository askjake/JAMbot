import uuid
from datetime import datetime

from sqlalchemy import (
    ForeignKey,
    func,
    Index,
    UUID as SQLAlchemyUUID,
)
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base
from app.db.mixin import LowerCaseEmailMixin


class UsageTracking(Base, LowerCaseEmailMixin):
    """
    SQLAlchemy model representing a token usage tracking record.
    """

    __tablename__ = "usage_tracking"
    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
        comment="Unique identifier for the usage tracking record",
    )
    owner_id: Mapped[str] = mapped_column(
        comment="Owner email",
    )
    chat_id: Mapped[uuid.UUID] = mapped_column(
        SQLAlchemyUUID(as_uuid=True),
        ForeignKey("chat.chat_id", ondelete="SET NULL"),
        nullable=True,
        comment="Foreign key linking to the chat",
    )
    timestamp: Mapped[datetime] = mapped_column(
        insert_default=func.now(),
        comment="Timestamp of the usage event",
    )
    model: Mapped[str] = mapped_column(
        comment="Model used for the chat",
    )
    task: Mapped[str] = mapped_column(
        comment="Task type, e.g., chat, tool, embedding, etc.",
    )
    input_tokens: Mapped[int] = mapped_column(
        default=0,
        comment="Number of input tokens used, not counting cache reads or creates",
    )
    input_cache_read: Mapped[int] = mapped_column(
        default=0,
        comment="Number of input tokens read from cache",
    )
    input_cache_create: Mapped[int] = mapped_column(
        default=0,
        comment="Number of input token cache created",
    )
    output_tokens: Mapped[int] = mapped_column(
        default=0,
        comment="Number of output tokens generated",
    )
    input_cost: Mapped[float] = mapped_column(
        default=0.0,
        comment="Cost incurred for input tokens",
    )
    output_cost: Mapped[float] = mapped_column(
        default=0.0,
        comment="Cost incurred for output tokens",
    )


# Optimize queries filtering by chat_id
Index("idx_usage_tracking_chat_id", UsageTracking.chat_id),

# Optimize queries filtering by owner_id and timestamp
Index(
    "idx_usage_tracking_owner_timestamp",
    UsageTracking.owner_id,
    UsageTracking.timestamp.desc(),
),
