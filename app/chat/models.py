from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    func,
    Index,
    CheckConstraint,
    UUID as SQLAlchemyUUID,
    ForeignKey,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from app.db.mixin import TimestampMixin, LowerCaseEmailMixin
from .schemas import ChatStatusEnum


class Chat(Base, TimestampMixin, LowerCaseEmailMixin):
    __tablename__ = "chat"

    chat_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    title: Mapped[str]
    owner_id: Mapped[str]
    last_message_at: Mapped[datetime] = mapped_column(insert_default=func.now())
    vault_mode: Mapped[bool] = mapped_column(insert_default=False)
    favorite: Mapped[bool] = mapped_column(insert_default=False)
    active_checkpoint: Mapped[UUID] = mapped_column(nullable=True)
    status: Mapped[ChatStatusEnum] = mapped_column(insert_default=ChatStatusEnum.normal)
    status_msg: Mapped[str] = mapped_column(nullable=True)
    namespace: Mapped[str] = mapped_column(
        server_default="generic",
        comment="Namespace for the chat (i.e. belongs to a certain app)",
    )
    # Foreign Key to the group table
    group_id: Mapped[UUID] = mapped_column(
        SQLAlchemyUUID(as_uuid=True),
        ForeignKey("chat_group.group_id", ondelete="SET NULL"),
        nullable=True,
        comment="Foreign key linking to the group of chats",
        index=True,
    )
    # Flag to prevent re-checking idle chats that have already been evaluated
    idle_checked: Mapped[bool] = mapped_column(
        insert_default=False,
        comment="True once idle checker has evaluated this chat; prevents re-checks",
    )

    # --- Table Arguments (Indexes, Constraints) ---
    __table_args__ = (
        CheckConstraint("char_length(title) > 0", name="chat_title_min_length"),
        # Optimize list chat operation
        Index(
            "idx_chat_owner_favorite_created",
            "owner_id",
            text("favorite DESC"),
            text("created_at DESC"),
        ),
        # Optimize vault operation
        Index("idx_chat_owner_vault", "owner_id", "vault_mode"),
        # Optimize simple title text match using a functional index
        Index("idx_chat_title_lower", text("lower(title)")),
        # Optimize idle chat checker query
        Index("idx_chat_idle_checked", "idle_checked"),
    )
