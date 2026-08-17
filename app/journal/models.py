from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import ForeignKey, Index, Text, Integer, String
from sqlalchemy.dialects.postgresql import UUID as SQLAlchemyUUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class UserJournal(Base):
    __tablename__ = "user_journal"

    journal_id: Mapped[UUID] = mapped_column(
        SQLAlchemyUUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    owner_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("user_state.owner_id", ondelete="CASCADE"),
        nullable=False
    )
    chat_id: Mapped[UUID] = mapped_column(
        SQLAlchemyUUID(as_uuid=True),
        ForeignKey("chat.chat_id", ondelete="SET NULL"),
        nullable=True
    )
    journal_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Journal Content
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    psychoanalysis: Mapped[dict] = mapped_column(JSONB, nullable=True)
    interaction_patterns: Mapped[dict] = mapped_column(JSONB, nullable=True)
    user_preferences: Mapped[dict] = mapped_column(JSONB, nullable=True)
    topics: Mapped[dict] = mapped_column(JSONB, nullable=True)
    sentiment_analysis: Mapped[dict] = mapped_column(JSONB, nullable=True)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(insert_default=datetime.utcnow)
    conversation_start: Mapped[datetime] = mapped_column(nullable=True)
    conversation_end: Mapped[datetime] = mapped_column(nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), insert_default="active")
    
    __table_args__ = (
        Index("idx_journal_owner", "owner_id", "created_at"),
        Index("idx_journal_chat", "chat_id"),
        Index("idx_journal_type", "journal_type"),
    )
