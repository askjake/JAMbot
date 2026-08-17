import uuid

from sqlalchemy import (
    ForeignKey,
    Index,
    Enum as SQLAlchemyEnum,
    UUID as SQLAlchemyUUID,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from app.db import Base
from .schemas import MessageRoleEnum

class MessageMD(Base):
    """
    SQLAlchemy model representing a message metadata record, tracking its versions
    and corresponding LangGraph checkpoint.
    """
    __tablename__ = "message_metadata"

    checkpoint_id: Mapped[str] = mapped_column(
        Text,
        primary_key=True,
        comment="The LangGraph checkpoint ID associated with this specific message version."
    )
    message_id: Mapped[uuid.UUID] = mapped_column(
        SQLAlchemyUUID(as_uuid=True),
        comment="Unique identifier for a logical message (shared across versions)."
    )
    # Foreign Key to the chat table
    chat_id: Mapped[uuid.UUID] = mapped_column(
        SQLAlchemyUUID(as_uuid=True),
        ForeignKey("chat.chat_id", ondelete="CASCADE"),
        nullable=False,
        comment="Foreign key linking to the chat session."
    )
    role: Mapped[MessageRoleEnum] = mapped_column(
        SQLAlchemyEnum(
            MessageRoleEnum,
            name="message_role_enum",
            values_callable=lambda enum: [e.value for e in enum],
        ),
        nullable=False,
        comment="Role of the message sender (user, assistant, tool)."
    )
    message_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=True,
        comment="The additional model kwargs used to generate the response message."
    )

    parent_checkpoint_id: Mapped[str] = mapped_column(
        Text,
        nullable=True,
        comment="The checkpoint_id of the preceding message in this conversational branch."
    )

    # Define indexes
    __table_args__ = (
        Index("idx_message_chat_id", "chat_id"), # Index for faster lookups by chat_id
        Index("idx_message_message_id", "message_id"), # Index for faster lookups by chat_id
    )

    def __repr__(self) -> str:
        return (
            f"<Message(message_id={self.message_id}, checkpoint_id='{self.checkpoint_id}')>"
            f"chat_id={self.chat_id}, role='{self.role.value}', "
        )
