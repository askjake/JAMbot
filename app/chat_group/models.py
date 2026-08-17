from uuid import UUID, uuid4

from sqlalchemy import func, Index, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from app.db.mixin import TimestampMixin, LowerCaseEmailMixin


class ChatGroup(Base, TimestampMixin, LowerCaseEmailMixin):
    __tablename__ = "chat_group"

    group_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    title: Mapped[str]
    owner_id: Mapped[str] = mapped_column(nullable=False, index=True)

    __table_args__ = (
        CheckConstraint('char_length(title) > 0', name='chat_title_min_length'),
    )


# Optimize simple title text match using a functional index
Index("idx_chat_group_title_lower", func.lower(ChatGroup.title)),
