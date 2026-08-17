from datetime import date

from sqlalchemy import ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class UserState(Base):
    __tablename__ = "user_state"

    owner_id: Mapped[str] = mapped_column(primary_key=True)
    current_chat_id = mapped_column(
        ForeignKey("chat.chat_id", ondelete="SET NULL"), nullable=True
    )
    last_release_date: Mapped[date] = mapped_column(
        DateTime, nullable=True, default=None
    )
