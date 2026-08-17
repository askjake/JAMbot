from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from app.db.mixin import TimestampMixin

from .schemas import ProgressStatusEnum

class BgTask(Base, TimestampMixin):
    """Database model for tracking background tasks"""
    __tablename__ = "bg_tasks"

    task_id: Mapped[UUID] = mapped_column(primary_key=True)
    task_type: Mapped[str] = mapped_column(default="generic")
    status: Mapped[ProgressStatusEnum] = mapped_column(default=ProgressStatusEnum.queued)
    progress: Mapped[int] = mapped_column(default=0)
    total: Mapped[int] = mapped_column(default=0)
    message: Mapped[Optional[str]] = mapped_column(default="")

    def calc_progress_percentage(self) -> int:
        """Calculate the progress percentage"""
        return int((self.progress / self.total) * 100) if self.total > 0 else 0