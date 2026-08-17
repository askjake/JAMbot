import logging
from datetime import datetime, date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.db.repository import BaseRepository
from .models import UsageTracking
from .schemas import (
    UsageTrackingCreate,
)

logger = logging.getLogger(__name__)


class UsageTrackingRepository(
    BaseRepository[UsageTracking, UsageTrackingCreate, UsageTrackingCreate]
):
    def __init__(self):
        super().__init__(UsageTracking)

    async def get_by_owner(
        self,
        db: AsyncSession,
        owner_id: str,
        *,
        before: datetime | date | None = None,
        after: datetime | date | None = None,
    ) -> list[UsageTracking]:
        """
        Retrieve usage tracking records for a specific owner, optionally filtered by a timestamp.

        params:
            db (AsyncSession): The database session.
            owner_id (str): The owner's email.
            before (Optional[datetime]): If provided, only records before this timestamp are returned.
            after (Optional[datetime]): If provided, only records after this timestamp are returned.
        returns:
            List of UsageTracking records.
        """
        # inclusive upper bound. lower bound is implicitely inclusive already.
        if isinstance(before, date):
            before = datetime.combine(before, datetime.max.time())
        try:
            q = select(self.model).filter(self.model.owner_id == owner_id)
            if before:
                q = q.filter(self.model.timestamp <= before)
            if after:
                q = q.filter(self.model.timestamp > after)
            q = q.order_by(self.model.timestamp.desc())
            rows = await db.execute(q)
            return rows.scalars().all()
        except SQLAlchemyError as e:
            self._handle_db_error(e, f"retrieve usage tracking for owner {owner_id}")
        except Exception as e:
            logger.error(
                f"Unexpected error retrieving usage tracking for owner {owner_id}: {e}",
                exc_info=True,
            )
            raise

    async def get_by_chat(
        self,
        db: AsyncSession,
        chat_id: str,
    ) -> list[UsageTracking]:
        """
        Retrieve usage tracking records for a specific chat, optionally filtered by a timestamp.

        params:
            db (AsyncSession): The database session.
            chat_id (str): The chat's uuid.
        returns:
            List of UsageTracking records.
        """
        try:
            q = select(self.model).filter(self.model.chat_id == chat_id)
            q = q.order_by(self.model.timestamp.desc())
            rows = await db.execute(q)
            return rows.scalars().all()
        except SQLAlchemyError as e:
            self._handle_db_error(e, f"retrieve usage tracking for chat {chat_id}")
        except Exception as e:
            logger.error(
                f"Unexpected error retrieving usage tracking for chat {chat_id}: {e}",
                exc_info=True,
            )
            raise
