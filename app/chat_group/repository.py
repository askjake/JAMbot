import logging
from typing import List

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.db.repository import BaseRepository
from .models import ChatGroup
from .schemas import (
    ChatGroupCreate,
    ChatGroupUpdate,
)

logger = logging.getLogger(__name__)


class ChatGroupRepository(
    BaseRepository[
        ChatGroup,
        ChatGroupCreate,
        ChatGroupUpdate
    ]
):
    def __init__(self):
        super().__init__(ChatGroup)
        
    async def count_by_owner(self, db: AsyncSession, owner_id: str, search: str = "",) -> int:
        """Counts the number of groups with chats owned by a specific user."""
        try:
            q = select(func.count(self.model.group_id)).where(self.model.owner_id == owner_id)
            if search:
              q = q.where(func.lower(self.model.title).like(f"%{search.lower()}%"))
            rows = await db.execute(q)
            count = rows.scalar_one_or_none()
            return count if count is not None else 0
        except SQLAlchemyError as e:
            self._handle_db_error(e, f"count groups with chats for owner {owner_id}")
        except Exception as e:
            logger.error(f"Unexpected error counting groups with chats for owner {owner_id}: {e}", exc_info=True)
            raise
        
    async def list_by_owner(
        self,
        db: AsyncSession,
        owner_id: str,
        *,
        offset: int = 0,
        limit: int = 0,
        search: str = "",
    ) -> List[ChatGroup]:
        """
        List all groups of chats owned by a specific user, ranked by favorite and creation time desc.
        Supports pagination and simple title search.
        """
        search = search.strip()
        try:
            q = select(self.model).where(self.model.owner_id == owner_id)
 
            if search:
                q = q.where(func.lower(self.model.title).like(f"%{search.lower()}%"))

            # ORDER BY creation time DESC
            q = q.order_by(self.model.created_at.desc())

            # Paginate query
            if limit > 0:
                q = q.limit(limit=limit)
            if offset > 0:
                q = q.offset(offset=offset)
            rows = await db.execute(q)
            objs = rows.scalars().all()

            return list(objs)
        except SQLAlchemyError as e:
            self._handle_db_error(e, f"list groups with chats for owner {owner_id}")
        except Exception as e:
            logger.error(f"Unexpected error listing groups with chats for owner {owner_id}: {e}", exc_info=True)
            raise