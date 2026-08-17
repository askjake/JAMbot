import logging
from typing import List, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.db.repository import BaseRepository
from .models import Chat
from .schemas import (
    ChatCreate,
    ChatUpdate,
)

logger = logging.getLogger(__name__)


class ChatRepository(BaseRepository[Chat, ChatCreate, ChatUpdate]):
    def __init__(self):
        super().__init__(Chat)

    async def count_by_owner(
        self,
        db: AsyncSession,
        owner_id: str,
        namespace: str,
        search: str = "",
        group_id: Optional[str] = "all",
    ) -> int:
        """Counts the number of chats owned by a specific user."""
        try:
            search = search.strip()
            q = select(func.count(self.model.chat_id)).filter(
                self.model.owner_id == owner_id,
            )

            if namespace == "generic":
                q = q.filter(self.model.namespace == namespace)
            else:
                q = q.filter(self.model.namespace.like(f"{namespace}%"))

            if group_id == "none":
                q = q.filter(self.model.group_id == None)

            elif group_id != "all":
                q = q.filter(self.model.group_id == group_id)

            if search:
                q = q.where(func.lower(self.model.title).like(f"%{search.lower()}%"))
            rows = await db.execute(q)
            count = rows.scalar_one_or_none()
            return count if count is not None else 0
        except SQLAlchemyError as e:
            self._handle_db_error(e, f"count chat for owner {owner_id}")
        except Exception as e:
            logger.error(
                f"Unexpected error counting chats for owner {owner_id}: {e}",
                exc_info=True,
            )
            raise

    async def list_by_owner(
        self,
        db: AsyncSession,
        owner_id: str,
        namespace: str,
        *,
        offset: int = 0,
        limit: int = 0,
        search: str = "",
        group_id: Optional[str] = "all",
    ) -> List[Chat]:
        """
        List all chats owned by a specific user, ranked by favorite and creation time desc.
        Supports pagination and simple title search.
        """
        search = search.strip()
        try:
            q = select(self.model).filter(self.model.owner_id == owner_id)

            if namespace == "generic":
                q = q.filter(self.model.namespace == namespace)
            else:
                q = q.filter(self.model.namespace.like(f"{namespace}%"))

            if group_id == "none":
                q = q.filter(self.model.group_id == None)

            elif group_id != "all":
                q = q.filter(self.model.group_id == group_id)

            if search:
                q = q.where(func.lower(self.model.title).like(f"%{search.lower()}%"))

            # ORDER BY favorite and creation time DESC
            q = q.order_by(self.model.favorite.desc(), self.model.created_at.desc())

            # Paginate query
            if limit > 0:
                q = q.limit(limit=limit)
            if offset > 0:
                q = q.offset(offset=offset)
            rows = await db.execute(q)
            objs = rows.scalars().all()

            return list(objs)
        except SQLAlchemyError as e:
            self._handle_db_error(e, f"list chats for owner {owner_id}")
        except Exception as e:
            logger.error(
                f"Unexpected error listing chats for owner {owner_id}: {e}",
                exc_info=True,
            )
            raise
