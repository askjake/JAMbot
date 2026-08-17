import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.db.repository import BaseRepository
from .models import MessageMD
from .schemas import MessageMetadata

logger = logging.getLogger(__name__)

class MessageMDRepository(
    BaseRepository[
        MessageMD,
        MessageMetadata,
        MessageMetadata,
    ]
):
    def __init__(self):
        super().__init__(MessageMD)

    async def get_all_by_chat_id(
        self,
        db: AsyncSession,
        chat_id: str
    ) -> list[MessageMD]:
        try:
            q = select(self.model).where(self.model.chat_id == chat_id)
            rows = await db.execute(q)
            objs = rows.scalars().all()

            return list(objs)
        except SQLAlchemyError as e:
            self._handle_db_error(e, f"getting message metadata history for chat {chat_id}")
        except Exception as e:
            logger.error(f"Unexpected error getting message metadata history for chat {chat_id}: {e}", exc_info=True)
            raise

    async def get_all_by_msg_id(
        self,
        db: AsyncSession,
        message_id: str
    ) -> list[MessageMD]:
        try:
            q = select(self.model).where(self.model.message_id == message_id)
            rows = await db.execute(q)
            objs = rows.scalars().all()

            return list(objs)
        except SQLAlchemyError as e:
            self._handle_db_error(e, f"getting message metadata for message {message_id}")
        except Exception as e:
            logger.error(f"Unexpected error getting message for message {message_id}: {e}", exc_info=True)
            raise