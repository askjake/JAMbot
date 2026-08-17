import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.repository import BaseRepository

from .models import UserState
from .schemas import UserStateCreate, UserStateUpdate

logger = logging.getLogger(__name__)

class UserStateRepository(BaseRepository[UserState, UserStateCreate, UserStateUpdate]):
    def __init__(self):
        super().__init__(UserState)

    async def set_current_chat(self, db: AsyncSession, user_email: str, chat_id: Optional[str]) -> UserState:
        """Set the current chat id for user"""
        try:
            obj = UserState(owner_id=user_email, current_chat_id=chat_id)
            merged_obj = await db.merge(obj)
            await db.flush()
            await db.refresh(merged_obj)
            logger.info(f"Set current chat for user '{user_email}' to chat_id '{chat_id}'.")
            return merged_obj

        except SQLAlchemyError as e:
            self._handle_db_error(e, f"set current chat for user {user_email}")
        except Exception as e:
            logger.error(f"Unexpected error setting current chat for user {user_email}: {e}", exc_info=True)
            raise

    async def set_last_release_date(self, db: AsyncSession, user_email: str,  date: datetime) -> UserState:
        """Set the last release date for user"""
        try:
            obj = UserState(owner_id=user_email, last_release_date=date)
            merged_obj = await db.merge(obj)
            await db.flush()
            await db.refresh(merged_obj)
            logger.info(f"Set the last release date for user '{user_email}'. Date '{date.strftime('%d.%m.%Y')}'.")
            return merged_obj
        except SQLAlchemyError as e:
            self._handle_db_error(e, f"set the last release date for user {user_email}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error setting  the last release date for user {user_email}: {e}", exc_info=True)
            raise