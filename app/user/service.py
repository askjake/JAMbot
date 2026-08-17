from datetime import datetime, date
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from .repository import UserStateRepository

async def get_user_current_chat(db: AsyncSession, user_email: str) -> Optional[str]:
    user_state_repo = UserStateRepository()
    data = await user_state_repo.get_one_by_id(db, user_email)
    if data and data.current_chat_id:
        return str(data.current_chat_id)
    return None

async def set_user_current_chat(db: AsyncSession, user_email: str, chat_id: Optional[str]) -> None:
    user_state_repo = UserStateRepository()
    await UserStateRepository().set_current_chat(db, user_email=user_email, chat_id=chat_id)

async def set_last_release_date(db: AsyncSession, user_email: str, date: datetime) -> None:
    await UserStateRepository().set_last_release_date(db, user_email=user_email, date=date)

async def get_last_release_date(db: AsyncSession, user_email: str) -> date | None:
    user_state_repo = UserStateRepository()
    data = await user_state_repo.get_one_by_id(db, user_email)
    if data and data.last_release_date:
        return data.last_release_date
    return None