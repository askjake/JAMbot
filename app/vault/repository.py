from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete

from app.db.repository import BaseRepository

from .models import VaultSession, VaultCred
from .schemas import VaultSessionDBRecord, VaultCredDBRecord

class VaultSessionRepository(
    BaseRepository[
        VaultSession,
        VaultSessionDBRecord,
        VaultSessionDBRecord
    ]
):
    def __init__(self):
        super().__init__(VaultSession)

    async def remove_by_user(self, db: AsyncSession, user_email: str):
        """Remove all sessions for a specific user"""
        stmt = delete(self.model).where(self.model.owner_id == user_email)
        await db.execute(stmt)
        await db.commit()


class VaultCredRepository(
    BaseRepository[
        VaultCred,
        VaultCredDBRecord,
        VaultCredDBRecord
    ]
):
    def __init__(self):
        super().__init__(VaultCred)