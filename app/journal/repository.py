from typing import Optional, List
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import BaseRepository
from .models import UserJournal
from .schemas import JournalCreate


class JournalRepository(BaseRepository[UserJournal, JournalCreate, dict]):
    def __init__(self):
        super().__init__(UserJournal)
    
    async def get_by_owner(
        self,
        db: AsyncSession,
        owner_id: str,
        limit: int = 50,
        offset: int = 0,
        journal_type: Optional[str] = None
    ) -> List[UserJournal]:
        """Get journals for a specific owner"""
        query = select(UserJournal).where(UserJournal.owner_id == owner_id)
        
        if journal_type:
            query = query.where(UserJournal.journal_type == journal_type)
        
        query = query.order_by(UserJournal.created_at.desc())
        query = query.offset(offset).limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    async def count_by_owner(
        self,
        db: AsyncSession,
        owner_id: str,
        journal_type: Optional[str] = None
    ) -> int:
        """Count journals for a specific owner"""
        query = select(func.count()).select_from(UserJournal).where(
            UserJournal.owner_id == owner_id
        )
        
        if journal_type:
            query = query.where(UserJournal.journal_type == journal_type)
        
        result = await db.execute(query)
        return result.scalar()
    
    async def get_by_chat_id(
        self,
        db: AsyncSession,
        chat_id: UUID
    ) -> Optional[UserJournal]:
        """Get journal for a specific chat"""
        query = select(UserJournal).where(UserJournal.chat_id == chat_id)
        result = await db.execute(query)
        return result.scalar_one_or_none()
