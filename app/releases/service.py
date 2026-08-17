import logging
from datetime import datetime
from typing import Optional, List

from sqlalchemy.ext.asyncio import AsyncSession

from app.releases.models import Release
from app.releases.repository import ReleaseRepository
from app.releases.schemas import ReleaseCreateResponse
from app.user.service import  set_last_release_date

logger = logging.getLogger(__name__)

def get_release_service():
    return ReleaseService()

class ReleaseService:
    def __init__(self, release_repo: Optional[ReleaseRepository] = None):
        self.release_repo = release_repo or ReleaseRepository()

    async def count_releases(self, db: AsyncSession,  search: str = "", date: Optional[datetime] = None) -> int:
        """Count number of releases"""
        return await self.release_repo.count_releases(db,  search=search, date=date)

    async def list_releases(
        self,
        db: AsyncSession,
        *,
        offset: int = 0,
        limit: int = 0,
        search: Optional[str] = "",
        date: Optional[datetime] = None,
    ) -> List[ReleaseCreateResponse]:
        releases = await self.release_repo.list_releases(db, offset=offset, limit=limit, search=search, date=date)
        return [
            ReleaseCreateResponse(
                release_id=str(release.release_id),
                title=release.title,
                date=release.date,
                changes=release.changes
            )
            for release in  releases
        ]

    async def last_release(
        self,
        db: AsyncSession,
    ) -> Release:
        release = await self.release_repo.last_release(db)
        return release


    async def update_last_release_date(self, db: AsyncSession, user_email: str, date: datetime) -> None:
        await set_last_release_date(db, user_email=user_email, date=date)
