import logging
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.db.repository import BaseRepository
from app.releases.exceptions import ReleaseNotFoundError
from app.releases.models import Release
from app.releases.schemas import ReleaseCreate, ReleaseUpdate

logger = logging.getLogger(__name__)


class ReleaseRepository(BaseRepository[Release, ReleaseCreate, ReleaseUpdate]):
    def __init__(self):
        super().__init__(Release)

    async def count_releases(
        self,
        db: AsyncSession,
        search: Optional[str] = "",
        date: Optional[datetime] = None,
    ) -> int:
        """Counts the number of releases."""
        try:
            search = search.strip()
            q = select(func.count()).select_from(self.model)
            if search:
                q = q.where(func.lower(self.model.title).like(f"%{search.lower()}%"))

            if date:
                target_date = date.date() if hasattr(date, "date") else date
                q = q.where(Release.date == target_date)
            rows = await db.execute(q)
            count = rows.scalar_one_or_none()
            return count if count is not None else 0
        except SQLAlchemyError as e:
            self._handle_db_error(e, f"count releases")
        except Exception as e:
            logger.error(f"Unexpected error counting releases: {e}", exc_info=True)
            raise

    async def list_releases(
        self,
        db: AsyncSession,
        *,
        offset: int = 0,
        limit: int = 0,
        search: Optional[str] = "",
        date: Optional[datetime] = None,
    ) -> List[Release]:

        search = search.strip()
        try:
            q = select(self.model)

            if search:
                q = q.where(func.lower(self.model.title).like(f"%{search.lower()}%"))

            if date:
                target_date = date.date() if hasattr(date, "date") else date
                q = q.where(Release.date == target_date)

            # ORDER BY favorite and creation time DESC
            q = q.order_by(self.model.date.desc())

            # Paginate query
            if limit > 0:
                q = q.limit(limit=limit)
            if offset > 0:
                q = q.offset(offset=offset)
            rows = await db.execute(q)
            objs = rows.scalars().all()

            return list(objs)
        except SQLAlchemyError as e:
            self._handle_db_error(e, f"list releases")
        except Exception as e:
            logger.error(f"Unexpected error listing releases: {e}", exc_info=True)
            raise

    async def last_release(self, db: AsyncSession) -> Release:
        try:
            q = select(self.model).order_by(self.model.date.desc()).limit(1)
            result = await db.execute(q)
            obj = result.scalar()
            if not obj:
                raise ReleaseNotFoundError("No release found")
            return obj

        except SQLAlchemyError as e:
            self._handle_db_error(e, f"last release")
            raise
        except Exception as e:
            logger.error(f"Unexpected error last release: {e}", exc_info=True)
            raise
