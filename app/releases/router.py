from datetime import datetime
from typing import Annotated, Optional

from fastapi import Depends, APIRouter, Query

from app.dependencies import DBSessionDep, UserEmailDep
from app.releases.schemas import ListReleasesResponse, ReleaseCreateResponse, UpdateLastReleaseLastDateResponse, \
    ReleaseUpdateLastDate
from app.releases.service import ReleaseService, get_release_service

ReleaseServiceDep = Annotated[ReleaseService, Depends(get_release_service)]

router = APIRouter()

# Releases endpoints
@router.get("/releases")
async def list_releases(
    db_session: DBSessionDep,
    release_service: ReleaseServiceDep,
    page: int = Query(ge=1, default=1),
    limit: int = Query(ge=1, le=100, default=50),
    search: Optional[str] = Query(max_length=40, default=""),
    date: Optional[datetime] = Query(default=None)
) -> ListReleasesResponse:
    """List all releases. Supports pagination and simple title search"""
    offset = (page - 1) * limit
    end_idx = offset + limit  # Index of end past one
    releases = await release_service.list_releases(
        db_session,
        offset=offset,
        limit=limit,
        search=search,
        date=date,
    )
    total_docs = await release_service.count_releases(db_session, search=search, date=date)

    return ListReleasesResponse(
        docs=releases,
        totalDocs=total_docs,
        limit=limit,
        page=page,
        totalPages=(total_docs + limit - 1) // limit,
        hasNextPage=end_idx < total_docs,
        nextPage=page + 1 if end_idx < total_docs else None,
        hasPrevPage=page > 1,
        prevPage=page - 1 if page > 1 else None,
    )

@router.get("/releases/last")
async def last_release(
    db_session: DBSessionDep,
    release_service: ReleaseServiceDep,
) -> ReleaseCreateResponse:
    """Get last release"""
    release = await release_service.last_release(db_session)

    return ReleaseCreateResponse(
        release_id=str(release.release_id),
        title=release.title,
        date=release.date,
        changes=release.changes
    )

@router.put("/releases", response_model=UpdateLastReleaseLastDateResponse)
async def update_last_release_date(
    email: UserEmailDep,
    db_session: DBSessionDep,
    release_service: ReleaseServiceDep,
    data: ReleaseUpdateLastDate,
) -> UpdateLastReleaseLastDateResponse:
    await release_service.update_last_release_date(db_session, user_email=email, date=data.date)
    return UpdateLastReleaseLastDateResponse(
        ok="The date has been updated.",
    )