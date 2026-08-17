from typing import Annotated
import datetime
from fastapi import APIRouter, Query, Depends, HTTPException

from app.dependencies import UserEmailDep

from .schemas import (
    ListBetaReportsResponse,
    ListIssueCandidatesResponse,
    IssueFeedback,
    IssueFeedbackResponse,
    ListReleaseResponse,
)
from .exceptions import ResourceNotFoundError
from .service import (
    BetaReportService,
    get_betareport_service,
)

BetaReportServiceDep = Annotated[BetaReportService, Depends(get_betareport_service)]

router = APIRouter(prefix="/betareport", tags=["betareport"])


@router.get("/releases")
async def list_releases(
    email: UserEmailDep,
    br_service: BetaReportServiceDep,
    page: int = Query(ge=1, default=1),
    limit: int = Query(ge=1, le=100, default=50),
    start_date: Annotated[datetime.date | None, Query()] = None,
    end_date: Annotated[datetime.date | None, Query()] = None,
    platform: Annotated[str | None, Query()] = None,
) -> ListReleaseResponse:
    """List all chats owned by the user. Supports pagination and simple title search"""
    offset = (page - 1) * limit
    end_idx = offset + limit  # Index of end past one
    total_docs, releases = await br_service.list_beta_releases(
        offset=offset,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        platform=platform,
    )

    return ListReleaseResponse(
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


@router.get("/available-platforms")
async def list_available_platforms(
    email: UserEmailDep,
    br_service: BetaReportServiceDep,
) -> list[str]:
    """List all chats owned by the user. Supports pagination and simple title search"""
    return await br_service.list_available_platforms()


@router.get("/available-devices")
async def list_available_devices(
    email: UserEmailDep,
    br_service: BetaReportServiceDep,
) -> list[str]:
    """List all chats owned by the user. Supports pagination and simple title search"""
    devices = await br_service.list_available_devices()
    return devices


@router.get("/reports")
async def list_reports(
    email: UserEmailDep,
    br_service: BetaReportServiceDep,
    page: int = Query(ge=1, default=1),
    limit: int = Query(ge=1, le=100, default=50),
    start_date: Annotated[datetime.date | None, Query()] = None,
    end_date: Annotated[datetime.date | None, Query()] = None,
    release: Annotated[int | None, Query()] = None,
    software: Annotated[str | None, Query()] = None,
    platform: Annotated[str | None, Query()] = None,
    device: Annotated[str | None, Query()] = None,
) -> ListBetaReportsResponse:
    """List all chats owned by the user. Supports pagination and simple title search"""
    offset = (page - 1) * limit
    end_idx = offset + limit  # Index of end past one
    total_docs, reports = await br_service.list_beta_reports(
        offset=offset,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        release=release,
        software=software,
        platform=platform,
        device=device,
    )

    return ListBetaReportsResponse(
        docs=reports,
        totalDocs=total_docs,
        limit=limit,
        page=page,
        totalPages=(total_docs + limit - 1) // limit,
        hasNextPage=end_idx < total_docs,
        nextPage=page + 1 if end_idx < total_docs else None,
        hasPrevPage=page > 1,
        prevPage=page - 1 if page > 1 else None,
    )


@router.get("/issues")
async def list_issues(
    email: UserEmailDep,
    br_service: BetaReportServiceDep,
    page: int = Query(ge=1, default=1),
    limit: int = Query(ge=1, le=100, default=50),
    start_date: Annotated[datetime.date | None, Query()] = None,
    end_date: Annotated[datetime.date | None, Query()] = None,
    release: Annotated[int | None, Query()] = None,
    platform: Annotated[str | None, Query()] = None,
    min_priority: Annotated[int | None, Query(ge=0, le=5)] = None,
    max_priority: Annotated[int | None, Query(ge=0, le=5)] = None,
) -> ListIssueCandidatesResponse:
    """List issue candidates with pagination"""
    offset = (page - 1) * limit
    end_idx = offset + limit

    total_docs, issues = await br_service.list_issue_candidates(
        email=email,
        offset=offset,
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        release=release,
        platform=platform,
        min_priority=min_priority,
        max_priority=max_priority,
    )

    return ListIssueCandidatesResponse(
        docs=issues,
        totalDocs=total_docs,
        limit=limit,
        page=page,
        totalPages=(total_docs + limit - 1) // limit,
        hasNextPage=end_idx < total_docs,
        nextPage=page + 1 if end_idx < total_docs else None,
        hasPrevPage=page > 1,
        prevPage=page - 1 if page > 1 else None,
    )


@router.post("/feedback")
async def take_feedback(
    email: UserEmailDep,
    br_service: BetaReportServiceDep,
    feedback: IssueFeedback,
) -> IssueFeedbackResponse:
    """Submit feedback on an issue candidate"""
    try:
        await br_service.process_issue_feedback(feedback, email)
    except ResourceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to process feedback")

    return IssueFeedbackResponse(success=True)
