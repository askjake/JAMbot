from enum import Enum
import datetime
from uuid import UUID

from pydantic import BaseModel

from app.core.schemas import Pagination


class PlatformEnum(str, Enum):
    ATV = "AndroidTV"
    STB = "DishTV"


# Pydantic models for data validation
class Release(BaseModel):
    id: int | None = None
    release: str
    platform: PlatformEnum
    release_date: datetime.date
    last_active: datetime.date | None = None
    description: str
    test_activities: str
    softwares: dict


class ReleaseShort(BaseModel):
    id: int
    release: str
    release_date: datetime.date


class BetaReport(BaseModel):
    id: int | None = None
    report_display_id: str
    url: str
    ingest_date: datetime.date
    platform: PlatformEnum
    release: int | None = None
    release_name: str | None = None  # Not part of DB schema
    receiver_id: str | None = None
    hopper_model: str | None = None
    hopperp_model: str | None = None
    joey_model: str | None = None
    hopperp_id: str | None = None
    joey_id: str | None = None
    hopper_software: str | None = None  # Not part of DB schema
    hopperp_software: str | None = None  # Not part of DB schema
    joey_software: str | None = None  # Not part of DB schema
    event_date: datetime.date
    event_time: str
    title: str
    detail: str
    marked_log: bool | None = None
    has_attachment: bool | None = None
    category: str | None = None
    analysis: str = ""
    formalized_report: str = ""
    related_issue: UUID | None = None


class IssueCandidate(BaseModel):
    id: UUID
    platform: PlatformEnum
    title: str
    description: str
    priority: int
    date: datetime.date
    last_updated_date: datetime.date
    accepted: bool | None = None


class DailySummary(BaseModel):
    date: datetime.date
    platform: PlatformEnum
    summary: str
    detailed_recap: str


class IssueFeedback(BaseModel):
    issue_id: UUID
    accept: bool
    comments: str | None = None


class IssueFeedbackResponse(BaseModel):
    success: bool


class ListBetaReportsResponse(Pagination[BetaReport]):
    pass


class ListIssueCandidatesResponse(Pagination[IssueCandidate]):
    pass


class ListReleaseResponse(Pagination[ReleaseShort]):
    pass
