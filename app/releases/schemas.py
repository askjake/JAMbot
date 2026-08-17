from datetime import date

from pydantic import BaseModel

from app.core.schemas import Pagination


class ReleaseCreateResponse(BaseModel):
    release_id: str
    title: str
    date: date
    changes: list[str]



class UpdateLastReleaseLastDateResponse(BaseModel):
    ok: str

class ListReleasesResponse(Pagination[ReleaseCreateResponse]):
    pass


class ReleaseCreate(BaseModel):
    title: str
    date: date
    changes: list[str]

class ReleaseUpdate(BaseModel):
    title: str
    date: date
    changes: list[str]

class ReleaseUpdateLastDate(BaseModel):
    date: date