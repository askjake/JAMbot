from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ..config import get_settings


def get_timestr_now_utc() -> str:
    return datetime.strftime(datetime.now(tz=timezone.utc), "%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class Health(BaseModel):
    status: str = "Healthy"
    version: str = get_settings().VERSION
    timestamp: str = Field(default_factory=get_timestr_now_utc)
