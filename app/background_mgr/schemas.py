from typing import Optional
from enum import Enum
from uuid import UUID

from pydantic import BaseModel

class ProgressStatusEnum(str, Enum):
    """Enum for tracking the status of background tasks"""
    queued = "queued"
    processing = "processing"
    ready = "ready"
    failed = "failed"

class BgTaskSchema(BaseModel):
    task_id: UUID
    task_type: Optional[str] = "generic"

class BgTaskProgress(BgTaskSchema):
    progress: int = 0
    status: ProgressStatusEnum = ProgressStatusEnum.queued
    message: str = ""