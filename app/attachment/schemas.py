from datetime import datetime
import uuid

from pydantic import BaseModel, Field

from app.core.utils import get_utc_now_notz
from app.background_mgr import ProgressStatusEnum

class AttachmentDBRecord(BaseModel):
    attachment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    media_type: str
    created_at: datetime = Field(default_factory=get_utc_now_notz)
    updated_at: datetime = Field(default_factory=get_utc_now_notz)
    vault_mode: bool = False
    owner_id: str
    s3_location: str
    preprocessed: bool = False

class AttachmentSchma(BaseModel):
    attachment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    media_type: str
    status: ProgressStatusEnum = ProgressStatusEnum.queued
    created_at: datetime = Field(default_factory=get_utc_now_notz)
    updated_at: datetime = Field(default_factory=get_utc_now_notz)
    vault_mode: bool = False
    owner_id: str

class AttachmentsUpdateReq(BaseModel):
    attachment_ids: list[str]

class AttachmentProcessStatus(BaseModel):
    status: ProgressStatusEnum = ProgressStatusEnum.queued
    progress: int = 0
    message: str = ""

class AttachmentStatusResp(AttachmentSchma, AttachmentProcessStatus):
    pass

class AttachmentUploadError(BaseModel):
    filename: str
    media_type: str
    error_message: str

class AttachmentsStatusResp(BaseModel):
    status: dict[str, AttachmentStatusResp | None]

class AttachmentCreateResp(BaseModel):
    attachments: list[AttachmentSchma | AttachmentUploadError]

class AttachmentFullInfo(AttachmentDBRecord, AttachmentProcessStatus):
    pass