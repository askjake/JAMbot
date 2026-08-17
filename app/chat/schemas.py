from typing import Annotated, Optional
from enum import Enum

from pydantic import BaseModel, Field, StringConstraints

from app.core.utils import *
from app.core.schemas import Pagination

class ChatStatusEnum(Enum):
    normal = "normal"
    readonly = "readonly"

class ChatCreateResponse(BaseModel):
    chat_id: str
    title: str
    owner_id: str
    namespace: str
    created_at: str
    last_message_at: str
    vault_mode: bool = False
    favorite: bool = False
    status: ChatStatusEnum = ChatStatusEnum.normal
    status_msg: Optional[str] = None
    group_id: Optional[str] = None

class ListChatsResponse(Pagination[ChatCreateResponse]):
    active_chat_id: str

class ChatUpdateRequest(BaseModel):
    title: Annotated[
                Optional[str],
                StringConstraints(
                    strip_whitespace=True, min_length=1, max_length=settings.MAX_TITLE_LEN
                ),
            ] = None
    favorite: Optional[bool] = None
    active: Optional[bool] = None
    group_id: Optional[str] = None

class ChatUpdateResponse(ChatUpdateRequest):
    chat_id: str
    updated_at: str = Field(default_factory=get_timestr_now_utc)

class ChatCreate(BaseModel):
    title: str
    owner_id: str
    namespace: str = "generic"
    vault_mode: Optional[bool]
    group_id: Optional[str] = None

class ChatUpdate(BaseModel):
    title: str
    favorite: bool
    group_id: Optional[str] = None

