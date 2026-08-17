from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import StreamingResponse

from app.dependencies import DBSessionDep, UserEmailDep
from app.core.utils import datetime_to_iso_utc, UUID_REGEX
from app.chat.dependencies import ChatServiceDep
from app.chat.exceptions import ChatNotFoundError, NotAuthorizedError, VaultAccessError
from app.vault.dependencies import VaultKeyDep
from app.usage_tracking.dependencies import UsageTrackingServiceDep
from app.config import get_settings
from app.message.dependencies import MessageServiceDep
from .schemas import (
    ChatHistoryResp,
    MsgHistoryRespPaginated,
    InputUserMessage,
    MessageVersionsResp,
    UserMessageUpdate,
    MessageVersionUpdate,
    MessageVersionUpdateOutput,
)
from .exceptions import (
    VersionLimitExceededError,
    NotUserMessageError,
    HasAttachmentsError,
    VersionNotFoundError,
    ChatReadOnlyError,
)

router = APIRouter()
settings = get_settings()


@router.get("/chats/{chat_id}")
async def get_chat_messages(
    chat_id: Annotated[str, Path(regex=UUID_REGEX)],
    email: UserEmailDep,
    db_session: DBSessionDep,
    chat_service: ChatServiceDep,
    message_service: MessageServiceDep,
    vault_key: VaultKeyDep,
) -> ChatHistoryResp:
    try:
        chat = await chat_service.get_chat_if_authorized(
            db_session, chat_id, email, vault_key != ""
        )
        messages = await message_service.get_chat_history(
            db_session, chat_id, email, vault_key
        )
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="Chat not found")
    except NotAuthorizedError:
        raise HTTPException(
            status_code=403, detail="Not authorized to delete this chat"
        )
    except VaultAccessError:
        raise HTTPException(
            status_code=401, detail="Accessing vault chats requires vault mode enabled"
        )

    return ChatHistoryResp(
        chat_id=str(chat.chat_id),
        title=chat.title[: settings.MAX_TITLE_LEN - 1],
        created_at=datetime_to_iso_utc(chat.created_at),
        last_message_at=datetime_to_iso_utc(chat.last_message_at),
        owner_id=chat.owner_id,
        namespace=chat.namespace,
        vault_mode=chat.vault_mode,
        favorite=chat.favorite,
        status=chat.status,
        status_msg=chat.status_msg,
        messages=messages,
    )


@router.get("/chats/{chat_id}/messages")
async def get_message_history_paginated(
    chat_id: Annotated[str, Path(regex=UUID_REGEX)],
    email: UserEmailDep,
    db_session: DBSessionDep,
    chat_service: ChatServiceDep,
    message_service: MessageServiceDep,
    vault_key: VaultKeyDep,
    page: int = Query(ge=1, default=1),
    limit: int = Query(ge=1, le=100, default=50),
) -> MsgHistoryRespPaginated:
    """Return a paginated view of message history for the requested chat.
    Messages returned in reverse order (newest-first)
    """
    try:
        await chat_service.get_chat_if_authorized(
            db_session, chat_id, email, vault_key != ""
        )
        messages = await message_service.get_chat_history(
            db_session, chat_id, email, vault_key
        )
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="Chat not found")
    except NotAuthorizedError:
        raise HTTPException(
            status_code=403, detail="Not authorized to delete this chat"
        )
    except VaultAccessError:
        raise HTTPException(
            status_code=401, detail="Accessing vault chats requires vault mode enabled"
        )
    messages = list(messages.values())
    messages.reverse()
    total_docs = len(messages)
    offset = (page - 1) * limit
    end_idx = offset + limit
    page_count = (len(messages) + limit - 1) // limit

    return MsgHistoryRespPaginated(
        docs=messages[offset : offset + limit],
        totalDocs=len(messages),
        limit=limit,
        page=page,
        totalPages=page_count,
        hasNextPage=end_idx < total_docs,
        nextPage=page + 1 if end_idx < total_docs else None,
        hasPrevPage=page > 1,
        prevPage=page - 1 if page > 1 else None,
    )


@router.post("/chats/{chat_id}/messages")
async def send_message(
    chat_id: Annotated[str, Path(regex=UUID_REGEX)],
    message: InputUserMessage,
    email: UserEmailDep,
    db_session: DBSessionDep,
    message_service: MessageServiceDep,
    usage_tracking_service: UsageTrackingServiceDep,
    vault_key: VaultKeyDep,
) -> StreamingResponse:
    try:
        stream_agen = await message_service.create_new_message(
            db_session, chat_id, email, message, vault_key
        )
        resp_agen = usage_tracking_service.track_astream_generator(
            stream_agen,
            profile={"owner_email": email, "chat_id": chat_id, "task": "chat"},
        )
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="Chat not found")
    except NotAuthorizedError:
        raise HTTPException(
            status_code=403, detail="Not authorized to delete this chat"
        )
    except VaultAccessError:
        raise HTTPException(
            status_code=401, detail="Accessing vault chats requires vault mode enabled"
        )
    except ChatReadOnlyError:
        raise HTTPException(
            status_code=403, detail="Modifying readonly chat is not allowed"
        )

    return StreamingResponse(resp_agen, media_type="text/event-stream")


@router.get("/chats/{chat_id}/messages/{message_id}/versions")
async def get_message_versions(
    chat_id: Annotated[str, Path(regex=UUID_REGEX)],
    message_id: Annotated[str, Path(regex=UUID_REGEX)],
    email: UserEmailDep,
    db_session: DBSessionDep,
    message_service: MessageServiceDep,
    vault_key: VaultKeyDep,
) -> MessageVersionsResp:
    try:
        message_versions = await message_service.get_message_versions(
            db_session, chat_id, email, message_id, vault_key
        )
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="Chat not found")
    except NotAuthorizedError:
        raise HTTPException(
            status_code=403, detail="Not authorized to delete this chat"
        )
    except VaultAccessError:
        raise HTTPException(
            status_code=401, detail="Accessing vault chats requires vault mode enabled"
        )

    return message_versions


@router.post("/chats/{chat_id}/messages/{message_id}/versions")
async def create_message_version(
    chat_id: Annotated[str, Path(regex=UUID_REGEX)],
    message_id: Annotated[str, Path(regex=UUID_REGEX)],
    user_message: UserMessageUpdate,
    email: UserEmailDep,
    db_session: DBSessionDep,
    message_service: MessageServiceDep,
    usage_tracking_service: UsageTrackingServiceDep,
    vault_key: VaultKeyDep,
):
    try:
        stream_agen = await message_service.branch_past_message(
            db_session, chat_id, message_id, email, user_message.content, vault_key
        )
        resp_agen = usage_tracking_service.track_astream_generator(
            stream_agen,
            profile={"owner_email": email, "chat_id": chat_id, "task": "chat"},
        )
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="Chat not found")
    except NotAuthorizedError:
        raise HTTPException(
            status_code=403, detail="Not authorized to delete this chat"
        )
    except VaultAccessError:
        raise HTTPException(
            status_code=401, detail="Accessing vault chats requires vault mode enabled"
        )
    except VersionLimitExceededError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotUserMessageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HasAttachmentsError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ChatReadOnlyError:
        raise HTTPException(
            status_code=403, detail="Modifying readonly chat is not allowed"
        )

    # Stream mock response for the new version
    return StreamingResponse(resp_agen, media_type="text/event-stream")


@router.put("/chats/{chat_id}/messages/{message_id}/versions")
async def update_message_version(
    chat_id: Annotated[str, Path(regex=UUID_REGEX)],
    message_id: Annotated[str, Path(regex=UUID_REGEX)],
    version: MessageVersionUpdate,
    email: UserEmailDep,
    db_session: DBSessionDep,
    message_service: MessageServiceDep,
    vault_key: VaultKeyDep,
) -> MessageVersionUpdateOutput:
    try:
        branch_history = await message_service.get_branch_history(
            db_session, chat_id, message_id, version.version_index, email, vault_key
        )
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="Chat not found")
    except NotAuthorizedError:
        raise HTTPException(
            status_code=403, detail="Not authorized to delete this chat"
        )
    except VaultAccessError:
        raise HTTPException(
            status_code=401, detail="Accessing vault chats requires vault mode enabled"
        )
    except VersionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not branch_history:
        raise HTTPException(
            status_code=404,
            detail="Branch history for requested message version was not found",
        )

    active_message = None
    for m in branch_history.values():
        if m.message_id == message_id:
            active_message = m
            break

    if not active_message:
        raise HTTPException(
            status_code=404, detail="Requested message not found in branch history"
        )

    return MessageVersionUpdateOutput(
        active_message=active_message, branched_history=branch_history
    )
