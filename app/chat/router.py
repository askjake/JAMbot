# /chat/router.py

from typing import Annotated, List
from fastapi import APIRouter, Query, HTTPException, Path

from app.core.utils import datetime_to_iso_utc
from app.dependencies import DBSessionDep, UserEmailDep
from app.core.utils import UUID_REGEX
from app.user.service import get_user_current_chat
from app.vault.dependencies import VaultKeyDep
from app.config import get_settings

from app.chat.models import Chat
from app.chat.schemas import (
    ChatCreateResponse,
    ListChatsResponse,
    ChatUpdateRequest,
    ChatUpdateResponse,
)
from app.chat.dependencies import ChatServiceDep
from app.chat.exceptions import (
    ChatLimitExceededError,
    ChatNotFoundError,
    NotAuthorizedError,
    VaultAccessError,
    NamespaceNotFoundError,
)

settings = get_settings()


def map_chat_to_chatcreate(
    objs: Chat | List[Chat],
) -> ChatCreateResponse | List[ChatCreateResponse]:
    """Map the Chat model to ChatCreateResponse schema"""
    return_list = True
    if not isinstance(objs, list):
        objs = [objs]
        return_list = False
    schemas = [
        ChatCreateResponse(
            chat_id=str(chat.chat_id),
            group_id=chat.group_id if chat.group_id == None else str(chat.group_id),
            title=chat.title[
                : settings.MAX_TITLE_LEN - 1
            ],  # Ensure title leng does not cause validation error in ui
            created_at=datetime_to_iso_utc(chat.created_at),
            last_message_at=datetime_to_iso_utc(chat.last_message_at),
            owner_id=chat.owner_id,
            namespace=chat.namespace,
            vault_mode=chat.vault_mode,
            favorite=chat.favorite,
            status=chat.status,
            status_msg=chat.status_msg,
        )
        for chat in objs
    ]

    if not return_list:
        return schemas[0]
    return schemas


router = APIRouter()


# Chat endpoints
@router.post("/chats", response_model=ChatCreateResponse)
async def create_chat(
    email: UserEmailDep,
    db_session: DBSessionDep,
    chat_service: ChatServiceDep,
    vault_key: VaultKeyDep,
    namespace: str = Query(max_length=100, default="generic"),
) -> ChatCreateResponse:
    """Create a new empty chat for the user"""
    is_vault_mode = bool(vault_key)

    try:
        new_chat = await chat_service.create_chat_for_user(
            db_session, email, namespace, is_vault_mode
        )
        return map_chat_to_chatcreate(new_chat)
    except ChatLimitExceededError as e:
        raise HTTPException(
            status_code=400, detail={"error": "CHAT_LIMIT_EXCEEDED", "message": str(e)}
        )
    except NamespaceNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/chats")
async def list_chats(
    email: UserEmailDep,
    db_session: DBSessionDep,
    chat_service: ChatServiceDep,
    page: int = Query(ge=1, default=1),
    limit: int = Query(ge=1, le=100, default=50),
    search: str = Query(max_length=40, default=""),
    group_id: str = Query(max_length=100, default="all"),
    namespace: str = Query(max_length=100, default="generic"),
) -> ListChatsResponse:
    """List all chats owned by the user. Supports pagination and simple title search"""
    offset = (page - 1) * limit
    end_idx = offset + limit  # Index of end past one
    user_chats = await chat_service.list_chat_for_user(
        db_session,
        email,
        namespace,
        offset=offset,
        limit=limit,
        search=search,
        group_id=group_id,
    )
    active_chat_id = await get_user_current_chat(db_session, email)
    total_docs = await chat_service.count_by_owner(
        db_session, email, namespace, search=search, group_id=group_id
    )

    return ListChatsResponse(
        docs=map_chat_to_chatcreate(user_chats),
        totalDocs=total_docs,
        limit=limit,
        page=page,
        totalPages=(total_docs + limit - 1) // limit,
        hasNextPage=end_idx < total_docs,
        nextPage=page + 1 if end_idx < total_docs else None,
        hasPrevPage=page > 1,
        prevPage=page - 1 if page > 1 else None,
        active_chat_id=active_chat_id or "",
    )


@router.put("/chats/{chat_id}", response_model=ChatUpdateResponse)
async def update_chat(
    chat_id: Annotated[str, Path(regex=UUID_REGEX)],
    update: ChatUpdateRequest,
    email: UserEmailDep,
    db_session: DBSessionDep,
    chat_service: ChatServiceDep,
    vault_key: VaultKeyDep,
) -> ChatUpdateResponse:
    is_vault_mode = bool(vault_key)

    try:
        updated_chat = await chat_service.update_chat(
            db_session, email, is_vault_mode, chat_id, update
        )

        return ChatUpdateResponse(
            chat_id=str(updated_chat.chat_id),
            title=updated_chat.title[: settings.MAX_TITLE_LEN - 1],
            favorite=updated_chat.favorite,
            active=chat_id == await get_user_current_chat(db_session, email),
        )
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="Chat not found")
    except NotAuthorizedError:
        raise HTTPException(
            status_code=403, detail="Not authorized to modify this chat"
        )
    except VaultAccessError:
        raise HTTPException(
            status_code=401, detail="Modifying vault chats requires vault mode enabled"
        )


@router.delete("/chats/{chat_id}", tags=["chat"])
async def delete_chat(
    chat_id: Annotated[str, Path(regex=UUID_REGEX)],
    email: UserEmailDep,
    db_session: DBSessionDep,
    chat_service: ChatServiceDep,
    vault_key: VaultKeyDep,
):
    is_vault_mode = bool(vault_key)

    try:
        deleted_chat = await chat_service.delete_chat(
            db_session, email, is_vault_mode, chat_id
        )
        return {"chat_id": deleted_chat.chat_id}
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="Chat not found")
    except NotAuthorizedError:
        raise HTTPException(
            status_code=403, detail="Not authorized to delete this chat"
        )
    except VaultAccessError:
        raise HTTPException(
            status_code=401, detail="Deleting vault chats requires vault mode enabled"
        )
