from typing import Annotated
from fastapi import APIRouter, Query, Depends, HTTPException, Path
from app.dependencies import DBSessionDep, UserEmailDep
from app.core.utils import UUID_REGEX

from .schemas import (
    ChatGroupResponse,
    ChatGroupUpdate,
    ListChatGroupsResponse,
)

from .service import (
    ChatGroupService,
    get_group_service
)

from .exceptions import (
    ChatGroupLimitExceededError,
    ChatGroupNotAuthorizedError,
    ChatGroupNotFoundError,
)

ChatGroupServiceDep = Annotated[ChatGroupService, Depends(get_group_service)]

router = APIRouter()

@router.post("/chats-groups", response_model=ChatGroupResponse)
async def create_group(
    email: UserEmailDep,
    db_session: DBSessionDep,
    group_service: ChatGroupServiceDep,
    data: ChatGroupUpdate,
) -> ChatGroupResponse:
    try:
        new_group = await group_service.create_group_for_user(db_session, email=email, data=data)
        return ChatGroupResponse(  
            group_id=str(new_group.group_id),  
            title=new_group.title,  
            owner_id=new_group.owner_id  
        ) 
    except ChatGroupLimitExceededError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": "GROUP_LIMIT_EXCEEDED", "message": str(e)}
        )

@router.get("/chats-groups")
async def list_groups(
    email: UserEmailDep,
    db_session: DBSessionDep,
    group_service: ChatGroupServiceDep,
    page: int = Query(ge=1, default=1),
    limit: int = Query(ge=1, le=100, default=50),
    search: str = Query(max_length=40, default=""),
) -> ListChatGroupsResponse:
    """List all groups owned by the user. Supports pagination and simple title search"""
    offset = (page - 1) * limit
    end_idx = offset + limit    # Index of end past one
    user_groups = await group_service.list_groups_for_user(
        db_session,
        email,
        offset=offset,
        limit=limit,
        search=search
    )
    total_docs = await group_service.count_groups_by_owner(db_session, email, search=search)

    return ListChatGroupsResponse(
        docs=user_groups,
        totalDocs=total_docs,
        limit=limit,
        page=page,
        totalPages=(total_docs + limit - 1) // limit,
        hasNextPage=end_idx < total_docs,
        nextPage=page + 1 if end_idx < total_docs else None,
        hasPrevPage=page > 1,
        prevPage=page - 1 if page > 1 else None,
    )

@router.put("/chats-groups/{group_id}", response_model=ChatGroupResponse)
async def update_group(
    group_id: Annotated[str, Path(regex=UUID_REGEX)],
    data: ChatGroupUpdate,
    email: UserEmailDep,
    db_session: DBSessionDep,
    group_service: ChatGroupServiceDep
) -> ChatGroupResponse:
    try:
        updated_group = await group_service.update_group(db_session, email=email, group_id=group_id, data=data)

        return ChatGroupResponse(
            group_id=str(updated_group.group_id),
            title=updated_group.title,
            owner_id=updated_group.owner_id
        )
    except ChatGroupNotFoundError:
        raise HTTPException(status_code=404, detail="Group not found")
    except ChatGroupNotAuthorizedError:
        raise HTTPException(status_code=403, detail="Not authorized to modify this group")

@router.delete("/chats-groups/{group_id}", tags=["chat_group"])
async def delete_chat(
    group_id: Annotated[str, Path(regex=UUID_REGEX)],
    email: UserEmailDep,
    db_session: DBSessionDep,
    group_service: ChatGroupServiceDep,
):
    try:
        deleted_group = await group_service.delete_group(db_session, email, group_id=group_id)
        return {"group_id": deleted_group.group_id}
    except ChatGroupNotFoundError:
        raise HTTPException(status_code=404, detail="Group not found")
    except ChatGroupNotAuthorizedError:
        raise HTTPException(status_code=403, detail="Not authorized to delete this group")