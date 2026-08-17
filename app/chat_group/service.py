from typing import Optional, List
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings

from .models import ChatGroup
from .schemas import ChatGroupCreate, ChatGroupResponse, ChatGroupUpdate
from .repository import ChatGroupRepository
from .exceptions import (
    ChatGroupLimitExceededError,
    ChatGroupNotAuthorizedError,
    ChatGroupNotFoundError,
)

setting = get_settings()
logger = logging.getLogger(__name__)

def get_group_service():
    return ChatGroupService()
    
class ChatGroupService:
    def __init__(self, chat_group: Optional[ChatGroupRepository] = None):
        self.chat_group_repo = chat_group or ChatGroupRepository()
        
    async def count_groups_by_owner(self, db: AsyncSession, email: str, search: str = "") -> int:
        """Count number of chats owned by a specific user"""
        return await self.chat_group_repo.count_by_owner(db, email, search=search)
    
    async def list_groups_for_user(
        self,
        db: AsyncSession,
        email: str,
        *,
        offset: int = 0,
        limit: int = 0,
        search: str = "",
    ) -> List[ChatGroupResponse]:
        chats = await self.chat_group_repo.list_by_owner(db, email, offset=offset, limit=limit, search=search)
        return [  
            ChatGroupResponse(
                group_id=str(chat.group_id),
                title=chat.title,
                owner_id=chat.owner_id
            )
            for chat in chats
        ]
    
    async def create_group_for_user(
        self,
        db: AsyncSession,
        email: str,
        data: ChatGroupCreate
    ) -> ChatGroup:
        """Create a new group for an user"""
        # Enforce groups count limit
        cur_groups_count = await self.chat_group_repo.count_by_owner(db, email)
        if cur_groups_count >= setting.MAX_GROUPS_WITH_CHATS_COUNT:
            raise ChatGroupLimitExceededError(
                f"Group limit reached. You can have up to {setting.MAX_GROUPS_WITH_CHATS_COUNT} groups."
            )

        # Create new group record
        group_create = ChatGroupCreate(title=data.title, owner_id=email)
        new_group = await self.chat_group_repo.create_one(db, obj_in=group_create)

        return new_group
    
    async def check_group(
        self,
        db: AsyncSession,
        group_id: str,
        user_email: str,
    ) -> ChatGroup:
        """Get a group if the user is authorized to access it"""
        group = await self.chat_group_repo.get_one_by_id(db, group_id)

        if not group:
            raise ChatGroupNotFoundError(f"Group with ID {group_id} not found")

        if group.owner_id != user_email:
            raise ChatGroupNotAuthorizedError("Not authorized to access this group")


        return group

    async def update_group(
        self,
        db: AsyncSession,
        email: str,
        group_id: str,
        data: ChatGroupUpdate
    ) -> ChatGroup:
        """Update attributes of a group record."""
        # This will raise appropriate exceptions if not authorized
        group = await self.check_group(db, group_id, email)
        # Update chat
        update_data = {}
        if data.title:
            update_data["title"] = data.title 

        updated_group = await self.chat_group_repo.update(db, db_obj=group, obj_in=update_data)

        return updated_group

    async def delete_group(
        self,
        db: AsyncSession,
        email: str,
        group_id: str
    ) -> ChatGroup:
        """Delete a chat record."""
        # This will raise appropriate exceptions if not authorized
        await self.check_group(db, group_id, email)

        deleted_group = await self.chat_group_repo.remove_by_id(db, id=group_id)
        logger.info(f"Group {group_id} deleted by user {email}")

        return deleted_group