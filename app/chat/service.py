# /chat/service.py

from typing import Optional, List
import logging
from datetime import datetime, UTC

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.user.service import set_user_current_chat
from app.core.constants import CHAT_NS_MAP
from app.agent.service import AgentService
from .models import Chat
from .schemas import ChatCreate, ChatUpdateRequest, ChatStatusEnum
from .repository import ChatRepository
from .exceptions import (
    ChatLimitExceededError,
    ChatNotFoundError,
    NotAuthorizedError,
    VaultAccessError,
    NamespaceNotFoundError,
)

setting = get_settings()
logger = logging.getLogger(__name__)


def get_chat_service():
    return ChatService()


class ChatService:
    def __init__(self, chat_repo: Optional[ChatRepository] = None):
        self.chat_repo = chat_repo or ChatRepository()

    async def count_by_owner(
        self,
        db: AsyncSession,
        email: str,
        namespace: str,
        search: str = "",
        group_id: str = "all",
    ) -> int:
        """Count number of chats owned by a specific user"""
        return await self.chat_repo.count_by_owner(
            db, email, namespace, search=search, group_id=group_id
        )

    async def create_chat_for_user(
        self,
        db: AsyncSession,
        email: str,
        namespace: str,
        vault: bool,
        group_id: Optional[str] = None,
    ) -> Chat:
        """Create a new chat for an user"""
        # Check ns validity
        if namespace not in CHAT_NS_MAP:
            raise NamespaceNotFoundError(
                f"Requested namespace {namespace} does not exist or is not exact."
            )

        # Enforce chat count limit
        cur_chat_count = await self.chat_repo.count_by_owner(db, email, namespace)
        if cur_chat_count >= setting.MAX_CHAT_COUNT:
            raise ChatLimitExceededError(
                f"Chat session limit reached. You can have up to {setting.MAX_CHAT_COUNT} chats."
            )
        if CHAT_NS_MAP[namespace].is_singleton and cur_chat_count > 0:
            raise ChatLimitExceededError(
                f"Requested namespace {namespace} only allow a single chat to exist."
            )

        # Create new chat record
        title = f"New chat {cur_chat_count + 1}"
        chat_create = ChatCreate(
            title=title,
            owner_id=email,
            namespace=namespace,
            vault_mode=vault,
            group_id=group_id if group_id == "all" else None,
        )
        new_chat = await self.chat_repo.create_one(db, obj_in=chat_create)

        # Set current chat to the new chat
        if namespace == "generic":
            await set_user_current_chat(db, email, new_chat.chat_id)

        return new_chat

    async def list_chat_for_user(
        self,
        db: AsyncSession,
        email: str,
        namespace: str,
        *,
        offset: int = 0,
        limit: int = 0,
        search: str = "",
        group_id: Optional[str] = "all",
    ) -> List[Chat]:
        chats = await self.chat_repo.list_by_owner(
            db,
            email,
            namespace,
            offset=offset,
            limit=limit,
            search=search,
            group_id=group_id,
        )
        return chats

    async def get_chat_if_authorized(
        self,
        db: AsyncSession,
        chat_id: str,
        user_email: str,
        is_vault_mode: bool = False,
    ) -> Chat:
        """Get a chat if the user is authorized to access it"""
        chat = await self.chat_repo.get_one_by_id(db, chat_id)

        if not chat:
            raise ChatNotFoundError(f"Chat with ID {chat_id} not found")

        if chat.owner_id != user_email:
            raise NotAuthorizedError("Not authorized to access this chat")

        if chat.vault_mode and not is_vault_mode:
            raise VaultAccessError("Accessing vault chats requires vault mode enabled")

        return chat

    async def update_chat(
        self,
        db: AsyncSession,
        email: str,
        is_vault_mode: bool,
        chat_id: str,
        update: ChatUpdateRequest,
    ) -> Chat:
        """Update attributes of a chat record."""
        # This will raise appropriate exceptions if not authorized
        chat = await self.get_chat_if_authorized(db, chat_id, email, is_vault_mode)

        # Update chat
        update_data = {}
        if update.title:
            update_data["title"] = update.title
        if update.favorite is not None:
            update_data["favorite"] = update.favorite
        if update.group_id != "all":
            update_data["group_id"] = update.group_id

        updated_chat = await self.chat_repo.update(db, db_obj=chat, obj_in=update_data)

        if update.active is not None:
            # Handle set active chat separately
            await set_user_current_chat(db, email, chat_id)

        return updated_chat

    async def delete_chat(
        self, db: AsyncSession, email: str, is_vault_mode: bool, chat_id: str
    ) -> Chat:
        """Delete a chat record."""
        # This will raise appropriate exceptions if not authorized
        chat = await self.get_chat_if_authorized(db, chat_id, email, is_vault_mode)
        agent_service = AgentService(
            agent_type=CHAT_NS_MAP.get(
                chat.namespace, CHAT_NS_MAP["generic"]
            ).chat_agent
        )

        # Delete chat in chat table
        deleted_chat = await self.chat_repo.remove_by_id(db, id=chat_id)

        # Delete chat history from langchain checkpointer
        await agent_service.delete_chat(chat_id)

        logger.info(f"Chat {chat_id} deleted by user {email}")
        # Current chat id and chat messages are automatically set to null by DB Foreign key constraint

        return deleted_chat

    async def update_modified_time(
        self, db: AsyncSession, email: str, chat_id: str, is_vault_mode: bool
    ) -> None:
        chat = await self.get_chat_if_authorized(db, chat_id, email, is_vault_mode)
        if chat:
            await self.chat_repo.update(
                db,
                db_obj=chat,
                obj_in={"last_message_at": datetime.now(UTC).replace(tzinfo=None)},
            )

    async def update_active_checkpoint(
        self,
        db: AsyncSession,
        email: str,
        chat_id: str,
        active_checkpoint: str,
        is_vault_mode: bool = False,
    ) -> None:
        """Update the active_checkpoint field of a chat."""
        chat = await self.get_chat_if_authorized(db, chat_id, email, is_vault_mode)
        if chat:
            await self.chat_repo.update(
                db, db_obj=chat, obj_in={"active_checkpoint": active_checkpoint}
            )

    async def set_readonly(
        self,
        db: AsyncSession,
        email: str,
        chat_id: str,
        is_vault_mode: bool,
        message: str,
    ) -> None:
        """Set a chat to readonly"""
        chat = await self.get_chat_if_authorized(db, chat_id, email, is_vault_mode)
        if chat:
            await self.chat_repo.update(
                db,
                db_obj=chat,
                obj_in={"status": ChatStatusEnum.readonly, "status_msg": message},
            )
