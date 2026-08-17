from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import Optional
import uuid
import asyncio
import logging
from functools import partial

from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import BaseMessage

from app.config import get_settings
from app.db import get_db_session_ctxmgr
from app.core.utils import extract_lc_msg_content
from app.agent.service import AgentService
from app.chat.service import ChatService
from app.chat.schemas import ChatStatusEnum, ChatUpdateRequest
from app.attachment.service import AttachmentService
from .models import MessageMD
from .schemas import (
    InputUserMessage,
    MessageSchema,
    MessageMetadata,
    MessageRoleEnum,
    MessageVersion,
    MessageVersionsResp,
    AttachmentSchma,
    MessageConfig,
)
from .repository import MessageMDRepository
from .utils import (
    sse_transformer_for_langgraph_astream,
    get_last_checkpoint_of_branch,
    get_chat_agent_type,
    get_chat_agent_param,
)
from .exceptions import (
    NotUserMessageError,
    VersionLimitExceededError,
    HasAttachmentsError,
    VersionNotFoundError,
    ChatReadOnlyError,
)

settings = get_settings()
logger = logging.getLogger(__name__)


class MessageService:

    def __init__(
        self,
        chat_service: ChatService,
        attachment_service: AttachmentService,
        message_repo: Optional[MessageMDRepository] = None,
    ):
        self.chat_service = chat_service
        self.attachment_service = attachment_service
        self.message_repo = message_repo or MessageMDRepository()

    async def _format_chat_history(
        self,
        db: AsyncSession,
        target_checkpoint_id: str,
        message_objs: list[MessageMD],
        checkpoint_messages: list[BaseMessage],
        email: str,
        vault_key: str = "",
    ) -> dict[str, MessageSchema]:
        """
        Converts LangGraph checkpoint messages and your Message model instances
        into a dictionary of MessageSchema objects for API response, reflecting the
        updated Message model (checkpoint_id as PK, no version_id).

        Args:
            target_checkpoint_id: The checkpoint ID of the last message in the current branch.
            message_objs: A list of all Message model instances associated with the
                        current chat_id, fetched from the database.
            checkpoint_messages: The full message history (list of BaseMessage) from the
                                LangGraph checkpoint state that corresponds to target_checkpoint_id.

        Returns:
            A dictionary of MessageSchema objects representing the chat history for the current branch.
        """
        if not message_objs:
            return {}

        message_map_by_cpid: dict[str, MessageMD] = {
            m.checkpoint_id: m for m in message_objs if m.checkpoint_id
        }

        last_db_message = message_map_by_cpid.get(target_checkpoint_id)
        if not last_db_message:
            if checkpoint_messages:
                logger.warning(
                    f"No database Message record found for target_checkpoint_id '{target_checkpoint_id}', "
                    f"but checkpoint_messages list is not empty. This indicates a potential data inconsistency."
                )
            return {}

        # Reconstruct the message branch
        branch_db_messages_reversed: list[MessageMD] = []
        current_db_message: Optional[MessageMD] = last_db_message
        while current_db_message:
            branch_db_messages_reversed.append(current_db_message)
            if not current_db_message.parent_checkpoint_id:
                break

            parent_cpid = current_db_message.parent_checkpoint_id
            current_db_message = message_map_by_cpid.get(parent_cpid)

            if not current_db_message:
                logger.warning(
                    f"Parent message with checkpoint_id '{parent_cpid}' not found in provided "
                    f"message_objs. Broken parent chain."
                )
                break

        branch_db_messages = list(reversed(branch_db_messages_reversed))

        if len(branch_db_messages) != len(checkpoint_messages):
            logger.error(
                f"Mismatch in message count for checkpoint '{target_checkpoint_id}'. "
                f"Database branch reconstruction yielded {len(branch_db_messages)} messages, "
                f"while LangGraph checkpoint_messages has {len(checkpoint_messages)}. "
                f"Cannot reliably format chat history."
            )
            return {}

        version_counts = defaultdict(int)
        # Group messages by logical message_id to later sort them for version_index
        logical_message_versions_map = defaultdict(list)
        for m_obj in message_objs:
            version_counts[m_obj.message_id] += 1
            logical_message_versions_map[m_obj.message_id].append(m_obj)

        # Sort versions within each logical message group by checkpoint_id
        for mid in logical_message_versions_map.keys():
            logical_message_versions_map[mid].sort(key=lambda m: m.checkpoint_id)

        # Extract content and metadata from langchain messages
        response_messages: dict[str, MessageSchema] = {}
        for i in range(len(branch_db_messages)):
            db_message_meta = branch_db_messages[i]
            lg_message = checkpoint_messages[i]

            created_at = ""
            attachment_ids = []
            attachments = []
            content = extract_lc_msg_content(lg_message)
            if (
                lg_message.additional_kwargs
                and "created_at" in lg_message.additional_kwargs
            ):
                created_at = str(lg_message.additional_kwargs["created_at"])
            if (
                lg_message.additional_kwargs
                and "attachment_ids" in lg_message.additional_kwargs
            ):
                attachment_ids = lg_message.additional_kwargs["attachment_ids"]
            current_message_logical_id = db_message_meta.message_id
            version_count = version_counts[current_message_logical_id]

            version_index = -1  # Default if not found (should not happen)
            sorted_versions_for_logical_id = logical_message_versions_map.get(
                current_message_logical_id, []
            )
            for idx, version_msg in enumerate(sorted_versions_for_logical_id):
                if version_msg.checkpoint_id == db_message_meta.checkpoint_id:
                    version_index = idx
                    break

            # Get attachment details
            if attachment_ids:
                attachments_status = (
                    await self.attachment_service.get_attachments_status(
                        db, attachment_ids, email, vault_key != ""
                    )
                )
                attachments = [
                    AttachmentSchma(**v.model_dump())
                    for v in attachments_status.values()
                    if v is not None
                ]

            if version_index == -1:
                print(
                    f"Warning: Could not determine version_index for message with checkpoint_id '{db_message_meta.checkpoint_id}'."
                )

            schema_msg = MessageSchema(
                message_id=str(db_message_meta.message_id),
                role=db_message_meta.role,
                content=content,
                message_config=db_message_meta.message_config or MessageConfig(),
                created_at=created_at,
                version_count=version_count,
                version_index=version_index,
                attachments=attachments,
            )

            response_messages[schema_msg.message_id] = schema_msg

        return response_messages

    async def get_chat_history(
        self, db: AsyncSession, chat_id: str, email: str, vault_key: str = ""
    ) -> dict[str, MessageSchema]:
        # Get agent for the chat
        chat = await self.chat_service.get_chat_if_authorized(
            db, chat_id, email, vault_key != ""
        )
        agent_service = AgentService(
            agent_type=get_chat_agent_type(chat.namespace),
            attachment_service=self.attachment_service,
        )

        message_mds = await self.message_repo.get_all_by_chat_id(db, chat_id)
        if not message_mds:
            logger.info(
                f"No message metadata found for chat {chat_id}, returning empty list."
            )
            return {}

        # Use active_checkpoint if available, otherwise use latest checkpoint
        checkpoint_id = None
        if chat.active_checkpoint:
            # Get state for the active checkpoint branch
            try:
                state = await agent_service.get_checkpoint_state(
                    db, chat_id, email, str(chat.active_checkpoint), vault_key
                )
                checkpoint_id = state.config["configurable"]["checkpoint_id"]
            except Exception as e:
                logger.warning(
                    f"Failed to get state for active_checkpoint {chat.active_checkpoint}: {e}"
                )
                # Fall back to latest checkpoint if active_checkpoint is invalid
                checkpoint_id = None

        if checkpoint_id is None:
            # Fall back to latest checkpoint
            state = await agent_service.get_latest_checkpoint(
                db, chat_id, email, vault_key
            )
            checkpoint_id = state.config["configurable"]["checkpoint_id"]

        chat_history = await self._format_chat_history(
            db, checkpoint_id, message_mds, state.values["messages"], email, vault_key
        )

        return chat_history

    async def _save_message_metadata(
        self,
        agent_service: AgentService,
        human_mid: uuid.UUID,
        ai_mid: uuid.UUID,
        parent_cid: str,
        message_config: MessageConfig,
        chat_id: str,
        email: str,
        vault_key: str = "",
    ) -> None:
        """Callback function to save metadata of new input output message pairs into db"""
        async with get_db_session_ctxmgr() as db:
            last_processed_checkpts = (
                await agent_service.get_last_processed_checkpoints(chat_id, vault_key)
            )

            # Use provided message ids for first human message and last ai messages, generate uuid for the rest
            mids = [uuid.uuid4() for _ in range(len(last_processed_checkpts) - 2)]
            mids.append(ai_mid)
            mids.insert(0, human_mid)
            last_cid = parent_cid
            message_mds = []
            for mid, (role, cid) in zip(mids, last_processed_checkpts):
                message_mds.append(
                    MessageMetadata(
                        checkpoint_id=cid,
                        message_id=mid,
                        chat_id=uuid.UUID(chat_id),
                        role=role,
                        message_config=(
                            message_config if role != MessageRoleEnum.TOOL else {}
                        ),
                        parent_checkpoint_id=last_cid,
                    )
                )
                last_cid = cid

            # Save to db
            await self.message_repo.create_many(db, objs_in=message_mds)

            # Update last_message_at in chat
            await self.chat_service.update_modified_time(
                db, email, chat_id, is_vault_mode=bool(vault_key)
            )

            # Update active_checkpoint to the latest AI message checkpoint
            await self.chat_service.update_active_checkpoint(
                db, email, chat_id, last_processed_checkpts[-1][1], bool(vault_key)
            )



        # Check if we should generate a journal entry
        try:
            # Get message count from the agent service
            state = await agent_service.get_checkpoint_state(
                None, chat_id, email, last_processed_checkpts[-1][1], vault_key
            )
            checkpoint_messages = state.values.get("messages", [])
            message_count = len(checkpoint_messages) if checkpoint_messages else 0
            
            # Get the user message content (first message in the last_processed_checkpts)
            user_message_content = ""
            if checkpoint_messages and len(checkpoint_messages) > 0:
                # The first message in the new batch should be the user message
                first_msg = checkpoint_messages[0]
                if hasattr(first_msg, 'content'):
                    user_message_content = first_msg.content
            
            # Check if journal should be generated
            if message_count >= 5 and user_message_content:
                if await self._should_generate_journal(chat_id, message_count, user_message_content):
                    import asyncio
                    asyncio.create_task(self._trigger_journal_generation(chat_id, email))
        except Exception as e:
            logger.error(f"Error checking journal generation in _save_message_metadata: {e}")

    async def _set_chat_title_cb(
        self,
        agent_service: AgentService,
        chat_id: str,
        email: str,
        vault_key: str = "",
    ) -> None:
        """Callback to generate and set chat title after first message is processed"""
        title = await agent_service.generate_title(chat_id, email, vault_key)
        if title:
            async with get_db_session_ctxmgr() as db:
                await self.chat_service.update_chat(
                    db,
                    email,
                    is_vault_mode=bool(vault_key),
                    chat_id=chat_id,
                    update=ChatUpdateRequest(title=title),
                )

    async def create_new_message(
        self,
        db: AsyncSession,
        chat_id: str,
        email: str,
        input_message: InputUserMessage,
        vault_key: str = "",
    ) -> AsyncGenerator[str, None]:
        # Get agent for the chat
        chat = await self.chat_service.get_chat_if_authorized(
            db, chat_id, email, vault_key != ""
        )
        agent_service = AgentService(
            agent_type=get_chat_agent_type(chat.namespace),
            attachment_service=self.attachment_service,
        )
        agent_params = get_chat_agent_param(chat.namespace)

        if chat.status == ChatStatusEnum.readonly:
            raise ChatReadOnlyError("The chat is readonly and cannot be modified.")

        # Get parent state checkpoint id first
        # If active_checkpoint is set, use that branch as the parent
        if chat.active_checkpoint:
            try:
                parent_state = await agent_service.get_checkpoint_state(
                    db, chat_id, email, str(chat.active_checkpoint), vault_key
                )
                parent_cid = parent_state.config["configurable"].get(
                    "checkpoint_id", ""
                )
            except Exception as e:
                logger.warning(
                    f"Failed to get state for active_checkpoint {chat.active_checkpoint}: {e}"
                )
                # Fall back to latest checkpoint if active_checkpoint is invalid
                parent_state = await agent_service.get_latest_checkpoint(
                    db, chat_id, email, vault_key
                )
                parent_cid = parent_state.config["configurable"].get(
                    "checkpoint_id", ""
                )
        else:
            parent_state = await agent_service.get_latest_checkpoint(
                db, chat_id, email, vault_key
            )
            parent_cid = parent_state.config["configurable"].get("checkpoint_id", "")

        # Get model config
        message_config = None
        if input_message.message_config:
            message_config = input_message.message_config.model_dump()

        # Get graph output stream and convert it for response
        resp_stream_coro = agent_service.process_new_user_message(
            email,
            chat_id,
            input_message.content,
            [att.attachment_id for att in input_message.attachments],
            checkpoint_id=parent_cid,
            model_config=message_config,
            agent_params=agent_params,
            vault_key=vault_key,
            nightly_rca_visual_context=(
                input_message.nightly_rca_visual_context.model_dump()
                if input_message.nightly_rca_visual_context
                else None
            ),
        )

        cbs = []
        cb_kwargs = []

        if parent_cid == "":
            # First message of a new chat, generate title
            cbs.append(self._set_chat_title_cb)
            cb_kwargs.append(
                {
                    "agent_service": agent_service,
                    "chat_id": chat_id,
                    "email": email,
                    "vault_key": vault_key,
                }
            )

        error_cb = partial(self._handle_stream_err, chat_id, email, vault_key=vault_key)

        return sse_transformer_for_langgraph_astream(
            resp_stream_coro,
            error_cb,
            self._save_message_metadata,
            {
                "agent_service": agent_service,
                "parent_cid": parent_cid,
                "message_config": message_config,
                "chat_id": chat_id,
                "email": email,
                "vault_key": vault_key,
            },
            on_stream_end_callbacks=cbs,
            callback_kwargs=cb_kwargs,
        )

    async def get_message_versions(
        self,
        db: AsyncSession,
        chat_id: str,
        email: str,
        message_id: str,
        vault_key: str = "",
    ):
        # Get agent for the chat
        chat = await self.chat_service.get_chat_if_authorized(
            db, chat_id, email, vault_key != ""
        )
        agent_service = AgentService(
            agent_type=get_chat_agent_type(chat.namespace),
            attachment_service=self.attachment_service,
        )

        # Get metadata of message versions, sort by creation time
        message_mds = await self.message_repo.get_all_by_msg_id(db, message_id)
        message_mds.sort(key=lambda m: m.checkpoint_id)
        tasks_get_states = [
            agent_service.get_checkpoint_state(
                db, chat_id, email, metadata.checkpoint_id, vault_key
            )
            for metadata in message_mds
        ]
        states = await asyncio.gather(*tasks_get_states)

        message_versions = list()
        for idx, state in enumerate(states):
            message = state.values["messages"][-1]
            message_versions.append(
                MessageVersion(
                    version_index=idx,
                    content=extract_lc_msg_content(message),
                    created_at=message.additional_kwargs.get("created_at", ""),
                )
            )

        return MessageVersionsResp(
            message_id=message_id,
            versions=message_versions,
            total_versions=len(message_versions),
        )

    async def _handle_stream_err(
        self, chat_id: str, email: str, exc: Exception, vault_key: str = ""
    ) -> str:
        """On streaming error, handle the exception and set corresponding msg"""

        # TODO Handle exceptions differently. Generate more user friendly message
        async with get_db_session_ctxmgr() as db:
            await self.chat_service.set_readonly(
                db, email, chat_id, bool(vault_key), str(exc)
            )
        return str(exc)

    async def branch_past_message(
        self,
        db: AsyncSession,
        chat_id: str,
        message_id: str,
        email: str,
        new_message: str,
        vault_key: str = "",
    ) -> AsyncGenerator[str, None]:
        # Get agent for the chat
        chat = await self.chat_service.get_chat_if_authorized(
            db, chat_id, email, vault_key != ""
        )
        agent_service = AgentService(
            agent_type=get_chat_agent_type(chat.namespace),
            attachment_service=self.attachment_service,
        )

        if chat.status == ChatStatusEnum.readonly:
            raise ChatReadOnlyError()

        # Get latest version and branch from it
        message_mds = await self.message_repo.get_all_by_msg_id(db, message_id)
        if len(message_mds) >= settings.MAX_VERSION_COUNT:
            raise VersionLimitExceededError(
                f"Version limit exceeded. You can have up to {settings.MAX_VERSION_COUNT} versions per message."
            )
        latest_version = max(message_mds, key=lambda m: m.checkpoint_id)
        if latest_version.role != MessageRoleEnum.USER:
            raise NotUserMessageError(
                f"Attempting to modify a non-user message. Type: {latest_version.role}"
            )
        # Check if the message has attachments
        state = await agent_service.get_checkpoint_state(
            db, chat_id, email, latest_version.checkpoint_id, vault_key
        )
        if state.values["messages"][-1].additional_kwargs.get("attachment_ids"):
            raise HasAttachmentsError(
                "Attempting to modify a message that has attachments"
            )

        # Get graph output stream and convert it for response
        parent_cid = latest_version.parent_checkpoint_id
        message_config = latest_version.message_config or MessageConfig().model_dump()
        resp_stream_coro = agent_service.branch_from_past_user_message(
            email,
            chat_id,
            latest_version.checkpoint_id,
            new_message,
            message_config,
            vault_key,
        )

        cbs = []
        cb_kwargs = []
        error_cb = partial(self._handle_stream_err, chat_id, email, vault_key=vault_key)

        return sse_transformer_for_langgraph_astream(
            resp_stream_coro,
            error_cb,
            self._save_message_metadata,
            {
                "agent_service": agent_service,
                "parent_cid": parent_cid,
                "message_config": message_config,
                "chat_id": chat_id,
                "email": email,
                "vault_key": vault_key,
            },
            input_message_id=uuid.UUID(message_id),
            input_version_index=len(message_mds),
            on_stream_end_callbacks=cbs,
            callback_kwargs=cb_kwargs,
        )


    async def _should_generate_journal(
        self,
        chat_id: str,
        message_count: int,
        last_message_content
    ) -> bool:
        """
        Determine if we should generate a journal entry for this conversation.
        
        Triggers:
        - Explicit goodbye/farewell detected
        - Chat has at least 5 messages
        - Long pause (handled externally)
        
        Args:
            chat_id: The chat ID
            message_count: Number of messages in the chat
            last_message_content: Content of the last user message (str or list)
            
        Returns:
            True if journal should be generated
        """
        # Don't generate for very short conversations
        if message_count < 5:
            return False
        
        # Handle content that might be a list or string
        if isinstance(last_message_content, list):
            # If it's a list, join all text content
            content_str = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in last_message_content
            )
        else:
            content_str = str(last_message_content) if last_message_content else ""
        
        # Detect explicit farewells
        farewell_phrases = [
            "goodbye", "bye", "see you", "talk to you later", 
            "thanks for your help", "that's all", "i'm done",
            "thank you", "ttyl", "gotta go", "have a good"
        ]
        
        content_lower = content_str.lower()
        if any(phrase in content_lower for phrase in farewell_phrases):
            logger.info(f"Detected farewell in chat {chat_id}, will generate journal")
            return True
        
        return False

    async def _trigger_journal_generation(
        self,
        chat_id: str,
        owner_email: str
    ) -> None:
        """
        Trigger background journal generation for a chat.
        
        Args:
            chat_id: The chat ID
            owner_email: The owner's email
        """
        from app.journal.service import generate_journal_background_task
        from app.background_mgr.service import get_task_manager
        
        try:
            task_manager = get_task_manager()
            task_id = str(uuid.uuid4())  # Generate a proper UUID
            
            # Use add_task instead of submit_task
            task = task_manager.add_task(
                generate_journal_background_task,
                task_id,
                chat_id=chat_id,
                owner_email=owner_email,
                task_type="journal_generation"
            )
            logger.info(f"Triggered journal generation for chat {chat_id}, task_id: {task_id}")
        except Exception as e:
            logger.error(f"Failed to trigger journal generation for chat {chat_id}: {e}")

    async def get_branch_history(
        self, db, chat_id, message_id, version_id, email, vault_key
    ) -> dict[str, MessageSchema]:
        # Get agent for the chat
        chat = await self.chat_service.get_chat_if_authorized(
            db, chat_id, email, vault_key != ""
        )
        agent_service = AgentService(
            agent_type=get_chat_agent_type(chat.namespace),
            attachment_service=self.attachment_service,
        )

        # Get metadata of message versions, sort by creation time
        message_mds = await self.message_repo.get_all_by_msg_id(db, message_id)
        message_mds.sort(key=lambda m: m.checkpoint_id)
        if version_id > len(message_mds) or version_id < 0:
            raise VersionNotFoundError(
                f"Version id {version_id} does not exist for message {message_id}"
            )
        target_metadata = message_mds[version_id]

        # Find latest leaf checkpoint id for this message version
        all_message_mds = await self.message_repo.get_all_by_chat_id(db, chat_id)
        leaf_cid = get_last_checkpoint_of_branch(
            all_message_mds, target_metadata.checkpoint_id
        )
        if not leaf_cid:
            logger.warning(
                f"No leaf checkpoint found for message {message_id} version {version_id}."
            )
            return []

        # Set this branch leaf as active checkpoint for the chat.
        await self.chat_service.update_active_checkpoint(
            db, email, chat_id, leaf_cid, bool(vault_key)
        )

        # Get message from the latest leaf checkpoint
        state = await agent_service.get_checkpoint_state(
            db, chat_id, email, leaf_cid, vault_key
        )
        branch_history = await self._format_chat_history(
            db, leaf_cid, all_message_mds, state.values["messages"], email, vault_key
        )

        # Return the full message history
        return branch_history
