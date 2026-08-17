from typing import Optional, Any
from collections.abc import AsyncIterator
import logging

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage
from langgraph.types import StateSnapshot
from langgraph.graph.message import RemoveMessage
from sqlalchemy.ext.asyncio import AsyncSession
from app.analytics.service import AnalyticsService
from app.config import get_settings
from app.core.utils import get_timestr_now_utc
from app.db import get_db_session_ctxmgr
from app.attachment.service import get_attachment_service, AttachmentService
from app.agent.agents.registry import get_agent_graph
from app.agent.utils import (
    text_content,
    image_content,
    doc_content,
    stringfy_messages,
)
from app.agent.nightly_rca_visual_context import build_nightly_rca_visual_content
from app.message.compression import (
    trim_message_history,
    should_compress_history,
    get_compression_stats
)


logger = logging.getLogger(__name__)
settings = get_settings()


class AgentService:
    def __init__(
        self,
        agent_type: str = "chat",
        attachment_service: AttachmentService = None,
    ):
        self.graph = get_agent_graph(agent_type)
        self.attmnt_service = attachment_service or get_attachment_service()

    def _compress_message_history_if_needed(
        self, 
        messages: list[BaseMessage],
        chat_id: str = ""
    ) -> list[BaseMessage]:
        """
        Compress message history if needed to prevent token overflow.
        
        Args:
            messages: List of messages to potentially compress
            chat_id: Chat ID for logging
            
        Returns:
            Possibly compressed list of messages
        """
        if not messages:
            return messages
        
        # Check if compression is needed
        if should_compress_history(messages):
            stats = get_compression_stats(messages)
            logger.info(
                f"Chat {chat_id}: Message history at {stats['context_usage_pct']:.1f}% capacity "
                f"({stats['total_tokens']} tokens, {stats['total_messages']} messages)"
            )
            
            # Trim to 70% of context to leave room for response
            max_tokens = int(settings.ELLM_CTX_LEN * 0.7)
            compressed = trim_message_history(
                messages,
                max_tokens=max_tokens,
                strategy="last",
                keep_system=True
            )
            
            new_stats = get_compression_stats(compressed)
            logger.info(
                f"Chat {chat_id}: Compressed to {new_stats['total_messages']} messages "
                f"({new_stats['total_tokens']} tokens, {new_stats['context_usage_pct']:.1f}%)"
            )
            
            return compressed
        
        return messages

    async def get_latest_checkpoint(
        self, db: AsyncSession, chat_id: str, email: str, vault_key: str = ""
    ) -> StateSnapshot:

        config = {
            "configurable": {
                "thread_id": chat_id,
                "encryption_key": vault_key or settings.MASTER_KEY,
            }, "recursion_limit": settings.LANGGRAPH_RECURSION_LIMIT
        }
        state = await self.graph.aget_state(config=config)
        return state

    async def get_checkpoint_state(
        self,
        db: AsyncSession,
        chat_id: str,
        email: str,
        checkpoint_id: str,
        vault_key: str = "",
    ) -> StateSnapshot:
        config = {
            "configurable": {
                "checkpoint_id": checkpoint_id,
                "thread_id": chat_id,
                "encryption_key": vault_key or settings.MASTER_KEY,
            }, "recursion_limit": settings.LANGGRAPH_RECURSION_LIMIT
        }
        state = await self.graph.aget_state(config=config)
        return state

    async def get_last_processed_checkpoints(
        self, chat_id: str, vault_key: str = ""
    ) -> list[tuple[str, str]]:
        """Return the checkpoints corresponding to newly added messages in the last conversation turn.
        i.e. all messages since the last human message.
        """
        config = {
            "configurable": {
                "thread_id": chat_id,
                "encryption_key": vault_key or settings.MASTER_KEY,
            }, "recursion_limit": settings.LANGGRAPH_RECURSION_LIMIT
        }
        checkpoint_ids = []
        async for state in self.graph.aget_state_history(config=config):
            # Human message marks the start of a conversation turn
            if isinstance(state.values["messages"][-1], HumanMessage):
                checkpoint_ids.append(
                    ("user", state.config["configurable"]["checkpoint_id"])
                )
                break

            # AIMessages duplicate on graph end, so exclude it
            if isinstance(state.values["messages"][-1], AIMessage) and state.next != (
                "__start__",
            ):
                checkpoint_ids.append(
                    ("assistant", state.config["configurable"]["checkpoint_id"])
                )

            if isinstance(state.values["messages"][-1], ToolMessage):
                checkpoint_ids.append(
                    ("tool", state.config["configurable"]["checkpoint_id"])
                )

        checkpoint_ids.reverse()
        return checkpoint_ids

    async def process_new_user_message(
        self,
        email: str,
        chat_id: str,
        message: str,
        attachment_ids: list[str],
        checkpoint_id: str = "",
        model_config: dict[str, Any] = {},
        agent_params: dict[str, str] = None,
        vault_key: str = "",
        nightly_rca_visual_context: dict[str, Any] | None = None,
    ) -> AsyncIterator:
        """Add a new user message to a chat with automatic history compression"""
        async with get_db_session_ctxmgr() as db:
            config = {
                "configurable": {
                    "thread_id": chat_id,
                    "encryption_key": vault_key or settings.MASTER_KEY,
                }, "recursion_limit": settings.LANGGRAPH_RECURSION_LIMIT
            }
            if checkpoint_id:
                config["configurable"]["checkpoint_id"] = checkpoint_id

            # Init message obj
            input_msg = HumanMessage(content=[])
            input_msg.additional_kwargs["created_at"] = get_timestr_now_utc()

            # Place trusted server-generated visual context before ordinary
            # user attachments, matching the D3F/D3H qualified model ordering.
            contents = []
            if nightly_rca_visual_context is not None:
                nightly_blocks, nightly_provenance = (
                    await build_nightly_rca_visual_content(
                        nightly_rca_visual_context,
                        request_id=f"{chat_id}:{checkpoint_id or 'root'}",
                        requester_email=email,
                        max_resolution=settings.MAX_IMAGE_RES,
                    )
                )
                contents.extend(nightly_blocks)
                input_msg.additional_kwargs["nightly_rca_visual_context"] = (
                    nightly_provenance
                )

            if attachment_ids:
                input_msg.additional_kwargs["attachment_ids"] = attachment_ids
                attachments = await self.attmnt_service.download_attachment_internal(
                    db, attachment_ids, email, vault_key
                )
                # Check for all existence
                if notready := [aid for aid, a in attachments.items() if a is None]:
                    raise ValueError(
                        f"Following attachments can't be retrieved: {notready}"
                    )

                for aid, (status, obj) in attachments.items():
                    if status.media_type in settings.SUPPORTED_IMAGE_TYPES:
                        # Image is sent base64 encoded
                        contents.extend(
                            image_content(status.filename, status.media_type, obj)
                        )

                    elif status.media_type in settings.SUPPORTED_DOC_TYPES:
                        # Document is put directly in context if small, else indexed for RAG
                        contents.append(
                            doc_content(obj, obj.est_size < settings.MAX_IN_CTX_DOC_LEN)
                        )

            # Add human message
            contents.append(text_content(message))
            input_msg.content = contents
            
            # Get existing state to check message history
            try:
                state = await self.graph.aget_state(config=config)
                existing_messages = state.values.get("messages", [])
                
                # Compress message history if needed BEFORE adding new message
                if existing_messages:
                    compressed_messages = self._compress_message_history_if_needed(
                        existing_messages,
                        chat_id=chat_id
                    )
                    
                    # If compression occurred, update the state
                    if len(compressed_messages) < len(existing_messages):
                        logger.info(
                            f"Chat {chat_id}: Updating state with compressed history "
                            f"({len(existing_messages)} -> {len(compressed_messages)} messages)"
                        )
                        # Remove old messages and add compressed ones
                        messages_to_remove = [
                            RemoveMessage(id=msg.id) 
                            for msg in existing_messages
                        ]
                        await self.graph.aupdate_state(
                            config,
                            {"messages": messages_to_remove + compressed_messages}
                        )
            except Exception as e:
                logger.warning(f"Could not compress message history: {e}")
                # Continue anyway - better to have a long context than crash

        return self.graph.astream(
            input={
                "messages": [input_msg],
                "model_config": model_config,
                "agent_params": agent_params,
            },
            config=config,
            stream_mode="messages",
        )

    async def branch_from_past_user_message(
        self,
        email: str,
        chat_id: str,
        checkpoint_id: str,
        message: str,
        model_config: dict[str, Any] = {},
        vault_key: str = "",
    ) -> AsyncIterator:
        """Branch from a specific checkpoint. No attachment allowed."""
        enc_key = vault_key or settings.MASTER_KEY

        input_msg = HumanMessage(content=message)
        input_msg.additional_kwargs["created_at"] = get_timestr_now_utc()

        # Update the checkpoint state to branch for the new message
        old_config = {
            "configurable": {
                "checkpoint_id": checkpoint_id,
                "thread_id": chat_id,
                "encryption_key": enc_key,
            }, "recursion_limit": settings.LANGGRAPH_RECURSION_LIMIT
        }
        old_state = await self.graph.aget_state(config=old_config)

        # set checkpoint_ns to "" if not exists to avoid key error in update_state
        old_config = old_state.config
        if "checkpoint_ns" not in old_config["configurable"]:
            old_config["configurable"]["checkpoint_ns"] = ""

        old_config["configurable"]["encryption_key"] = enc_key
        msgid_to_replace = old_state.values["messages"][-1].id
        
        # Check if compression needed before branching
        existing_messages = old_state.values.get("messages", [])
        compressed_messages = self._compress_message_history_if_needed(
            existing_messages,
            chat_id=chat_id
        )
        
        # Update with compressed history if needed
        messages_to_update = [RemoveMessage(msgid_to_replace), input_msg]
        if len(compressed_messages) < len(existing_messages):
            logger.info(f"Chat {chat_id}: Compressing branch history")
            messages_to_remove = [
                RemoveMessage(id=msg.id) 
                for msg in existing_messages[:-1]  # Keep last message for replacement
            ]
            messages_to_update = messages_to_remove + compressed_messages[:-1] + [
                RemoveMessage(msgid_to_replace), 
                input_msg
            ]
        
        new_config = await self.graph.aupdate_state(
            old_config,
            {
                "messages": messages_to_update,
                "model_config": model_config,
            },
        )
        new_config["configurable"]["encryption_key"] = enc_key

        new_config["recursion_limit"] = settings.LANGGRAPH_RECURSION_LIMIT

        return self.graph.astream(input=None, config=new_config, stream_mode="messages")

    async def generate_title(
        self, chat_id: str, email: str, vault_key: str = ""
    ) -> str | None:
        """Generate title for a chat"""
        # Needs to get db session for itself as it may be called after a request ends.
        async with get_db_session_ctxmgr() as db:
            state = await self.get_latest_checkpoint(db, chat_id, email, vault_key)

            title_gen_agent = get_agent_graph("title_gen")
            messages: list[BaseMessage] = state.values["messages"]
            message_history = stringfy_messages(messages)
            # Truncate conversation before sending to title model.
            # Title generation only needs enough context to understand the topic;
            # it does NOT need full tool responses or log blobs.
            # Sending the full history on heavy investigation sessions (200k+ tokens)
            # causes a hard Bedrock ValidationException: prompt is too long.
            if len(message_history) > settings.TITLE_MAX_CONTEXT_CHARS:
                message_history = (
                    message_history[: settings.TITLE_MAX_CONTEXT_CHARS]
                    + "\n[... conversation truncated for title generation ...]"
                )
                logger.debug(
                    "generate_title: truncated message_history to %d chars for chat %s",
                    settings.TITLE_MAX_CONTEXT_CHARS,
                    chat_id,
                )
            resp = await title_gen_agent.ainvoke(
                input={
                    "messages": [
                        HumanMessage(
                            content=f"<conversation>\n{message_history}\n</conversation>"
                        ),
                    ]
                }
            )

            title = resp.get("title", None)
            return title

    async def complete_turn_and_summarize(
        self,
        chat_id: str,
        email: str,
        vault_key: str = "",
    ) -> None:
        async with get_db_session_ctxmgr() as db:
            analytics = AnalyticsService()
            await analytics.summarize_chat(db, chat_id=chat_id, owner_email=email)

    async def delete_chat(
        self,
        chat_id: str,
    ) -> None:
        # Delete the chat from checkpointer
        # Requires langgraph >= 0.4.2
        await self.graph.checkpointer.adelete_thread(thread_id=chat_id)
