import datetime as dt
import json
import logging
import uuid
from collections.abc import Sequence
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.llm import get_model
from app.db.base import sessionmanager
from app.message.models import MessageMD
from .models import BackendInsight, ChatSummary
from app.background_mgr.service import TaskProgressTracker

logger = logging.getLogger(__name__)
settings = get_settings()


SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert conversation analyst and engineering coach. You receive full transcripts of conversations between a user and a technical assistant (a chatbot). Your job is to:
1) Summarise the conversation.
2) Evaluate conversation quality along multiple dimensions.
3) Provide feedback for both the bot and the user.
4) Suggest concrete improvements to the chatbot backend.

Return ONLY valid JSON in the following structure (no prose outside JSON):
{{
  "summary": "<short text>",
  "conversation_quality": {{
    "context_continuity": {{"score": 0.0-1.0, "notes": "..."}},
    "productivity": {{"score": 0.0-1.0, "notes": "..."}},
    "accuracy": {{"score": 0.0-1.0, "notes": "..."}},
    "psychoanalysis": {{"summary": "...", "risk_flags": ["..."]}}
  }},
  "communication_feedback": {{
    "bot_strengths": ["..."],
    "bot_weaknesses": ["..."],
    "user_suggestions": ["..."],
    "bot_suggestions": ["..."]
  }},
  "backend_enhancement_ideas": [
    {{
      "id": "<short stable id>",
      "title": "<short title>",
      "description": "<detailed description>",
      "component": "<component to improve>",
      "priority": "low|medium|high"
    }}
  ]
}}""",
        ),
        (
            "human",
            """Here is the full conversation transcript. Use it to fill in the JSON structure described above.

{conversation}""",
        ),
    ]
)


async def save_web_search(
    chat_id: Optional[str],
    query: str,
    results_json: Optional[dict] = None
) -> None:
    """
    Persist a record of a web search for analytics/UX.
    
    Args:
        chat_id: The chat session this search belongs to (optional)
        query: The search query string
        results_json: Optional JSON blob of search results
    """
    try:
        # Convert dict to JSON string for JSONB column
        results_json_str = json.dumps(results_json) if results_json else None
        
        async with sessionmanager.session() as session:
            await session.execute(
                text("""
                    INSERT INTO web_searches (id, chat_id, query, results_json, created_at)
                    VALUES (:id, :chat_id, :query, CAST(:results_json AS jsonb), :created_at)
                """),
                {
                    "id": str(uuid.uuid4()),
                    "chat_id": chat_id,
                    "query": query,
                    "results_json": results_json_str,
                    "created_at": dt.datetime.utcnow(),
                },
            )
            await session.commit()
            logger.info(f"Saved web search: {query[:50]}... for chat {chat_id}")
    except Exception as e:
        logger.error(f"Failed to save web search: {e}")
        # Don't fail the search if logging fails




async def summarise_conversation_async(
    chat_id: str,
    owner_email: str,
    bg_tracker: Optional[TaskProgressTracker] = None
) -> None:
    """
    Async wrapper for summarizing a conversation with progress tracking.
    This is called by the background task manager when a journal entry should be generated.
    
    Args:
        chat_id: The chat ID to summarize
        owner_email: The owner's email
        bg_tracker: Optional progress tracker from background task manager
    """
    from app.db import get_db_session_ctxmgr
    
    try:
        if bg_tracker:
            await bg_tracker.start(total=1, message="Starting conversation summary")
        
        logger.info(f"Starting journal generation for chat {chat_id}")
        
        async with get_db_session_ctxmgr() as db:
            analytics_service = AnalyticsService()
            summary = await analytics_service.summarize_chat(db, chat_id, owner_email)
            # Access summary_text while session is still open
            summary_preview = summary.summary_text[:100] if summary.summary_text else "Summary created"
            
        if bg_tracker:
            await bg_tracker.complete(message=f"Journal entry created: {summary_preview}")
        
        logger.info(f"Successfully generated journal for chat {chat_id}")
        
    except Exception as e:
        logger.error(f"Failed to generate journal for chat {chat_id}: {e}")
        if bg_tracker:
            await bg_tracker.fail(f"Journal generation failed: {str(e)}")
        raise


class AnalyticsService:
    async def summarize_chat(
        self,
        db: AsyncSession,
        chat_id: str,
        owner_email: str,
    ) -> ChatSummary:
        """Summarise a chat and persist analytics signals."""
        messages = await self._get_messages_for_chat(db, chat_id, owner_email)
        if not messages:
            raise ValueError(f"No messages found for chat_id={chat_id}")

        transcript = self._format_conversation_for_summary(messages)
        max_chars = 20_000
        if len(transcript) > max_chars:
            transcript = transcript[-max_chars:]

        model = get_model(efficient=True)
        # Format the prompt with the conversation transcript
        messages = SUMMARY_PROMPT.format_messages(conversation=transcript)
        resp = await model.ainvoke(messages)

        content = getattr(resp, "content", None)
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )

        if not isinstance(content, str):
            logger.warning(
                "Summariser returned non-string content for chat_id=%s: %r",
                chat_id,
                content,
            )
            data = {
                "summary": "Conversation summarisation failed (no model content).",
                "conversation_quality": {},
                "communication_feedback": {},
                "backend_enhancement_ideas": [],
            }
        else:
            try:
                data = json.loads(content)
            except Exception:
                logger.warning(
                    "Summariser returned non-JSON for chat_id=%s; wrapping raw content",
                    chat_id,
                )
                data = {
                    "summary": content,
                    "conversation_quality": {},
                    "communication_feedback": {},
                    "backend_enhancement_ideas": [],
                }

        summary = ChatSummary(
            chat_id=chat_id,
            owner_email=owner_email,
            model_version=getattr(settings, "VERSION", "unknown"),
            summary_text=str(data.get("summary", ""))[:4000],
            metrics=data,
            backend_enhancement_ideas=data.get("backend_enhancement_ideas") or None,
        )
        db.add(summary)
        await db.flush()

        backend_ideas = data.get("backend_enhancement_ideas")
        if isinstance(backend_ideas, list):
            for idea in backend_ideas:
                db.add(
                    BackendInsight(
                        source="chat_summary",
                        payload={
                            "chat_id": chat_id,
                            "owner_email": owner_email,
                            "idea": idea,
                        },
                    )
                )

        await db.commit()
        await db.refresh(summary)
        return summary

    async def _get_messages_for_chat(
        self,
        db: AsyncSession,
        chat_id: str,
        owner_email: str,
    ) -> list[BaseMessage]:
        """Get messages from LangGraph checkpoint for a chat."""
        from app.agent.service import AgentService
        from app.attachment.service import AttachmentService
        from sqlalchemy import select
        from app.chat.models import Chat
        from app.message.utils import get_chat_agent_type
        
        # Get chat to determine namespace/agent type
        stmt = select(Chat).where(Chat.chat_id == chat_id)
        result = await db.execute(stmt)
        chat = result.scalar_one_or_none()
        
        if not chat:
            logger.warning(f"Chat {chat_id} not found")
            return []
        
        # Get agent service
        attachment_service = AttachmentService()
        agent_service = AgentService(
            agent_type=get_chat_agent_type(chat.namespace),
            attachment_service=attachment_service
        )
        
        # Get latest checkpoint state
        try:
            state = await agent_service.get_latest_checkpoint(
                db, chat_id, owner_email, ""
            )
            return state.values.get("messages", [])
        except Exception as e:
            logger.error(f"Failed to get checkpoint messages for chat {chat_id}: {e}")
            return []

    def _format_conversation_for_summary(
        self,
        messages: list[BaseMessage],
    ) -> str:
        """Format LangGraph messages for summary."""
        from app.core.utils import extract_lc_msg_content
        
        lines: list[str] = []
        for m in messages:
            # Get role from message type
            msg_type = type(m).__name__
            if "Human" in msg_type:
                role = "USER"
            elif "AI" in msg_type or "Assistant" in msg_type:
                role = "ASSISTANT"
            elif "Tool" in msg_type:
                role = "TOOL"
            else:
                role = "SYSTEM"
            
            # Extract content - returns dict of {index: {type, text/...}}
            content_dict = extract_lc_msg_content(m)
            
            # Convert dict to readable text
            text_parts = []
            for idx in sorted(content_dict.keys()):
                item = content_dict[idx]
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, dict):
                    # Handle other content types (images, etc)
                    text_parts.append(f"[{item.get('type', 'unknown')}: {item.get('source', 'N/A')}]")
                else:
                    text_parts.append(str(item))
            
            content_str = " ".join(text_parts)
            lines.append(f"{role}: {content_str}")

        return "\n\n".join(lines)
