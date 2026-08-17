import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.core.llm import get_model
from app.db import get_db_session_ctxmgr
from .models import BackendInsight, ChatSummary

logger = logging.getLogger(__name__)
settings = get_settings()


async def _collect_recent_summaries(
    db: AsyncSession,
    since: datetime,
) -> List[dict]:
    """Return a list of serialisable summaries created since `since`."""
    stmt = (
        select(ChatSummary)
        .where(ChatSummary.created_at >= since)
        .order_by(ChatSummary.created_at.asc())
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    payload: List[dict] = []
    for row in rows:
        payload.append(
            {
                "chat_id": row.chat_id,
                "owner_email": row.owner_email,
                "created_at": row.created_at.isoformat()
                if row.created_at
                else None,
                "summary_text": row.summary_text,
                "metrics": row.metrics,
                "backend_enhancement_ideas": row.backend_enhancement_ideas,
            }
        )

    return payload


async def run_backend_review(days: int = 1) -> None:
    """Aggregate recent ChatSummary rows into BackendInsight recommendations.

    This function is intended to be run as a periodic job (e.g. daily
    via a cron or K8s CronJob). It:
    - Fetches recent ChatSummary entries.
    - Asks an LLM to deduplicate and prioritise backend improvements.
    - Stores the recommendations in BackendInsight with source="review".
    """
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(days=days)

    async with get_db_session_ctxmgr() as db:
        summaries = await _collect_recent_summaries(db, cutoff)
        if not summaries:
            logger.info("No ChatSummary rows found since %s; nothing to review", cutoff)
            return

        model = get_model(efficient=True)

        system = SystemMessage(
            content=(
                "You are a senior architect reviewing many conversation-level "
                "analytics records from a chatbot system. Each record may "
                "contain:\n"
                "- A conversation summary.\n"
                "- Conversation-quality metrics.\n"
                "- Backend enhancement ideas specific to that chat.\n\n"
                "Your goal is to aggregate these into a deduplicated list of "
                "backlog items that would most improve the chatbot backend.\n\n"
                "Return ONLY valid JSON: a list of recommendation objects like:\n"
                "[\n"
                "  {\n"
                "    \"id\": \"<short stable id>\",\n"
                "    \"title\": \"<short title>\",\n"
                "    \"description\": \"<detailed description>\",\n"
                "    \"component\": \"<component to improve>\",\n"
                "    \"priority\": \"low|medium|high\",\n"
                "    \"rationale\": \"<why this matters>\"\n"
                "  }\n"
                "]\n"
            )
        )


        human = HumanMessage(
            content=(
                "Here are recent per-chat analytics records in JSON form. "
                "Analyse them and produce a deduplicated, prioritised list of "
                "backend improvement recommendations.\n"
                + json.dumps(summaries, ensure_ascii=False)
            )
        )

        resp = await model.ainvoke([system, human])
        content = getattr(resp, "content", None)
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )

        if not isinstance(content, str):
            logger.warning(
                "Backend review produced non-string content; wrapping as generic insight"
            )
            recommendations = [
                {
                    "id": "non-string-output",
                    "title": "Unparsed backend review output",
                    "description": str(content),
                    "component": "unknown",
                    "priority": "medium",
                    "rationale": "LLM output was not a string; stored raw content.",
                }
            ]
        else:
            try:
                recommendations = json.loads(content)
                if not isinstance(recommendations, list):
                    raise TypeError("Expected list of recommendations")
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Backend review returned non-JSON or unexpected format; storing raw output"
                )
                recommendations = [
                    {
                        "id": "raw-review-output",
                        "title": "Unparsed backend review output",
                        "description": content[:4000],
                        "component": "unknown",
                        "priority": "medium",
                        "rationale": "LLM output could not be parsed as JSON; see description.",
                    }
                ]

        for rec in recommendations:
            db.add(BackendInsight(source="review", payload=rec))

        # get_db_session_ctxmgr will commit if we didn't raise
        logger.info("Stored %d BackendInsight rows from review job", len(recommendations))


if __name__ == "__main__":
    asyncio.run(run_backend_review())
