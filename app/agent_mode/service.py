from __future__ import annotations

from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import func, select

from app.agent_mode.models import AgentModeRun, AgentModeRunStatus
from app.db import get_db_session_ctxmgr


class AgentModeRunService:
    """CRUD and query helpers for Agent Mode runs.

    This service is intentionally thin and stateless – it just owns the
    DB queries so routers / tools stay clean.
    """

    @staticmethod
    async def create_run(
        *,
        chat_id: str,
        owner_id: str,
        agent_name: str,
        title: str | None = None,
        metadata_json: str | None = None,
    ) -> AgentModeRun:
        """Create a new (queued) run record."""
        async with get_db_session_ctxmgr() as db:
            run = AgentModeRun(
                chat_id=chat_id,
                owner_id=owner_id,
                agent_name=agent_name,
                title=title,
                metadata_json=metadata_json,
                status=AgentModeRunStatus.QUEUED,
            )
            db.add(run)
            await db.commit()
            await db.refresh(run)
        return run

    @staticmethod
    async def list_runs_for_chat(
        *,
        chat_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[list[AgentModeRun], int]:
        """Return (runs, total) for a given chat id."""
        async with get_db_session_ctxmgr() as db:
            # Total count for pagination
            total_stmt = (
                select(func.count(AgentModeRun.run_id))
                .where(AgentModeRun.chat_id == chat_id)
            )
            total = (await db.execute(total_stmt)).scalar_one()

            # Page of rows
            stmt = (
                select(AgentModeRun)
                .where(AgentModeRun.chat_id == chat_id)
                .order_by(AgentModeRun.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            result = await db.execute(stmt)
            runs = list(result.scalars().all())

        return runs, total

    @staticmethod
    async def get_run(run_id: UUID) -> Optional[AgentModeRun]:
        """Fetch a single run by id."""
        async with get_db_session_ctxmgr() as db:
            run = await db.get(AgentModeRun, run_id)
        return run

    @staticmethod
    async def update_status(
        *,
        run_id: UUID,
        status: AgentModeRunStatus,
        last_error: str | None = None,
        started: bool | None = None,
        finished: bool | None = None,
    ) -> Optional[AgentModeRun]:
        """Best-effort status update helper.

        This is meant to be called by agent-execution code to mark
        a run as running / completed / failed, and optionally set
        started_at / finished_at timestamps.
        """
        async with get_db_session_ctxmgr() as db:
            run = await db.get(AgentModeRun, run_id)
            if not run:
                return None

            run.status = status
            if last_error is not None:
                run.last_error = last_error

            from datetime import datetime, timezone

            now = datetime.now(tz=timezone.utc)
            if started and run.started_at is None:
                run.started_at = now
            if finished:
                run.finished_at = now

            await db.commit()
            await db.refresh(run)
        return run
