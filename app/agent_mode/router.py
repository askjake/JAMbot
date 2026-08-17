# app/agent_mode/router.py
from fastapi import APIRouter, Depends, Query
from typing import Optional
from uuid import UUID

from app.db import get_session  # whatever your db helper is

router = APIRouter(prefix="/rest/api/v1/agent-mode", tags=["agent-mode"])

@router.get("/runs")
async def list_runs(
    chat_id: Optional[UUID] = Query(None),
    session = Depends(get_session),
):
    """
    Return recent agent-mode runs, optionally filtered by chat_id.
    """
    # Pseudo-SQLAlchemy; adapt to your models.
    q = session.query(AgentRun).order_by(AgentRun.started_at.desc())
    if chat_id:
        q = q.filter(AgentRun.chat_id == chat_id)

    runs = q.limit(100).all()

    return {
        "items": [
            {
                "id": str(run.id),
                "command": run.command,
                "repo_url": run.repo_url,
                "chat_id": str(run.chat_id) if run.chat_id else None,
                "status": run.status,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "error": run.error,
                "artifacts": run.artifacts or [],
            }
            for run in runs
        ]
    }

