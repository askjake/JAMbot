from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query

from app.agent_mode.schemas import AgentModeRunListResponse, AgentModeRunSchema
from app.agent_mode.service import AgentModeRunService

router = APIRouter(
    prefix="/runs",
    tags=["agent-mode"],
)


@router.get(
    "",
    response_model=AgentModeRunListResponse,
    summary="List Agent Mode runs for a chat",
)
async def list_runs(
    chat_id: Annotated[
        str,
        Query(
            description="Chat id whose Agent Mode runs you want to list",
        ),
    ],
    limit: Annotated[
        int,
        Query(ge=1, le=100, description="Max runs to return (default 50)"),
    ] = 50,
    offset: Annotated[
        int,
        Query(ge=0, description="Offset for pagination (default 0)"),
    ] = 0,
):
    """Return the most recent Agent Mode runs for a given chat.

    This is wired into the new Agent Mode run table and backed by
    real SQL queries.
    """
    runs, total = await AgentModeRunService.list_runs_for_chat(
        chat_id=chat_id,
        limit=limit,
        offset=offset,
    )
    return AgentModeRunListResponse(items=runs, total=total)


@router.get(
    "/{run_id}",
    response_model=AgentModeRunSchema,
    summary="Get a single Agent Mode run by id",
)
async def get_run(run_id: UUID):
    """Fetch a single Agent Mode run.

    Returns 404 if the run does not exist.
    """
    run = await AgentModeRunService.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Agent Mode run not found")
    return run
