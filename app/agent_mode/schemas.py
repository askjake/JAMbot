from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.agent_mode.models import AgentModeRunStatus


class AgentModeRunSchema(BaseModel):
    """Public representation of an Agent Mode run.

    This is what we return via the API. It intentionally mirrors
    the ORM model but keeps only the fields that are useful to the UI.
    """

    run_id: UUID
    chat_id: str
    owner_id: str

    agent_name: str
    title: Optional[str] = None

    status: AgentModeRunStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    last_error: Optional[str] = None
    metadata_json: Optional[str] = None

    # Pydantic v2 configuration
    model_config = ConfigDict(from_attributes=True)


class AgentModeRunListResponse(BaseModel):
    """Paginated list wrapper for Agent Mode runs."""

    items: List[AgentModeRunSchema]
    total: int
