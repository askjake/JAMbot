from __future__ import annotations

import enum
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Index
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.mixin import TimestampMixin


class AgentModeRunStatus(str, enum.Enum):
    """Lifecycle status for an Agent Mode run.

    We deliberately keep this small and human-readable so it can be
    surfaced directly in the UI and logs.
    """

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class AgentModeRun(TimestampMixin, Base):
    """Represents a single Agent Mode execution for a chat.

    The intent is to track:

    * Which chat and user initiated the run
    * Which agent handled it
    * High-level status + timestamps for basic SLAs
    * Optional free-form summary and error text
    * Optional JSON metadata for richer debug / replay in the future
    """

    __tablename__ = "agent_mode_run"

    # Core identifiers
    run_id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
        comment="Stable identifier for this run",
    )
    chat_id: Mapped[str] = mapped_column(
        nullable=False,
        index=True,
        comment="Foreign key to the chat this run is associated with (string id)",
    )
    owner_id: Mapped[str] = mapped_column(
        nullable=False,
        comment="User id / employee id who triggered the run",
    )

    # Agent + UX context
    agent_name: Mapped[str] = mapped_column(
        nullable=False,
        comment="Logical name of the agent that executed this run",
    )
    title: Mapped[str | None] = mapped_column(
        nullable=True,
        comment="Optional short human-readable title/summary for the run",
    )

    # Lifecycle + timing
    status: Mapped[AgentModeRunStatus] = mapped_column(
        SAEnum(AgentModeRunStatus, name="agent_mode_run_status"),
        nullable=False,
        default=AgentModeRunStatus.QUEUED,
        comment="High-level lifecycle state for the run",
    )
    started_at: Mapped[datetime | None] = mapped_column(
        nullable=True,
        comment="When the agent actually started doing work (if known)",
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        nullable=True,
        comment="When the run reached a terminal state (success/failure/cancel)",
    )

    # Debug / observability
    last_error: Mapped[str | None] = mapped_column(
        nullable=True,
        comment="Last error message / trace summary (if any)",
    )
    metadata_json: Mapped[str | None] = mapped_column(
        nullable=True,
        comment="Optional small JSON blob with extra metadata for the run",
    )


# Helpful compound index: most recent runs per chat
Index(
    "ix_agent_mode_run_chat_created_at",
    AgentModeRun.chat_id,
    AgentModeRun.created_at.desc(),
)
