from datetime import datetime, UTC, timedelta
from uuid import UUID

from sqlalchemy import Index, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from app.db.mixin import TimestampMixin
from app.config import get_settings

settings = get_settings()


class VaultCred(Base):
    __tablename__ = "vault_credential"

    owner_id: Mapped[str] = mapped_column(primary_key=True)
    encrypted_dek: Mapped[str]
    vhash: Mapped[str]


class VaultSession(Base, TimestampMixin):
    __tablename__ = "vault_session"

    session_id: Mapped[UUID] = mapped_column(primary_key=True)
    owner_id: Mapped[str]
    expires_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC).replace(tzinfo=None)
        + timedelta(seconds=settings.VAULT_SESSION_DURATION_SEC),
    )
    rotate_token_hash: Mapped[str]

    __table_args__ = (Index("idx_vault_owner_id", "owner_id"),)
