"""
app/attachment/models.py  (REFACTORED)

Changes from original:
  - Replaced s3_delete_by_prefix with storage_delete_by_prefix
  - The after_delete event now uses the unified storage layer
"""

from uuid import UUID, uuid4
import logging
import asyncio

from sqlalchemy import Index, event
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from app.db.mixin import TimestampMixin, LowerCaseEmailMixin
from app.storage import storage_delete_by_prefix

logger = logging.getLogger(__name__)

class Attachment(Base, TimestampMixin, LowerCaseEmailMixin):
    __tablename__ = "attachment"

    attachment_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    filename: Mapped[str]
    owner_id: Mapped[str]
    media_type: Mapped[str]
    vault_mode: Mapped[bool] = mapped_column(insert_default=False)
    s3_location: Mapped[str]
    preprocessed: Mapped[bool] = mapped_column(insert_default=False)

# Index for improved query performance on chat_id and owner_id
Index("idx_attchmnt_ownerid", Attachment.owner_id, Attachment.created_at.desc())

@event.listens_for(Attachment, "after_delete")
def delete_s3_file(mapper, conn, target: Attachment):
    """Delete attachment file from storage after the Attachment record is deleted"""
    try:
        s3_path = target.s3_location.replace("s3://", "")
        parts = s3_path.split("/", 1)

        if len(parts) == 2:
            bucket, prefix = parts
            # TODO: Temp fix. Use bgtask mgr.
            loop = asyncio.get_event_loop()
            loop.run_until_complete(storage_delete_by_prefix(bucket, prefix))
        else:
            logger.warning(f"Invalid storage path format: {target.s3_location}")
    except Exception as e:
        logger.error(f"Error setting up storage deletion for attachment {target.attachment_id}: {str(e)}")
