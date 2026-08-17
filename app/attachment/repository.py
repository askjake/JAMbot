import logging
from typing import List

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.db.repository import BaseRepository
from .models import Attachment
from .schemas import (
    AttachmentDBRecord
)

logger = logging.getLogger(__name__)

class AttachmentRepository(
    BaseRepository[
        Attachment,
        AttachmentDBRecord,
        AttachmentDBRecord
    ]
):
    def __init__(self):
        super().__init__(Attachment)