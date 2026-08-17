import logging

from app.db.repository import BaseRepository

from .models import BgTask
from .schemas import (
    BgTaskSchema
)

logger = logging.getLogger(__name__)

class BgTaskRepository(
    BaseRepository[
        BgTask,
        BgTaskSchema,
        BgTaskSchema
    ]
):
    def __init__(self):
        super().__init__(BgTask)