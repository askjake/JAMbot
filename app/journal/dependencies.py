from typing import Annotated
from fastapi import Depends

from .service import JournalService


def get_journal_service() -> JournalService:
    return JournalService()


JournalServiceDep = Annotated[JournalService, Depends(get_journal_service)]
