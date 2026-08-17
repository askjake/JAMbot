from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db_session
from app.core.user import get_user_email
from app.background_mgr.service import get_task_manager

DBSessionDep = Annotated[AsyncSession, Depends(get_db_session)]
UserEmailDep = Annotated[str, Depends(get_user_email)]