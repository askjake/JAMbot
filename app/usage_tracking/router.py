from typing import Annotated
import datetime

from fastapi import APIRouter, HTTPException, Path, Query
from app.core.utils import UUID_REGEX
from app.dependencies import DBSessionDep, UserEmailDep
from app.vault.dependencies import VaultKeyDep

from .dependencies import UsageTrackingServiceDep
from .schemas import TokenUsageResp
from .exceptions import ChatNotFoundError, NotAuthorizedError, VaultAccessError


router = APIRouter()


@router.get("/token_usage", response_model=TokenUsageResp)
async def get_token_usage(
    email: UserEmailDep,
    db_session: DBSessionDep,
    usage_service: UsageTrackingServiceDep,
    start_date: Annotated[datetime.date, Query()],
    end_date: Annotated[datetime.date, Query()],
) -> TokenUsageResp:
    """
    Retrieve token usage for a specific user, optionally filtered by start and end date.
    """
    if start_date > end_date:
        raise HTTPException(
            status_code=400, detail="Start date must be before end date."
        )

    try:
        return await usage_service.get_usage_by_owner(
            db_session, email, before=end_date, after=start_date
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.get("/chats/{chat_id}/token_usage", response_model=TokenUsageResp)
async def get_chat_token_usage(
    chat_id: Annotated[str, Path(regex=UUID_REGEX)],
    email: UserEmailDep,
    db_session: DBSessionDep,
    usage_service: UsageTrackingServiceDep,
    vault_key: VaultKeyDep,
) -> TokenUsageResp:
    """
    Retrieve token usage for a specific chat.
    """
    try:
        return await usage_service.get_usage_by_chat(
            db_session, email, chat_id, bool(vault_key)
        )
    except ChatNotFoundError:
        raise HTTPException(status_code=404, detail="Chat not found.")
    except NotAuthorizedError:
        raise HTTPException(
            status_code=403, detail="Requesting usage from a chat you do not own."
        )
    except VaultAccessError:
        raise HTTPException(
            status_code=401,
            detail="Requesting usage from a vault chat requires vault mode.",
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error.")
