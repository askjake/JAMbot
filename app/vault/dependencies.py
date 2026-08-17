from typing import Annotated
import logging

from fastapi import Depends, HTTPException, status, Cookie, Response

from app.config import get_settings
from app.dependencies import UserEmailDep, DBSessionDep

from .service import VaultServiceDep
from .exceptions import InvalidVaultCookieError

logger = logging.getLogger(__name__)
settings = get_settings()

async def get_vault_dek(
    response: Response,
    email: UserEmailDep,
    db_session: DBSessionDep,
    vault_service: VaultServiceDep,
    vault_session: Annotated[str | None, Cookie()] = None
) -> str:
    """
    FastAPI dependency that validates the vault session cookie and returns the DEK if exists.
    Refresh vault session if vailidation succeed.
    
    Raises:
        HTTPException: If the vault session is invalid

    Returns:
        str: The data encryption key (DEK)
    """
    if not vault_session:
        return ""

    try:
        dek = await vault_service.verify_vault_session(db_session, email, vault_session)
    except InvalidVaultCookieError as e:
        # Clear the cookie
        response.delete_cookie(key="vault_session")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        # Clear the cookie
        response.delete_cookie(key="vault_session")
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validating vault session: {str(e)}"
        )
    if dek:
        # Refresh vault session
        refreshed_session = await vault_service.refresh_vault_session(db_session, email, vault_session)
        response.set_cookie(
            key="vault_session",
            value=refreshed_session,
            httponly=True,
            secure=not settings.LOCAL,
            samesite="lax",
            max_age=settings.VAULT_SESSION_DURATION_SEC
        )
    else:
        # Clear the cookie
        response.delete_cookie(key="vault_session")
    
    return dek

VaultKeyDep = Annotated[str, Depends(get_vault_dek)]