from typing import Annotated
import logging

from fastapi import APIRouter, HTTPException, Response, Cookie

from app.config import get_settings
from app.dependencies import DBSessionDep, UserEmailDep
from .schemas import (
    VaultExistsResp,
    VaultStatusResp,
    VaultPassword,
    VaultSetupResp,
    VaultValidationResp,
    VaultPasswordChangeReq
)
from .service import VaultServiceDep
from .dependencies import VaultKeyDep
from .exceptions import (
    VaultAlreadyExistsError,
    VaultNotExistError,
    InvalidHashException
)

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()

@router.get("/vault/exists")
async def check_vault_exists(
    email: UserEmailDep,
    db_session: DBSessionDep,
    vault_service: VaultServiceDep
) -> VaultExistsResp:
    """ Check if a user has their vault mode set up. """
    try:
        exists = await vault_service.vault_exists(db_session, email)
        return VaultExistsResp(exists=exists)
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vault/status")
async def check_vault_status(dek: VaultKeyDep) -> VaultStatusResp:
    return VaultStatusResp(enabled=bool(dek))

@router.post("/vault/setup")
async def setup_vault_password(
    data: VaultPassword,
    email: UserEmailDep,    
    db_session: DBSessionDep,
    vault_service: VaultServiceDep
) -> VaultSetupResp:
    try:
        success = await vault_service.setup_vault(db_session, email, data.password)
        return VaultSetupResp(success=success)
    except VaultAlreadyExistsError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))    

@router.post("/vault/verify")
async def verify_vault_password(
    data: VaultPassword,
    email: UserEmailDep,
    db_session: DBSessionDep,
    vault_service: VaultServiceDep,
    response: Response
) -> VaultValidationResp:
    # Validate and create vault session
    try:
        encrypted_session = await vault_service.verify_vault_password(db_session, email, data.password)
    except VaultNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidHashException as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))

    # Set cookie
    response.set_cookie(
        key="vault_session",
        value=encrypted_session,
        httponly=True,
        secure=not settings.LOCAL,
        samesite="lax",
        max_age=settings.VAULT_SESSION_DURATION_SEC
    )

    return VaultValidationResp(success=True)

@router.post("/vault/disable")
async def disable_vault_mode(
    email: UserEmailDep,
    db_session: DBSessionDep,
    vault_service: VaultServiceDep,
    response: Response,
    vault_session: Annotated[str | None, Cookie()] = None
):
    """Disable vault mode by deleting the session and clearing the cookie"""
    # Delete the session from database if it exists
    if vault_session:
        try:
            await vault_service.delete_vault_session(db_session, email, vault_session)
        except Exception as e:
            logger.exception(e)
    
    # Always clear the cookie, even if session deletion failed
    response.delete_cookie(key="vault_session")
    
    # From the user's perspective, disabling vault mode is successful if the cookie is cleared
    return {"success": True}

@router.post("/vault/change-password")
async def change_vault_password(
    data: VaultPasswordChangeReq,
    email: UserEmailDep,
    db_session: DBSessionDep,
    vault_service: VaultServiceDep,
    response: Response
):
    """Change the vault password and invalidate all sessions"""
    try:
        success = await vault_service.change_vault_password(
            db_session, 
            email, 
            data.oldPassword, 
            data.newPassword
        )
    except VaultNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidHashException as e:
        raise HTTPException(status_code=401, detail="Invalid current password")
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
    
    # Clear the current session cookie
    response.delete_cookie(key="vault_session")
    
    return {"success": success}


@router.post("/vault/reset-password")
async def reset_vault_password(
    email: UserEmailDep,
    db_session: DBSessionDep,
    vault_service: VaultServiceDep,
    response: Response
):
    """Reset vault by removing all vault credentials and sessions"""
    try:
        success = await vault_service.reset_vault(db_session, email)
    except VaultNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
    
    # Clear the current session cookie
    response.delete_cookie(key="vault_session")
    
    return {"success": success}
