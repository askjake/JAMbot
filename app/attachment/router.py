from typing import Annotated, Optional
import logging
import os

from fastapi import APIRouter, Cookie, Depends, UploadFile, File, HTTPException, Path
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .exceptions import AttachmentNotFoundError, NotAuthorizedError, VaultAccessError

from app.dependencies import DBSessionDep, UserEmailDep
from app.core.utils import UUID_REGEX
from app.vault.dependencies import VaultKeyDep

from .schemas import (
    AttachmentCreateResp,
    AttachmentsUpdateReq,
    AttachmentsStatusResp,
    AttachmentStatusResp
)
from .service import get_attachment_service, AttachmentService
from .utils import safe_unlink
AttachmentServiceDep = Annotated[AttachmentService, Depends(get_attachment_service)]

logger = logging.getLogger(__name__)
router = APIRouter()

# Attachment endpoints
@router.post("/attachments")
async def upload_attachments(
    attachments: Annotated[
        list[UploadFile], File(description="Multiple files as UploadFile")
    ],
    email: UserEmailDep,
    db_session: DBSessionDep,
    attachment_service: AttachmentServiceDep,
    vault_key: VaultKeyDep,
) -> AttachmentCreateResp:
    upload_status = await attachment_service.upload_files_for_user(attachments, email, db_session, vault_key)
    return AttachmentCreateResp(attachments=upload_status)

@router.post("/attachments/status")
async def get_attachments_status(
    data: AttachmentsUpdateReq,
    email: UserEmailDep,
    db_session: DBSessionDep,
    attachment_service: AttachmentServiceDep,
    vault_key: VaultKeyDep,
) -> AttachmentsStatusResp:
    """
    Get status of multiple attachments with processing information.
    Returns null for attachments that don't exist or user isn't authorized to access.
    """
    try:
        status_dict = await attachment_service.get_attachments_status(
            db_session,
            data.attachment_ids,
            email,
            vault_key
        )
        return AttachmentsStatusResp(status=status_dict)

    except Exception as e:
        logger.exception(f"Error retrieving attachment status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving attachment status"
        )

@router.get("/attachments/{attachment_id}")
async def get_attachment(
    attachment_id: Annotated[str, Path(regex=UUID_REGEX)],
    email: UserEmailDep,
    db_session: DBSessionDep,
    attachment_service: AttachmentServiceDep,
    vault_key: VaultKeyDep,
):
    """
    Download an attachment file.

    Returns the file as a streaming response with appropriate content type.
    """
    try:
        file_path, media_type, filename = await attachment_service.get_attachment_file(
            db_session, attachment_id, email, vault_key
        )

        # Create a streaming response from the temporary file
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename,
            background=BackgroundTask(lambda: os.unlink(file_path))   # Delete file after response is sent
        )

    except AttachmentNotFoundError:
        raise HTTPException(status_code=404, detail="Attachment not found")
    except NotAuthorizedError:
        raise HTTPException(status_code=403, detail="Not authorized to access this attachment")
    except VaultAccessError:
        raise HTTPException(status_code=401, detail="Accessing vault attachments requires vault mode")
    except Exception as e:
        logger.exception(f"Error retrieving attachment: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve attachment")

# In /attachment/router.py
@router.get("/attachments/{attachment_id}/status", response_model=AttachmentStatusResp)
async def get_attachment_status(
    attachment_id: Annotated[str, Path(regex=UUID_REGEX)],
    email: UserEmailDep,
    db_session: DBSessionDep,
    attachment_service: AttachmentServiceDep,
    vault_key: VaultKeyDep,
) -> AttachmentStatusResp:
    """
    Get status of a specific attachment with processing information.
    """
    try:
        status = await attachment_service.get_attachment_status(
            db_session, attachment_id, email, vault_key
        )
        return status

    except AttachmentNotFoundError:
        raise HTTPException(status_code=404, detail="Attachment not found")
    except NotAuthorizedError:
        raise HTTPException(status_code=403, detail="Not authorized to access this attachment")
    except VaultAccessError:
        raise HTTPException(status_code=401, detail="Accessing vault attachments requires vault mode")
    except Exception as e:
        logger.exception(f"Error retrieving attachment status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving attachment status"
        )