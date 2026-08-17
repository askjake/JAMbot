from typing import Annotated

from fastapi import Depends

from app.attachment.service import AttachmentService


def get_attachment_service():
    return AttachmentService()


AttachmentServiceDep = Annotated[AttachmentService, Depends(get_attachment_service)]
