from typing import Annotated

from fastapi import Depends

from app.message.service import MessageService
from app.chat.dependencies import ChatServiceDep
from app.attachment.dependencies import AttachmentServiceDep


def get_message_service(
    chat_service: ChatServiceDep, attachment_service: AttachmentServiceDep
) -> MessageService:
    return MessageService(chat_service, attachment_service)


MessageServiceDep = Annotated[MessageService, Depends(get_message_service)]
