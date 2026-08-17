from typing import Annotated

from fastapi import Depends

from app.chat.service import ChatService


def get_chat_service():
    return ChatService()


ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
