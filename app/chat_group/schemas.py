from typing import Annotated
from pydantic import BaseModel, StringConstraints
from app.core.schemas import Pagination

class ChatGroupCreate(BaseModel):
    title: Annotated[
                str,
                StringConstraints(
                    strip_whitespace=True, min_length=1, max_length=40
                ),
            ] = None
    owner_id: str  
    
class ChatGroupUpdate(BaseModel):
    title: Annotated[
                str,
                StringConstraints(
                    strip_whitespace=True, min_length=1, max_length=40
                ),
            ] = None

class ChatGroupResponse(BaseModel):
    group_id: str
    title: str
    owner_id: str

class ListChatGroupsResponse(Pagination[ChatGroupResponse]):
    pass
        