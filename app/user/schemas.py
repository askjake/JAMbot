from datetime import date as date_type
from typing import Optional
from pydantic import BaseModel

class User(BaseModel):
    email: str
    first_name: str
    last_name: str
    last_release_date: date_type | None

class UserStateUpdate(BaseModel):
    current_chat_id: Optional[str]

class UserStateCreate(UserStateUpdate):
    owner_id: str
