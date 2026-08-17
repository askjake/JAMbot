from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

class VaultSessionDBRecord(BaseModel):
    session_id: UUID
    user_email: str
    rotate_token_hash: str

class VaultCredDBRecord(BaseModel):
    user_email: str
    encrypted_dek: str
    vhash: str

class VaultExistsResp(BaseModel):
    exists: bool

class VaultStatusResp(BaseModel):
    enabled: bool

class VaultPassword(BaseModel):
    password: str

class VaultSetupResp(BaseModel):
    success: bool

class VaultValidationResp(BaseModel):
    success: bool

class VaultPasswordChangeReq(BaseModel):
    oldPassword: str
    newPassword: str
