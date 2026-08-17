import logging
from typing import Optional, Annotated
import json
from base64 import b64encode, b64decode
from uuid import uuid4, UUID
import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from Crypto.Random import get_random_bytes
from Crypto.Hash import SHA256

from app.core.encryption import cipher, decipher
from app.config import get_settings

from .schemas import VaultCredDBRecord, VaultSessionDBRecord
from .repository import VaultCredRepository, VaultSessionRepository
from .exceptions import (
    VaultAlreadyExistsError,
    VaultNotExistError,
    InvalidHashException,
    InvalidVaultCookieError
)
from .utils import encrypt_dek, decrypt_dek

logger = logging.getLogger(__name__)
settings = get_settings()

class VaultService:
    def __init__(
        self,
        cred_repo: Optional[VaultCredRepository] = None,
        session_repo: Optional[VaultSessionRepository] = None
    ):
        self.cred_repo = cred_repo or VaultCredRepository()
        self.session_repo = session_repo or VaultSessionRepository()

    async def vault_exists(self, db: AsyncSession, user_email: str) -> bool:
        """ Check if a user has their vault mode set up. """
        cred = await self.cred_repo.get_one_by_id(db, user_email)
        if cred:
            return True
        else:
            return False

    async def setup_vault(
        self,
        db: AsyncSession,
        user_email: str,
        password: str
    ) -> bool:
        """ Setup vault for user for the first time"""
        exists = await self.vault_exists(db, user_email)
        if exists:
            logger.warning(f"User {user_email} attempts to setup vault when vault already exists.")
            raise VaultAlreadyExistsError("Setting up vault when a vault already exists is not allowed.")

        # Generate dek
        vhash, encrypted_dek = encrypt_dek(password)
        vault_cred = VaultCredDBRecord(
            user_email=user_email,
            encrypted_dek=encrypted_dek,
            vhash=vhash
        )

        # Save cred
        cred_obj = await self.cred_repo.create_one(db, obj_in=vault_cred)

        return cred_obj is not None

    async def verify_vault_password(
        self,
        db: AsyncSession,
        user_email: str,
        password: str
    ) -> str:
        # Get cred
        cred_obj = await self.cred_repo.get_one_by_id(db, user_email)
        if not cred_obj:
            logger.warning(f"User {user_email} attempts to enter vault when vault doesn't exist.")
            raise VaultNotExistError("Vault does not exist. Setup vault first.")

        # Attempt decryption
        try:
            dek = decrypt_dek(password, cred_obj.vhash, cred_obj.encrypted_dek)
        except InvalidHashException as e:
            logger.warning(f"User {user_email} failed vault validation attempt.")
            raise e

        # Create cookie session
        # Generate rotating token
        rotate_token = get_random_bytes(32)
        sha256 = SHA256.new(rotate_token)
        rotate_token_hash = b64encode(sha256.digest()).decode("ascii")

        session_db_record = VaultSessionDBRecord(
            session_id=uuid4(),
            user_email=user_email,
            rotate_token_hash=rotate_token_hash
        )
        session_db_obj = await self.session_repo.create_one(db, obj_in=session_db_record)

        session_cookie = {
            "session_id": str(session_db_obj.session_id),
            "dek": dek,
            "rotate_token": b64encode(rotate_token).decode("ascii")
        }

        # Encrypt the session cookie with master key
        encrypted_session = b64encode(
            cipher(json.dumps(session_cookie).encode("ascii"), settings.MASTER_KEY)
        ).decode("ascii")

        return encrypted_session

    async def verify_vault_session(
        self,
        db: AsyncSession,
        user_email: str,
        encrypted_session: str,
    ) -> str:
        """ Validate the vault session and return the data encryption key on success"""
        # Decrypt session cookie
        try:
            session = json.loads(decipher(b64decode(encrypted_session), settings.MASTER_KEY))
            session_id = session["session_id"]
            dek = session["dek"]
            rotate_token = session["rotate_token"]
        except Exception as e:
            logger.warning(
                "Encountered invalid vault session cookie while attempting "
                f"to verify for user {user_email}: {encrypted_session}. Error: {str(e)}"
            )
            raise InvalidVaultCookieError(
                f"Encountered invalid vault session cookie while attempting to verify: {str(e)}"
            )

        # Validate session
        session_db_record = await self.session_repo.get_one_by_id(db, UUID(session_id))
        # Cannot find session
        if not session_db_record:
            logger.warning(
                "Encountered vault session cookie with session_id not in record."
                f"User: {user_email}, session_id: {session_id}"
            )
            return ""

        # Session expired
        if datetime.datetime.now(datetime.UTC).replace(tzinfo=None) > session_db_record.expires_at:
            await self.session_repo.remove_by_id(db, id=UUID(session_id))
            return ""

        # Check if rotate token not match. Potential attack
        rotate_token_hash = SHA256.new(b64decode(rotate_token)).digest()
        if b64encode(rotate_token_hash).decode("ascii") != session_db_record.rotate_token_hash:
            logger.warning(
                "Encountered invalid vault session cookie with mismatched rotate token. "
                f"User: {user_email}, session_id: {session_id}, rotate_token: {rotate_token}")
            return ""

        # Valid
        return dek

    async def refresh_vault_session(
        self,
        db: AsyncSession,
        user_email: str,
        encrypted_session: str,
    ) -> str:
        """Refresh the vault session by extending expiration time and generating a new rotate token"""
        # Decrypt session cookie
        try:
            session = json.loads(decipher(b64decode(encrypted_session), settings.MASTER_KEY))
            session_id = session["session_id"]
            dek = session["dek"]
        except Exception as e:
            logger.warning(
                "Encountered invalid vault session cookie while attempting "
                f"to refresh for user {user_email}: {encrypted_session}. Error: {str(e)}"
            )
            raise InvalidVaultCookieError(
                f"Encountered invalid vault session cookie while attempting to refresh: {str(e)}"
            )

        # Generate new rotating token
        rotate_token = get_random_bytes(32)
        rotate_token_hash = b64encode(SHA256.new(rotate_token).digest()).decode("ascii")

        # Update session with new expiration time and rotate token
        session_db_obj = await self.session_repo.get_one_by_id(db, UUID(session_id))
        if not session_db_obj:
            logger.warning(f"Session {session_id} not found during refresh for user {user_email}")
            raise InvalidVaultCookieError("Session not found")

        # Calculate new expiration time using session duration from settings
        new_expires_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None) + datetime.timedelta(seconds=settings.VAULT_SESSION_DURATION_SEC)

        # Update session with new rotate token hash and expiration time
        await self.session_repo.update(
            db,
            db_obj=session_db_obj,
            obj_in={
                "rotate_token_hash": rotate_token_hash,
                "expires_at": new_expires_at
            }
        )

        # Create new session cookie with updated rotate token
        new_session = {
            "session_id": session_id,
            "dek": dek,
            "rotate_token": b64encode(rotate_token).decode("ascii")
        }

        # Encrypt the new session cookie
        encrypted_new_session = b64encode(
            cipher(json.dumps(new_session).encode("ascii"), settings.MASTER_KEY)
        ).decode("ascii")

        return encrypted_new_session

    async def delete_vault_session(
        self,
        db: AsyncSession,
        user_email: str,
        encrypted_session: str
    ) -> bool:
        """Delete the vault session from the database"""
        try:
            # Decrypt session cookie to get session_id
            session = json.loads(decipher(b64decode(encrypted_session), settings.MASTER_KEY))
            session_id = session["session_id"]

            # Delete the session from database
            result = await self.session_repo.remove_by_id(db, id=UUID(session_id))
            return result
        except Exception as e:
            logger.exception(f"Failed to delete vault session for user: {user_email}")
            return False

    async def change_vault_password(
        self,
        db: AsyncSession,
        user_email: str,
        current_password: str,
        new_password: str
    ) -> bool:
        """Change the vault password for a user"""
        # Get current credentials
        cred_obj = await self.cred_repo.get_one_by_id(db, user_email)
        if not cred_obj:
            logger.warning(f"User {user_email} attempts to change vault password when vault doesn't exist.")
            raise VaultNotExistError("Vault does not exist. Setup vault first.")

        # Verify current password
        try:
            decrypt_dek(current_password, cred_obj.vhash, cred_obj.encrypted_dek)
        except InvalidHashException as e:
            logger.warning(f"User {user_email} failed vault password change attempt - invalid current password.")
            raise e

        # Generate new credentials with new password
        vhash, encrypted_dek = encrypt_dek(new_password)

        # Update credentials
        await self.cred_repo.update(
            db,
            db_obj=cred_obj,
            obj_in={
                "encrypted_dek": encrypted_dek,
                "vhash": vhash
            }
        )

        # Delete all sessions for this user
        await self.session_repo.remove_by_user(db, user_email)

        return True

    async def reset_vault(
        self,
        db: AsyncSession,
        user_email: str
    ) -> bool:
        """Reset vault by removing all vault credentials and sessions for a user"""
        # Check if vault exists
        cred_obj = await self.cred_repo.get_one_by_id(db, user_email)
        if not cred_obj:
            logger.warning(f"User {user_email} attempts to reset vault when vault doesn't exist.")
            raise VaultNotExistError("Vault does not exist.")

        # Delete all sessions for this user
        await self.session_repo.remove_by_user(db, user_email)

        # Delete vault credentials
        await self.cred_repo.remove_by_id(db, id=user_email)

        return True


def get_vault_service():
    return VaultService()

VaultServiceDep = Annotated[VaultService, Depends(get_vault_service)]
