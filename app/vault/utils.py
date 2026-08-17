from base64 import b64encode, b64decode
import re
from typing import Optional
import logging

from Crypto.Random import get_random_bytes
from Crypto.Hash import SHA256
import argon2

from app.config import get_settings
from app.core.encryption import cipher, decipher
from .exceptions import InvalidHashException

settings = get_settings()
logger = logging.getLogger(__name__)


def encrypt_dek(passphrase: str, dek: Optional[str] = None) -> tuple[str, str]:
    '''
    Generate a data encryption key encrypted with user passphrase.
    Argon2id is used to generate a passphrase-derived key, and this key is used to encrypt the file key with AES-256. 
    If a key is provided simply encrypt the provided key with passphrase.
    If a key is not provided, generate a random 256 bit key and encrypt it.
    A SHA-256 hash of the passphrase-derived key is calculated for passphrase validation purpose.
    
    Args:
        passphrase (str): A passphrase that user provides.
        file_key (str): A 256 bit file key. (Optional)
    
    Return:
        vhash (str): Base64 encoded 256 bit validation hash
        encrypted_file_key (str): A base64 encoded encrypted file key in the format of {argon2_params}{aes_params}${cipheredkey}
    '''
    
    # If no key is provided, generate a random 256-bit key
    if dek is None:
        dek = get_random_bytes(32)
    else:
        dek = b64decode(dek)

    print(b64encode(dek).decode('ascii'))
        
    if len(dek) != 32:
        raise ValueError("Data encryption key must be 256 bits.")

    # Create an Argon2 hasher
    argon2_hasher = argon2.PasswordHasher(
        time_cost=settings.ARGON2_TIME,
        memory_cost=settings.ARGON2_MEM,
        parallelism=settings.ARGON2_PRL,
        hash_len=32,
        salt_len=16,
        encoding='ascii'
    )

    # Hash the passphrase to derive a key
    argon2_hash = argon2_hasher.hash(passphrase)
    derived_key = argon2_hash.split('$')[-1] + '='  # Add base64 padding
    argon2_params = '$'.join(argon2_hash.split('$')[:-1])
    
    # Cipher the data key with passphrase derived key
    ciphered_key = cipher(dek, derived_key)
    
    # Hash the derived key
    sha256 = SHA256.new()
    sha256.update(b64decode(derived_key))
    vhash = b64encode(sha256.digest()).decode("ascii")
    
    return vhash, argon2_params + f"${b64encode(ciphered_key).decode('ascii')}"

def decrypt_dek(passphrase: str, vhash: str, encrypted_dek: str) -> str:
    '''
    Validate passphrase and decrypt data encryption key to be used for chat data decryption
    
    Args:
        passphrase (str): Valid user passphrase
        vhash(str): Validation hash
        encrypted_dek (str): Encrypted data key and metadata
    
    Return:
        Base64 encoded 256 bit data key if passphrase is valid, else None
    '''
    
    # Split the encrypted key into Argon2 parameters and ciphered key
    parts = encrypted_dek.split('$')
    match = re.search(r'm=(\d+),t=(\d+),p=(\d+)', parts[3])
    m = int(match.group(1))
    t = int(match.group(2))
    p = int(match.group(3))
    argon2_salt = b64decode(parts[4] + "==")
    aes_ciphertext = b64decode(parts[5])

    # Recreate the Argon2 hasher with the same parameters
    argon2_hasher = argon2.PasswordHasher(
        time_cost=t,
        memory_cost=m,
        parallelism=p,
        hash_len=32,
        salt_len=16,
        encoding="ascii")

    # Get the derived key
    try:
        derived_key = argon2_hasher.hash(passphrase, salt=argon2_salt).split('$')[-1] + "="
    except Exception as e:
        logger.error(f"Failed to get the derived key: {str(e)}")
        raise

    # Validate the derived key
    sha256 = SHA256.new()
    sha256.update(b64decode(derived_key))
    if sha256.digest() != b64decode(vhash):
        logger.error("Failed to decrypt data key: vhash doesn't match")
        raise InvalidHashException("Validation hash of the password doesn't match record.")

    # Decrypt the data key
    try:
        decrypted_dek = decipher(aes_ciphertext, derived_key)
    except Exception as e:
        # Handle any other exceptions
        logger.error(f"Failed to decrypt data key: {str(e)}")
        raise
    
    return b64encode(decrypted_dek).decode("ascii")
