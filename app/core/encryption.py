
from base64 import b64decode
from typing import BinaryIO
from tempfile import SpooledTemporaryFile
import logging
import os

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

from app.config import get_settings

settings = get_settings()
CHUNK_SIZE = 64 * 1024  # 64KB chunks for file encryption
logger = logging.getLogger(__name__)

def cipher(plaintext: bytes, b64key: str) -> bytes:
    """
    Cipher the input plaintext with provided base64 encoded key with AES-256 in GCM mode. 
    Return ciphertext in bytes
    """
    key = b64decode(b64key)
    if not len(key) == 32:
        raise ValueError("Encryption Key is not 256 bits.")

    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)

    return nonce + tag + ciphertext

def decipher(ciphertext: bytes, b64key: str) -> bytes:
    """
    Decipher the input ciphertext with provided base64 encoded key with AES-256 in GCM mode.
    Return plaintext in bytes
    """
    key = b64decode(b64key)
    if not len(key) == 32:
        raise ValueError("Encryption Key is not 256 bits.")

    nonce = ciphertext[:12]
    tag = ciphertext[12:28]
    actual_ciphertext = ciphertext[28:]

    cipher = AES.new(key, AES.MODE_GCM, nonce)
    plaintext = cipher.decrypt_and_verify(actual_ciphertext, tag)

    return plaintext

def cipher_file(inputfile: BinaryIO, b64key: str) -> SpooledTemporaryFile:
    '''
    Encrypt an arbitrarily sized file with provided key using AES-256 in GCM mode.
    '''
    
    # Decode the base64 key
    key = b64decode(b64key)
    
    # Generate a random nonce
    nonce = get_random_bytes(12)
    
    # Create cipher object
    aes_cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    outfile = SpooledTemporaryFile(max_size=settings.SPOOLED_MAX_SIZE)


    outfile.write(nonce)

    # Reserve room for tag
    outfile.seek(16, 1)

    # Process the file in chunks
    while True:
        chunk = inputfile.read(CHUNK_SIZE)
        if len(chunk) == 0:
            break
        encrypted_chunk = aes_cipher.encrypt(chunk)
        outfile.write(encrypted_chunk)

        # Get and write the tag
        tag = aes_cipher.digest()
        outfile.seek(12, 0)
        outfile.write(tag)

def decipher_file(in_filepath, out_filepath, b64key) -> bool:
    '''
    Decrypt an arbitrarily sized file with provided key using AES-256 in GCM mode.
    '''

    # Decode the base64 key
    key = b64decode(b64key)
    
    # Open input and output files
    with open(in_filepath, 'rb') as infile, open(out_filepath, 'wb') as outfile:
        # Read the nonce
        nonce = infile.read(12)

        # Read the tag
        tag = infile.read(16)
        
        # Create cipher object
        aes_cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        
        # Process the file in chunks
        while True:
            chunk = infile.read(CHUNK_SIZE)
            if len(chunk) == 0:
                break
            decrypted_chunk = aes_cipher.decrypt(chunk)
            outfile.write(decrypted_chunk)
        
        # Read and verify the tag
        try:
            aes_cipher.verify(tag)
            return True

        except ValueError:
            os.remove(out_filepath)  # Remove the output file if verification fails
            logger.error("Key incorrect or message corrupted")
            return False        
