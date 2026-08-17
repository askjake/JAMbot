import pytest
from base64 import b64encode
from Crypto.Random import get_random_bytes
from app.core.encryption import cipher, decipher

def generate_b64key():
    return b64encode(get_random_bytes(32)).decode()

def test_cipher_and_decipher_roundtrip():
    b64key = generate_b64key()
    plaintext = b"hello world"
    ciphertext = cipher(plaintext, b64key)
    decrypted = decipher(ciphertext, b64key)
    assert decrypted == plaintext

def test_decipher_with_wrong_key_raises():
    b64key = generate_b64key()
    b64key_wrong = generate_b64key()
    plaintext = b"secret"
    ciphertext = cipher(plaintext, b64key)
    with pytest.raises(ValueError):
        decipher(ciphertext, b64key_wrong)

def test_cipher_with_invalid_key_length_raises():
    bad_b64key = b64encode(get_random_bytes(16)).decode()  # 128 bits, not 256
    with pytest.raises(ValueError):
        cipher(b"data", bad_b64key)

def test_decipher_with_invalid_key_length_raises():
    b64key = generate_b64key()
    plaintext = b"test"
    ciphertext = cipher(plaintext, b64key)
    bad_b64key = b64encode(get_random_bytes(16)).decode()
    with pytest.raises(ValueError):
        decipher(ciphertext, bad_b64key)

def test_decipher_with_tampered_ciphertext_raises():
    b64key = generate_b64key()
    plaintext = b"attack at dawn"
    ciphertext = cipher(plaintext, b64key)
    tampered = bytearray(ciphertext)
    tampered[-1] ^= 0xFF  # flip last byte
    with pytest.raises(ValueError):
        decipher(bytes(tampered), b64key)