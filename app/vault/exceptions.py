class VaultException(Exception):
    """Base class for vault-related errors"""
    pass

class VaultNotExistError(VaultException):
    """Raised when user tries to use vault function without vault setup."""
    pass

class VaultAlreadyExistsError(VaultException):
    """Raised when user tries to re-setup their vault
    without deleting their existing vault."""
    pass

class InvalidHashException(VaultException):
    """Raised when password hashes doesn't match"""
    pass

class InvalidVaultCookieError(VaultException):
    """Raised when an error occurs during vault session cookie decyption and parsing"""