class ReleaseException(Exception):
    """Base exception for chat-related errors"""
    pass


class ReleaseNotFoundError(ReleaseException):
    """Raised when a requested chat doesn't exist"""
    pass