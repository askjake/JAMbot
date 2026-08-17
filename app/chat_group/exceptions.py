class ChatGroupException(Exception):
    """Base exception for group-related errors"""
    pass

class ChatGroupLimitExceededError(ChatGroupException):
    """Raised when a user has reached their maximum allowed groups"""
    pass

class ChatGroupNotFoundError(ChatGroupException):
    """Raised when a requested group doesn't exist"""
    pass

class ChatGroupNotAuthorizedError(ChatGroupException):
    """Raised when a user is not authorized to access a group"""
    pass