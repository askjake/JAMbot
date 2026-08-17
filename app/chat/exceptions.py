class ChatException(Exception):
    """Base exception for chat-related errors"""
    pass

class ChatLimitExceededError(ChatException):
    """Raised when a user has reached their maximum allowed chats"""
    pass

class ChatNotFoundError(ChatException):
    """Raised when a requested chat doesn't exist"""
    pass

class NotAuthorizedError(ChatException):
    """Raised when a user is not authorized to access a chat"""
    pass

class VaultAccessError(ChatException):
    """Raised when attempting to access vault chats without vault mode enabled"""
    pass

class NamespaceNotFoundError(ChatException):
    """Raised when a requested namespace doesn't exist"""