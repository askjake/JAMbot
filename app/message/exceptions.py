class MessageException(Exception):
    """Base exception for message-related errors"""
    pass

class NotUserMessageError(MessageException):
    """Raised when user tries to branch a non-user message"""
    pass

class VersionLimitExceededError(MessageException):
    """Raised when user tries to create too many message versions"""

class HasAttachmentsError(MessageException):
    """Raised when user tries to branch from a message with attachments"""

class VersionNotFoundError(MessageException):
    """Raised when user tries to get a nonexistent message version"""

class ChatReadOnlyError(MessageException):
    """Raised when user tries to modify a readonly chat"""