class AttachmentException(Exception):  
    """Base exception for attachment-related errors"""  
    pass  

class AttachmentNotFoundError(AttachmentException):  
    """Raised when an attachment doesn't exist"""  
    pass  

class NotAuthorizedError(AttachmentException):  
    """Raised when a user is not authorized to access an attachment"""  
    pass  

class VaultAccessError(AttachmentException):  
    """Raised when attempting to access vault attachments without vault mode enabled"""  
    pass