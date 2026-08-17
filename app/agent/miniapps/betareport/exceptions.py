class BetaReportException(Exception):
    """Base exception for chat-related errors"""

    pass


class ResourceNotFoundError(BetaReportException):
    """Raised when a requested issue or report doesn't exist"""

    pass
