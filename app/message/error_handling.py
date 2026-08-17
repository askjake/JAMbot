"""
Enhanced error handling patch for message service.
This provides better recovery from AWS token errors and other transient failures.
"""
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def is_recoverable_error(exc: Exception) -> bool:
    """
    Determine if an error is recoverable (shouldn't set chat to readonly).
    
    Args:
        exc: Exception to check
        
    Returns:
        True if error is recoverable
    """
    # AWS token expiration is recoverable (will be retried)
    if isinstance(exc, ClientError):
        error_code = exc.response.get('Error', {}).get('Code', '')
        if error_code in ['ExpiredTokenException', 'TokenRefreshRequired']:
            return True
        # Throttling errors are also recoverable
        if error_code in ['ThrottlingException', 'TooManyRequestsException']:
            return True
    
    # Network timeouts are recoverable
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    
    # Message too long errors (these will be handled by compression)
    error_msg = str(exc).lower()
    if any(keyword in error_msg for keyword in [
        'too long', 'token limit', 'context length', 'maximum context'
    ]):
        return True
    
    return False

def get_user_friendly_error_message(exc: Exception) -> str:
    """
    Convert technical errors into user-friendly messages.
    
    Args:
        exc: Exception to convert
        
    Returns:
        User-friendly error message
    """
    if isinstance(exc, ClientError):
        error_code = exc.response.get('Error', {}).get('Code', '')
        
        if error_code == 'ExpiredTokenException':
            return (
                "AWS credentials expired. The system will automatically retry with fresh credentials. "
                "Please send your message again if this persists."
            )
        elif error_code in ['ThrottlingException', 'TooManyRequestsException']:
            return (
                "The AI service is experiencing high demand. "
                "Please wait a moment and try again."
            )
        elif error_code == 'ValidationException':
            return (
                "Your message couldn't be processed. "
                "Please try rephrasing or breaking it into smaller parts."
            )
    
    error_msg = str(exc).lower()
    if 'token limit' in error_msg or 'context length' in error_msg:
        return (
            "This conversation has grown too long. "
            "The system will automatically compress older messages. Please try again."
        )
    
    # Default message for unrecoverable errors
    return f"An error occurred: {str(exc)}"

# Enhanced error handler to replace _handle_stream_err in MessageService
async def handle_stream_error_enhanced(
    self,
    chat_id: str,
    email: str,
    exc: Exception,
    vault_key: str = "",
    chat_service=None
) -> str:
    """
    Enhanced error handler that doesn't set readonly for recoverable errors.
    
    Args:
        chat_id: ID of the chat
        email: User email
        exc: Exception that occurred
        vault_key: Vault encryption key if applicable
        chat_service: ChatService instance
        
    Returns:
        User-friendly error message
    """
    from app.db import get_db_session_ctxmgr
    
    logger.error(f"Stream error in chat {chat_id}: {type(exc).__name__} - {str(exc)}", exc_info=True)
    
    user_message = get_user_friendly_error_message(exc)
    
    # Only set readonly for non-recoverable errors
    if not is_recoverable_error(exc):
        logger.warning(f"Chat {chat_id}: Setting to readonly due to unrecoverable error")
        async with get_db_session_ctxmgr() as db:
            if chat_service:
                await chat_service.set_readonly(
                    db, email, chat_id, bool(vault_key), user_message
                )
    else:
        logger.info(f"Chat {chat_id}: Recoverable error, not setting readonly. User can retry.")
    
    return user_message
