from typing import Annotated, Optional

from fastapi import Header, HTTPException, status

from app.config import get_settings

settings = get_settings()


async def get_user_email(
    x_auth_request_email: Annotated[Optional[str], Header()] = None,
) -> str:
    """
    Get user email from request header or return default if auth is disabled.
    
    In production with oauth2-proxy, the X-Auth-Request-Email header is injected.
    In development or testing, AUTH_DISABLED can be set to True to bypass auth.
    
    Priority:
    1. If X-Auth-Request-Email header is present, use it (real Okta user)
    2. If AUTH_DISABLED is true and no header, use default (for local dev/testing)
    3. Otherwise, raise 401
    """
    # If header is present, always use it (real authenticated user)
    if x_auth_request_email is not None:
        return x_auth_request_email.lower()
    
    # If auth is disabled and no header, return default email
    if settings.AUTH_DISABLED:
        return settings.DEFAULT_USER_EMAIL.lower()
    
    # Otherwise, require the header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing X-Auth-Request-Email header",
    )
