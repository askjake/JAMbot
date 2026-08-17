"""
Shared pytest fixtures for servicenow_mcp tests.
"""
import pytest

INSTANCE_URL = "https://dish.service-now.com"
FAKE_CLIENT_ID = "test-client-id"
FAKE_CLIENT_SECRET = "test-client-secret"
FAKE_TOKEN = "fake-access-token-abc123"


def make_token_payload(token: str = FAKE_TOKEN, expires_in: int = 1800) -> dict:
    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": expires_in,
        "scope": "useraccount",
    }
