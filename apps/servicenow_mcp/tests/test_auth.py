"""
Tests for servicenow_mcp.auth — ServiceNowAuth

Tests 1-4 inject an httpx.AsyncClient so respx can intercept the OAuth token call.
Test 5 exercises Basic Auth (no HTTP needed).
"""
from __future__ import annotations

import base64

import httpx
import pytest
import respx

from servicenow_mcp.auth import ServiceNowAuth, ServiceNowAuthError
from tests.conftest import (
    FAKE_CLIENT_ID,
    FAKE_CLIENT_SECRET,
    FAKE_TOKEN,
    INSTANCE_URL,
    make_token_payload,
)


# ── Test 1: happy-path token fetch ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_token_fetches_and_returns_token():
    async with httpx.AsyncClient() as injected_client:
        with respx.mock(base_url=INSTANCE_URL) as mock:
            mock.post("/oauth_token.do").mock(
                return_value=httpx.Response(200, json=make_token_payload())
            )
            auth = ServiceNowAuth(
                instance_url=INSTANCE_URL,
                auth_mode="oauth",
                client_id=FAKE_CLIENT_ID,
                client_secret=FAKE_CLIENT_SECRET,
                _http_client=injected_client,
            )
            token = await auth.get_token()
    assert token == FAKE_TOKEN


# ── Test 2: token is cached on second call ─────────────────────────────────────

@pytest.mark.asyncio
async def test_get_token_caches_token():
    async with httpx.AsyncClient() as injected_client:
        with respx.mock(base_url=INSTANCE_URL) as mock:
            route = mock.post("/oauth_token.do").mock(
                return_value=httpx.Response(200, json=make_token_payload())
            )
            auth = ServiceNowAuth(
                instance_url=INSTANCE_URL,
                auth_mode="oauth",
                client_id=FAKE_CLIENT_ID,
                client_secret=FAKE_CLIENT_SECRET,
                _http_client=injected_client,
            )
            token1 = await auth.get_token()
            token2 = await auth.get_token()
    # Token endpoint should only be called once
    assert route.call_count == 1
    assert token1 == token2 == FAKE_TOKEN


# ── Test 3: HTTP 401 raises ServiceNowAuthError ────────────────────────────────

@pytest.mark.asyncio
async def test_get_token_raises_on_http_error():
    async with httpx.AsyncClient() as injected_client:
        with respx.mock(base_url=INSTANCE_URL) as mock:
            mock.post("/oauth_token.do").mock(
                return_value=httpx.Response(401, text="Unauthorized")
            )
            auth = ServiceNowAuth(
                instance_url=INSTANCE_URL,
                auth_mode="oauth",
                client_id=FAKE_CLIENT_ID,
                client_secret=FAKE_CLIENT_SECRET,
                _http_client=injected_client,
            )
            with pytest.raises(ServiceNowAuthError) as exc_info:
                await auth.get_token()
    assert "401" in str(exc_info.value)


# ── Test 4: missing access_token key raises ServiceNowAuthError ────────────────

@pytest.mark.asyncio
async def test_get_token_raises_on_missing_access_token():
    async with httpx.AsyncClient() as injected_client:
        with respx.mock(base_url=INSTANCE_URL) as mock:
            mock.post("/oauth_token.do").mock(
                return_value=httpx.Response(200, json={"token_type": "Bearer"})
            )
            auth = ServiceNowAuth(
                instance_url=INSTANCE_URL,
                auth_mode="oauth",
                client_id=FAKE_CLIENT_ID,
                client_secret=FAKE_CLIENT_SECRET,
                _http_client=injected_client,
            )
            with pytest.raises(ServiceNowAuthError) as exc_info:
                await auth.get_token()
    assert "access_token" in str(exc_info.value)


# ── Test 5: Basic Auth returns base64-encoded credential ──────────────────────

@pytest.mark.asyncio
async def test_basic_auth_returns_encoded_header():
    auth = ServiceNowAuth(
        instance_url=INSTANCE_URL,
        auth_mode="basic",
        username="admin",
        password="s3cr3t",
    )
    expected = base64.b64encode(b"admin:s3cr3t").decode()
    token = await auth.get_token()
    assert token == expected
    assert auth.get_auth_header_scheme() == "Basic"
