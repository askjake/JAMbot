from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx


class ServiceNowAuthError(Exception):
    """Raised when ServiceNow authentication fails."""

    def __init__(self, message: str, response_body: str = "") -> None:
        super().__init__(message)
        self.response_body = response_body


class ServiceNowAuth:
    """
    Handles ServiceNow authentication.

    Supports two modes:
      - "oauth"  : OAuth 2.0 Client Credentials flow
      - "basic"  : HTTP Basic Auth (username + password)

    The optional *_http_client* parameter allows injecting an httpx.AsyncClient
    for testing (e.g. via respx).
    """

    def __init__(
        self,
        instance_url: str,
        auth_mode: str = "oauth",
        client_id: str = "",
        client_secret: str = "",
        username: str = "",
        password: str = "",
        _http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.instance_url = instance_url.rstrip("/")
        self.auth_mode = auth_mode.lower()
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self._http_client = _http_client

        # Token cache (OAuth only)
        self._token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    async def get_token(self) -> str:
        """Return a valid auth token/credential string."""
        if self.auth_mode == "basic":
            return self._basic_credential()
        # OAuth
        if self._token is None or self._is_token_expired():
            await self._fetch_token()
        return self._token  # type: ignore[return-value]

    def get_auth_header_scheme(self) -> str:
        """Return the Authorization header scheme ('Bearer' or 'Basic')."""
        if self.auth_mode == "basic":
            return "Basic"
        return "Bearer"

    # ── Private helpers ────────────────────────────────────────────────────────

    def _basic_credential(self) -> str:
        raw = f"{self.username}:{self.password}"
        return base64.b64encode(raw.encode()).decode()

    def _is_token_expired(self) -> bool:
        if self._token_expiry is None:
            return True
        # Refresh 60 s before actual expiry
        return datetime.now(tz=timezone.utc) >= self._token_expiry - timedelta(seconds=60)

    def _handle_token_response(self, response: httpx.Response) -> None:
        """Parse and store the token from an OAuth token endpoint response."""
        if response.status_code != 200:
            raise ServiceNowAuthError(
                f"Token endpoint returned {response.status_code}",
                response_body=response.text,
            )
        data = response.json()
        if "access_token" not in data:
            raise ServiceNowAuthError(
                "Token response missing 'access_token'",
                response_body=response.text,
            )
        self._token = data["access_token"]
        expires_in = data.get("expires_in", 1800)
        self._token_expiry = datetime.now(tz=timezone.utc) + timedelta(seconds=int(expires_in))

    async def _fetch_token(self) -> None:
        url = f"{self.instance_url}/oauth_token.do"
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        if self._http_client is not None:
            client = self._http_client
            response = await client.post(url, data=payload)
            self._handle_token_response(response)
        else:
            async with httpx.AsyncClient() as client:
                response = await client.post(url, data=payload)
            self._handle_token_response(response)
