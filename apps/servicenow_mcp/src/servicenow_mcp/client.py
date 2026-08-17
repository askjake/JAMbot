from __future__ import annotations

import httpx

from servicenow_mcp.auth import ServiceNowAuth, ServiceNowAuthError


class ServiceNowClientError(Exception):
    """Raised when a ServiceNow table API call fails."""


class ServiceNowClient:
    """Thin async HTTP client for the ServiceNow Table API."""

    def __init__(self, auth: ServiceNowAuth) -> None:
        self.auth = auth

    async def get(
        self,
        table: str,
        *,
        sysparm_query: str = "",
        sysparm_fields: str = "",
        sysparm_limit: int = 20,
    ) -> list[dict]:
        """Query a ServiceNow table and return the result rows."""
        token = await self.auth.get_token()
        scheme = self.auth.get_auth_header_scheme()
        headers = {
            "Authorization": f"{scheme} {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        params: dict = {"sysparm_limit": str(sysparm_limit)}
        if sysparm_query:
            params["sysparm_query"] = sysparm_query
        if sysparm_fields:
            params["sysparm_fields"] = sysparm_fields

        url = f"{self.auth.instance_url}/api/now/table/{table}"
        async with httpx.AsyncClient() as http:
            resp = await http.get(url, headers=headers, params=params)

        if resp.status_code != 200:
            raise ServiceNowClientError(
                f"ServiceNow table API returned {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json().get("result", [])
