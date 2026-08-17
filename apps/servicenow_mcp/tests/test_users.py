"""Tests for servicenow_mcp.tools.users"""
from __future__ import annotations

import httpx
import pytest
import respx

from servicenow_mcp.auth import ServiceNowAuth
from servicenow_mcp.client import ServiceNowClient, ServiceNowClientError
from servicenow_mcp.tools.users import search_users

INSTANCE = "https://dish.service-now.com"


def make_auth() -> ServiceNowAuth:
    return ServiceNowAuth(
        instance_url=INSTANCE,
        auth_mode="basic",
        username="u",
        password="p",
    )


@pytest.mark.asyncio
async def test_search_users_returns_results():
    auth = make_auth()
    client = ServiceNowClient(auth)
    fake_rows = [
        {
            "sys_id": {"value": "usr001"},
            "name": {"value": "Alice Smith"},
            "email": {"value": "alice.smith@dish.com"},
            "department": {"value": "Engineering"},
            "title": {"value": "SRE"},
        }
    ]
    with respx.mock(base_url=INSTANCE) as mock:
        mock.get("/api/now/table/sys_user").mock(
            return_value=httpx.Response(200, json={"result": fake_rows})
        )
        results = await search_users(client, "alice")
    assert len(results) == 1
    assert results[0]["name"] == "Alice Smith"
    assert results[0]["email"] == "alice.smith@dish.com"


@pytest.mark.asyncio
async def test_search_users_api_error_raises():
    auth = make_auth()
    client = ServiceNowClient(auth)
    with respx.mock(base_url=INSTANCE) as mock:
        mock.get("/api/now/table/sys_user").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        with pytest.raises(Exception):
            await search_users(client, "alice")
