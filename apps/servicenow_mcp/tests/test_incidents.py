"""Tests for servicenow_mcp.tools.incidents"""
from __future__ import annotations

import httpx
import pytest
import respx

from servicenow_mcp.auth import ServiceNowAuth
from servicenow_mcp.client import ServiceNowClient, ServiceNowClientError
from servicenow_mcp.tools.incidents import search_incidents, get_incident

INSTANCE = "https://dish.service-now.com"


def make_auth() -> ServiceNowAuth:
    return ServiceNowAuth(
        instance_url=INSTANCE,
        auth_mode="basic",
        username="u",
        password="p",
    )


@pytest.mark.asyncio
async def test_search_incidents_returns_results():
    auth = make_auth()
    client = ServiceNowClient(auth)
    fake_rows = [
        {
            "sys_id": {"value": "abc123"},
            "number": {"value": "INC001"},
            "short_description": {"value": "disk full"},
            "state": {"value": "1"},
            "priority": {"value": "2"},
            "assigned_to": {"value": "joe"},
            "opened_at": {"value": "2024-01-01"},
            "sys_updated_on": {"value": "2024-01-02"},
        }
    ]
    with respx.mock(base_url=INSTANCE) as mock:
        mock.get("/api/now/table/incident").mock(
            return_value=httpx.Response(200, json={"result": fake_rows})
        )
        results = await search_incidents(client, "disk")
    assert len(results) == 1
    assert results[0]["number"] == "INC001"
    assert results[0]["short_description"] == "disk full"


@pytest.mark.asyncio
async def test_search_incidents_api_error_raises():
    auth = make_auth()
    client = ServiceNowClient(auth)
    with respx.mock(base_url=INSTANCE) as mock:
        mock.get("/api/now/table/incident").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )
        with pytest.raises(Exception):
            await search_incidents(client, "disk")
