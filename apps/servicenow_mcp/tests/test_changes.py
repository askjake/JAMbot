"""Tests for servicenow_mcp.tools.changes"""
from __future__ import annotations

import httpx
import pytest
import respx

from servicenow_mcp.auth import ServiceNowAuth
from servicenow_mcp.client import ServiceNowClient, ServiceNowClientError
from servicenow_mcp.tools.changes import search_changes, get_change

INSTANCE = "https://dish.service-now.com"


def make_auth() -> ServiceNowAuth:
    return ServiceNowAuth(
        instance_url=INSTANCE,
        auth_mode="basic",
        username="u",
        password="p",
    )


@pytest.mark.asyncio
async def test_search_changes_returns_results():
    auth = make_auth()
    client = ServiceNowClient(auth)
    fake_rows = [
        {
            "sys_id": {"value": "chg001"},
            "number": {"value": "CHG0001234"},
            "short_description": {"value": "network upgrade"},
            "state": {"value": "-1"},
            "type": {"value": "normal"},
            "start_date": {"value": "2024-02-01"},
            "end_date": {"value": "2024-02-02"},
        }
    ]
    with respx.mock(base_url=INSTANCE) as mock:
        mock.get("/api/now/table/change_request").mock(
            return_value=httpx.Response(200, json={"result": fake_rows})
        )
        results = await search_changes(client, "network")
    assert len(results) == 1
    assert results[0]["number"] == "CHG0001234"
    assert results[0]["short_description"] == "network upgrade"


@pytest.mark.asyncio
async def test_search_changes_api_error_raises():
    auth = make_auth()
    client = ServiceNowClient(auth)
    with respx.mock(base_url=INSTANCE) as mock:
        mock.get("/api/now/table/change_request").mock(
            return_value=httpx.Response(403, text="Forbidden")
        )
        with pytest.raises(Exception):
            await search_changes(client, "network")
