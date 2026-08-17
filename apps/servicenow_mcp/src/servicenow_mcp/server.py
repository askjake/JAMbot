#!/usr/bin/env python3
"""
ServiceNow ITSM MCP Server
==========================
A FastMCP server exposing ServiceNow incident, change, and user search as MCP tools.
Integrates with Jakes-agent via streamable_http transport.

Start:
  python apps/servicenow_mcp/src/servicenow_mcp/server.py
  uvicorn servicenow_mcp.server:mcp_app --host 127.0.0.1 --port 8095
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from fastmcp import FastMCP

from servicenow_mcp.auth import ServiceNowAuth
from servicenow_mcp.client import ServiceNowClient
from servicenow_mcp.config import get_settings
from servicenow_mcp.tools.incidents import search_incidents, get_incident
from servicenow_mcp.tools.changes import search_changes, get_change
from servicenow_mcp.tools.users import search_users
from servicenow_mcp.tools.records import query_table

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

# ── Config ─────────────────────────────────────────────────────────────────────
settings = get_settings()
PORT = int(os.getenv("SNOW_MCP_PORT", str(settings.snow_mcp_port)))
HOST = os.getenv("SNOW_MCP_HOST", "127.0.0.1")

# ── Auth & client (lazy — built on first tool call) ───────────────────────────
_auth: Optional[ServiceNowAuth] = None
_client: Optional[ServiceNowClient] = None


def _get_client() -> ServiceNowClient:
    global _auth, _client
    if _client is None:
        _auth = ServiceNowAuth(
            instance_url=settings.snow_instance_url,
            auth_mode=settings.snow_auth_mode,
            client_id=settings.snow_client_id,
            client_secret=settings.snow_client_secret,
            username=settings.snow_username,
            password=settings.snow_password,
        )
        _client = ServiceNowClient(_auth)
    return _client


# ── FastMCP app ────────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="servicenow-mcp",
    instructions=(
        "ServiceNow ITSM tools for DISH Network. "
        "Search and retrieve incidents, change requests, and users from ServiceNow."
    ),
)


@mcp.tool()
async def snow_search_incidents(
    query: str,
    limit: int = 10,
    state: str = "",
) -> list[dict]:
    """Search ServiceNow incidents by keyword. Returns sys_id, number, short_description, state, priority, assigned_to, opened_at, updated_at."""
    return await search_incidents(_get_client(), query, limit=limit, state=state)


@mcp.tool()
async def snow_get_incident(number: str) -> dict:
    """Fetch a single ServiceNow incident by INC number (e.g. INC0012345)."""
    return await get_incident(_get_client(), number)


@mcp.tool()
async def snow_search_changes(
    query: str,
    limit: int = 10,
    state: str = "",
) -> list[dict]:
    """Search ServiceNow change requests by keyword."""
    return await search_changes(_get_client(), query, limit=limit, state=state)


@mcp.tool()
async def snow_get_change(number: str) -> dict:
    """Fetch a single ServiceNow change request by CHG number (e.g. CHG0012345)."""
    return await get_change(_get_client(), number)


@mcp.tool()
async def snow_search_users(query: str, limit: int = 10) -> list[dict]:
    """Search ServiceNow users by name or email."""
    return await search_users(_get_client(), query, limit=limit)


@mcp.tool()
async def snow_query_table(
    table: str,
    query: str,
    fields: str = "",
    limit: int = 20,
) -> list[dict]:
    """Generic ServiceNow table query. Specify table name and sysparm_query string."""
    return await query_table(_get_client(), table, query, fields=fields, limit=limit)


# ── ASGI app (for uvicorn) ─────────────────────────────────────────────────────
mcp_app = mcp.http_app(path="/mcp")


def main() -> None:
    import uvicorn
    logger.info("Starting ServiceNow MCP server on %s:%s", HOST, PORT)
    uvicorn.run(mcp_app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
