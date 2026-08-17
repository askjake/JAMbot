from __future__ import annotations

from servicenow_mcp.client import ServiceNowClient


async def query_table(
    client: ServiceNowClient,
    table: str,
    query: str,
    fields: str = "",
    limit: int = 20,
) -> list[dict]:
    """Generic ServiceNow table query. Returns raw result rows."""
    return await client.get(
        table,
        sysparm_query=query,
        sysparm_fields=fields,
        sysparm_limit=limit,
    )
