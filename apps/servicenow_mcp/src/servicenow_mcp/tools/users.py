from __future__ import annotations

from servicenow_mcp.client import ServiceNowClient

_USER_FIELDS = "sys_id,name,email,department,title"


async def search_users(
    client: ServiceNowClient,
    query: str,
    limit: int = 10,
) -> list[dict]:
    """Search ServiceNow users by name or email."""
    q = f"nameLIKE{query}^ORemailLIKE{query}"
    rows = await client.get(
        "sys_user",
        sysparm_query=q,
        sysparm_fields=_USER_FIELDS,
        sysparm_limit=limit,
    )
    return [_flatten(r) for r in rows]


def _flatten(r: dict) -> dict:
    return {
        "sys_id": _val(r.get("sys_id")),
        "name": _val(r.get("name")),
        "email": _val(r.get("email")),
        "department": _val(r.get("department")),
        "title": _val(r.get("title")),
    }


def _val(v):
    if isinstance(v, dict):
        return v.get("display_value") or v.get("value") or ""
    return v or ""
