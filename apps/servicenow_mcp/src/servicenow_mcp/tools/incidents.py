from __future__ import annotations

from typing import Optional

from servicenow_mcp.client import ServiceNowClient

_INCIDENT_FIELDS = (
    "sys_id,number,short_description,state,priority,assigned_to,opened_at,sys_updated_on"
)


async def search_incidents(
    client: ServiceNowClient,
    query: str,
    limit: int = 10,
    state: str = "",
) -> list[dict]:
    """Search incidents by keyword across short_description, number, and description."""
    q = f"short_descriptionLIKE{query}^ORnumberLIKE{query}^ORdescriptionLIKE{query}"
    if state:
        q += f"^state={state}"
    rows = await client.get(
        "incident",
        sysparm_query=q,
        sysparm_fields=_INCIDENT_FIELDS,
        sysparm_limit=limit,
    )
    return [_flatten(r) for r in rows]


async def get_incident(client: ServiceNowClient, number: str) -> dict:
    """Fetch a single incident by INC number."""
    rows = await client.get(
        "incident",
        sysparm_query=f"number={number}",
        sysparm_limit=1,
    )
    if not rows:
        return {}
    return rows[0]


def _flatten(r: dict) -> dict:
    return {
        "sys_id": _val(r.get("sys_id")),
        "number": _val(r.get("number")),
        "short_description": _val(r.get("short_description")),
        "state": _val(r.get("state")),
        "priority": _val(r.get("priority")),
        "assigned_to": _val(r.get("assigned_to")),
        "opened_at": _val(r.get("opened_at")),
        "updated_at": _val(r.get("sys_updated_on")),
    }


def _val(v):
    if isinstance(v, dict):
        return v.get("display_value") or v.get("value") or ""
    return v or ""
