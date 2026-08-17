from __future__ import annotations

from servicenow_mcp.client import ServiceNowClient

_CHANGE_FIELDS = (
    "sys_id,number,short_description,state,type,start_date,end_date"
)


async def search_changes(
    client: ServiceNowClient,
    query: str,
    limit: int = 10,
    state: str = "",
) -> list[dict]:
    """Search change requests by keyword across short_description and number."""
    q = f"short_descriptionLIKE{query}^ORnumberLIKE{query}"
    if state:
        q += f"^state={state}"
    rows = await client.get(
        "change_request",
        sysparm_query=q,
        sysparm_fields=_CHANGE_FIELDS,
        sysparm_limit=limit,
    )
    return [_flatten(r) for r in rows]


async def get_change(client: ServiceNowClient, number: str) -> dict:
    """Fetch a single change request by CHG number."""
    rows = await client.get(
        "change_request",
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
        "type": _val(r.get("type")),
        "start_date": _val(r.get("start_date")),
        "end_date": _val(r.get("end_date")),
    }


def _val(v):
    if isinstance(v, dict):
        return v.get("display_value") or v.get("value") or ""
    return v or ""
