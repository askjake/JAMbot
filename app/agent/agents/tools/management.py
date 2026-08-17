"""Permanently bound read-only tool-registry management facades."""

from __future__ import annotations

from langchain_core.tools import tool

from app.agent.tool_activation_intent import RegistryToolIndex, activation_intent_from_request


def _csv(value: str) -> list[str]:
    return sorted(set(part.strip() for part in str(value or "").split(",") if part.strip()))


_SUMMARY_LIST_LIMIT = 32
_FAILED_TOOLSET_LIMIT = 20


def _bounded_values(values, limit: int = _SUMMARY_LIST_LIMIT) -> tuple[list, int, bool]:
    rows = list(values or [])
    return rows[:limit], len(rows), len(rows) > limit


def _toolset_summary(status: dict) -> dict:
    toolsets = dict((status or {}).get("toolsets") or {})
    counts: dict[str, int] = {}
    failed: list[dict] = []
    for name, state in sorted(toolsets.items()):
        state = state if isinstance(state, dict) else {}
        current = str(state.get("status") or "unknown")
        counts[current] = counts.get(current, 0) + 1
        if current not in {"loaded", "ok", "healthy"} or state.get("error_code"):
            failed.append({
                "name": name,
                "status": current,
                "error_code": state.get("error_code") or "",
                "error_leaf_codes": list(state.get("error_leaf_codes") or [])[:8],
                "error_summary": str(state.get("error_summary") or "")[:480],
                "health": state.get("health"),
                "tool_count": state.get("tool_count", 0),
                "using_last_known_good": bool(state.get("using_last_known_good", False)),
            })
    return {
        "toolset_count": len(toolsets),
        "toolset_status_counts": dict(sorted(counts.items())),
        "failed_toolsets": failed[:_FAILED_TOOLSET_LIMIT],
        "failed_toolsets_truncated": len(failed) > _FAILED_TOOLSET_LIMIT,
    }


def _compact_policy_state(policy: dict) -> dict:
    policy = dict(policy or {})
    # Preserve the bounded, server-owned policy identity/lifecycle fields that
    # existing callers use to correlate this facade with checkpointed state.
    # The payload reduction must remove bulk, not change the published-state
    # contract.  Keep legacy aliases too when a future/older snapshot supplies
    # them.
    out = {
        key: policy.get(key)
        for key in (
            "schema",
            "policy_state_schema",
            "state_known",
            "reason",
            "methodology",
            "thread_scoped",
            "authorization_flags",
            "authorization_material_stored",
            "authorization_material_logged",
            "tool_profile_signature",
            "tool_registry_generation",
            "tool_registry_content_signature",
            "tool_registry_refresh_epoch",
            "tool_registry_health_signature",
            "tool_policy_version",
            "activation_request_revision",
            "continuity_task_scope",
            "continuity_methodology",
            "continuity_environment",
            "continuity_revision",
            "mcop_children_forbidden",
            "profile_signature",
            "policy_version",
        )
        if key in policy
    }
    for key in (
        "active_toolsets",
        "requested_extra_tools",
        "eligible_extra_tools",
        "pending_authorization_extra_tools",
        "pending_registry_requests",
        "unavailable_extra_tools",
        "requested_toolsets",
        "last_bound_tool_names",
    ):
        if key not in policy:
            continue
        values, count, truncated = _bounded_values(policy.get(key))
        out[key] = values
        out[f"{key}_count"] = count
        out[f"{key}_truncated"] = truncated
    activation = policy.get("activation_status")
    if isinstance(activation, dict):
        items = sorted(activation.items())
        out["activation_status"] = dict(items[:_SUMMARY_LIST_LIMIT])
        out["activation_status_count"] = len(items)
        out["activation_status_truncated"] = len(items) > _SUMMARY_LIST_LIMIT
    return out


def _compact_operational_state(value: dict) -> dict:
    value = dict(value or {})
    out = {
        key: value.get(key)
        for key in (
            "state_known", "workflow", "reason", "activation_status",
            "ssh_environment", "note",
        )
        if key in value
    }
    for key in (
        "requested_operational_tools", "pending_operational_tools",
        "eligible_operational_tools", "unavailable_operational_tools",
        "missing_authorization_grants",
    ):
        if key not in value:
            continue
        rows, count, truncated = _bounded_values(value.get(key))
        out[key] = rows
        out[f"{key}_count"] = count
        out[f"{key}_truncated"] = truncated
    required = value.get("required_authorizations_by_tool")
    if isinstance(required, dict):
        items = sorted(required.items())
        out["required_authorizations_by_tool"] = dict(items[:_SUMMARY_LIST_LIMIT])
        out["required_authorizations_by_tool_count"] = len(items)
        out["required_authorizations_by_tool_truncated"] = len(items) > _SUMMARY_LIST_LIMIT
    return out


@tool("diship_backend_tool_inventory_status")
def diship_backend_tool_inventory_status(scope: str = "summary") -> dict:
    """Return bounded registry status by default; use scope=registry for full detail."""
    from app.agent.agents.tools.registry import get_mcp_registry_status, get_tool_inventory_signature

    if scope not in {"registry", "summary"}:
        return {"ok": False, "result_code": "UNSUPPORTED_SCOPE", "supported_scopes": ["summary", "registry"]}
    status = get_mcp_registry_status()
    output = {
        "ok": True,
        "schema": "diship_backend_tool_inventory_status.v2",
        "scope": scope,
        "inventory_signature": get_tool_inventory_signature(),
        "registry_generation": status.get("generation"),
        "registry_content_signature": status.get("content_signature"),
        "registry_refresh_epoch": status.get("refresh_epoch"),
        "registry_health_signature": status.get("health_signature"),
        "canonicalization_version": status.get("canonicalization_version"),
        "family_health": {
            name: state.get("health")
            for name, state in sorted((status.get("families") or {}).items())
        },
        "family_inventory_source": {
            name: state.get("source")
            for name, state in sorted((status.get("families") or {}).items())
        },
        "initialized": status.get("initialized"),
        **_toolset_summary(status),
        "refresh_supported": True,
        "write_performed": False,
    }
    if scope == "registry":
        output["toolsets"] = status.get("toolsets")
        output["detail_warning"] = "Full registry detail requested explicitly; response may be large."
    else:
        output["detail_reader"] = "Call with scope=registry only when per-toolset raw status is required."
    return output


def _operational_workflow_state(policy: dict) -> dict:
    """Report the real operational-workflow activation state.

    D3B2A: assistant prose must never collapse "tool not bound",
    "authorization missing", "SSH identity unavailable", and "repository
    unreachable" into one claim.  These are distinct, separately reported
    server-owned states.
    """
    from app.agent.operational_workflows import (
        operational_tools_in,
        ssh_environment_status,
        workflow_from_requested_tools,
    )
    from app.agent.tool_execution_gate import required_authorizations_for_tool

    if not policy.get("state_known"):
        return {
            "state_known": False,
            "workflow": "",
            "reason": policy.get("reason", ""),
            "ssh_environment": ssh_environment_status(),
        }
    requested = list(policy.get("requested_extra_tools") or [])
    ops = list(operational_tools_in(requested))
    flags = policy.get("authorization_flags") or {}
    missing = sorted({
        key
        for name in ops
        for key in required_authorizations_for_tool(name)
        if not flags.get(key)
    })
    pending = list(operational_tools_in(policy.get("pending_authorization_extra_tools") or []))
    eligible = list(operational_tools_in(policy.get("eligible_extra_tools") or []))
    unavailable = list(operational_tools_in(policy.get("unavailable_extra_tools") or []))
    if not ops:
        status = "NO_OPERATIONAL_WORKFLOW"
    elif missing or pending:
        status = "ACTIVATION_REQUESTED_PENDING_AUTHORIZATION"
    else:
        status = "OPERATIONAL_TOOLS_ELIGIBLE"
    return {
        "state_known": True,
        "workflow": workflow_from_requested_tools(requested),
        "requested_operational_tools": sorted(ops),
        "pending_operational_tools": sorted(pending),
        "eligible_operational_tools": sorted(eligible),
        "unavailable_operational_tools": sorted(unavailable),
        "missing_authorization_grants": missing,
        "required_authorizations_by_tool": {
            name: list(required_authorizations_for_tool(name)) for name in sorted(ops)
        },
        "activation_status": status,
        "ssh_environment": ssh_environment_status(),
        "note": (
            "Tool binding state is not evidence about SSH identity or repository reachability."
        ),
    }


@tool("diship_backend_tool_binding_status")
def diship_backend_tool_binding_status(methodology: str = "", detail: str = "summary") -> dict:
    """Report actual tool-policy state with a bounded summary by default."""
    from app.agent.agents.tools.registry import get_mcp_registry_status, get_tool_inventory_signature
    from app.agent.tool_policy_runtime import policy_runtime_context_or_unknown

    detail = str(detail or "summary").lower()
    if detail not in {"summary", "full"}:
        return {"ok": False, "result_code": "UNSUPPORTED_DETAIL", "supported_detail": ["summary", "full"]}
    status = get_mcp_registry_status()
    policy = policy_runtime_context_or_unknown()
    operational = _operational_workflow_state(policy)
    return {
        "ok": True,
        "schema": "diship_backend_tool_binding_status.v2",
        "detail": detail,
        "methodology": methodology or None,
        "inventory_signature": get_tool_inventory_signature(),
        "registry_generation": status.get("generation"),
        "registry_content_signature": status.get("content_signature"),
        "registry_refresh_epoch": status.get("refresh_epoch"),
        "registry_health_signature": status.get("health_signature"),
        "canonicalization_version": status.get("canonicalization_version"),
        "family_health": {
            name: state.get("health")
            for name, state in sorted((status.get("families") or {}).items())
        },
        "family_inventory_source": {
            name: state.get("source")
            for name, state in sorted((status.get("families") or {}).items())
        },
        "thread_state_status": (
            "ACTUAL_CHECKPOINTED_THREAD_STATE"
            if policy.get("state_known")
            else "UNKNOWN_NO_ACTIVE_PARENT_TURN"
        ),
        "tool_policy_state": policy if detail == "full" else _compact_policy_state(policy),
        "operational_workflow_state": operational if detail == "full" else _compact_operational_state(operational),
        "detail_reader": "Call with detail=full only when complete checkpointed policy state is required.",
        "note": "This facade does not accept mutable graph state as model arguments.",
        "write_performed": False,
    }


def _actual_activation_status(extras: list[str]) -> dict:
    """Report the real per-tool activation state from checkpointed state."""
    from app.agent.tool_policy_runtime import policy_runtime_context_or_unknown

    policy = policy_runtime_context_or_unknown()
    if not policy.get("state_known"):
        return {"state_known": False, "reason": policy.get("reason", "")}
    statuses = policy.get("activation_status") or {}
    return {
        "state_known": True,
        "by_tool": {name: statuses.get(name, "REQUESTED") for name in extras},
    }


def _required_authorizations_by_tool(extras: list[str]) -> dict:
    """Report the authorizations each requested tool actually requires.

    D3B0: the facade must not understate requirements.  An arbitrary
    code-execution tool has to advertise its privileged requirements here, from
    the same shared classifier the execution gate uses.
    """
    from app.agent.tool_execution_gate import required_authorizations_for_tool

    return {name: list(required_authorizations_for_tool(name)) for name in extras}


def _code_execution_tools(extras: list[str]) -> list[str]:
    """Report which requested tools are in the arbitrary-code-execution class."""
    from app.agent.tool_profiles import is_code_execution_tool

    return [name for name in extras if is_code_execution_tool(name)]


@tool("diship_backend_activate_tool_binding")
def diship_backend_activate_tool_binding(tool_names: str = "", toolsets: str = "") -> dict:
    """Record an additive tool-activation *request*. This does not activate anything.

    The returned ``activation_status`` is the only authoritative statement about
    activation state. A request is not a submission, an activation, or an
    execution. No authorization token is accepted, produced, or echoed.
    """
    registry = RegistryToolIndex.live()
    intent = activation_intent_from_request(
        requested_toolsets=_csv(toolsets),
        requested_tools=_csv(tool_names),
        registry_index=registry,
    )
    extras = list(intent.requested_exact_tools)
    requested_toolsets = list(intent.requested_toolsets)
    if intent.ambiguous_requests:
        status = "AMBIGUOUS_TOOL_NAME"
    elif intent.pending_registry_requests:
        status = "PENDING_REGISTRY"
    elif intent.unavailable_requests or intent.unhealthy_requests:
        status = "UNAVAILABLE_UPSTREAM"
    elif extras or requested_toolsets:
        status = "REQUESTED"
    else:
        status = "NO_VALID_REQUESTS"
    return {
        "ok": not bool(intent.ambiguous_requests),
        "schema": "diship_backend_tool_activation_request.v1",
        "requested_extra_tools": extras,
        "requested_toolsets": requested_toolsets,
        "activation_mode": "ADDITIVE_THREAD_STICKY",
        "activation_status": status,
        "actual_activation_status": _actual_activation_status(extras),
        "required_authorizations_by_tool": _required_authorizations_by_tool(extras),
        "code_execution_tools_requested": _code_execution_tools(extras),
        "pending_registry_requests": list(intent.pending_registry_requests),
        "unavailable_requests": list(intent.unavailable_requests),
        "unhealthy_requests": list(intent.unhealthy_requests),
        "ambiguous_requests": {
            key: list(value) for key, value in sorted(intent.ambiguous_requests.items())
        },
        "request_recorded": False,
        "recording_deferred_to_orchestration_hook": True,
        "activation_performed": False,
        "execution_performed": False,
        "authorization_material_accepted": False,
        "state_change_required": bool(extras or requested_toolsets),
        "write_performed": False,
        "note": (
            "This facade is request-only. An executed ToolMessage is merged by "
            "the orchestration checkpoint hook; this result alone is not a commit."
        ),
    }


@tool("diship_backend_tool_refresh_status")
def diship_backend_tool_refresh_status(detail: str = "summary") -> dict:
    """Return bounded refresh/discovery state without refreshing anything."""
    from app.agent.agents.tools.registry import get_mcp_registry_status

    detail = str(detail or "summary").lower()
    if detail not in {"summary", "full"}:
        return {"ok": False, "result_code": "UNSUPPORTED_DETAIL", "supported_detail": ["summary", "full"]}
    status = get_mcp_registry_status()
    output = {
        "ok": True,
        "schema": "diship_backend_tool_refresh_status.v2",
        "detail": detail,
        "refresh_supported": True,
        "refresh_requires_server_side_call": True,
        "generation": status.get("generation"),
        "initialized_at": status.get("initialized_at"),
        **_toolset_summary(status),
        "write_performed": False,
    }
    if detail == "full":
        output["toolsets"] = status.get("toolsets")
        output["detail_warning"] = "Full toolset discovery state requested explicitly; response may be large."
    else:
        output["detail_reader"] = "Call with detail=full only when complete per-toolset discovery state is required."
    return output


BACKEND_MANAGEMENT_FACADES = [
    diship_backend_tool_inventory_status,
    diship_backend_tool_binding_status,
    diship_backend_activate_tool_binding,
    diship_backend_tool_refresh_status,
]
