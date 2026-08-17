"""Server-owned hidden runtime context for tool-policy management facades.

The management facades must report *actual* current thread state, not a
hypothetical recommendation.  They are read-only tools, so they cannot receive
graph state as an argument without exposing it to model editing.

Propagation note
----------------
A plain ContextVar set *inside* a graph node does not reach sibling node tasks,
because LangGraph creates each node task from the parent context.  The repo
already solves this for audit correlation with ``AuditRequestState``, which is
bound once per HTTP request in middleware and is mutable and shared by
reference.  This module publishes the policy snapshot onto that same
per-request object so the facade, which executes in the *tools* node, observes
what the *agent* node computed.  A ContextVar fallback keeps non-HTTP
entrypoints (tests, cron) working.

The snapshot is bounded and redacted and is never a tool parameter.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Mapping

from app.agent.tool_policy_state import (
    ACTIVATION_STATES,
    FIELD_BOUNDS,
    MAX_ACTIVATION_STATUS_ENTRIES,
    MAX_LAST_BOUND_TOOL_NAMES,
    TOOL_POLICY_STATE_SCHEMA,
    TOOL_POLICY_VERSION,
    normalize_name_list,
)
from app.agent.tool_profiles import normalize_authorization_flags

POLICY_RUNTIME_SCHEMA = "diship_tool_policy_runtime.v1"

_policy_runtime_var: ContextVar[dict[str, Any] | None] = ContextVar(
    "diship_tool_policy_runtime", default=None
)


def build_policy_runtime_snapshot(
    policy_state: Mapping[str, Any] | None,
    *,
    methodology: str = "",
    thread_scoped: bool = True,
) -> dict[str, Any]:
    """Build the bounded, redacted snapshot the facades may report."""
    state = policy_state or {}
    snapshot: dict[str, Any] = {
        "schema": POLICY_RUNTIME_SCHEMA,
        "policy_state_schema": TOOL_POLICY_STATE_SCHEMA,
        "tool_policy_version": int(state.get("tool_policy_version") or TOOL_POLICY_VERSION),
        "methodology": str(methodology or "")[:128],
        "thread_scoped": bool(thread_scoped),
        "authorization_flags": normalize_authorization_flags(state.get("authorization_flags")),
        "tool_profile_signature": str(state.get("tool_profile_signature") or "")[:128],
        "tool_registry_generation": str(state.get("tool_registry_generation") or "")[:128],
        "activation_request_revision": max(0, int(state.get("activation_request_revision") or 0)),
        "activation_status": {},
        "authorization_material_stored": False,
        "mcop_children_forbidden": bool(
            state.get("mcop_children_forbidden", False)
        ),
    }
    for key, bound in (
        ("active_toolsets", FIELD_BOUNDS["active_toolsets"]),
        ("requested_toolsets", FIELD_BOUNDS["requested_toolsets"]),
        ("requested_extra_tools", FIELD_BOUNDS["requested_extra_tools"]),
        ("eligible_extra_tools", FIELD_BOUNDS["eligible_extra_tools"]),
        ("pending_authorization_extra_tools", FIELD_BOUNDS["pending_authorization_extra_tools"]),
        ("pending_registry_requests", FIELD_BOUNDS["pending_registry_requests"]),
        ("unavailable_extra_tools", FIELD_BOUNDS["unavailable_extra_tools"]),
        ("last_bound_tool_names", MAX_LAST_BOUND_TOOL_NAMES),
    ):
        snapshot[key] = normalize_name_list(state.get(key), bound)

    raw_status = state.get("last_activation_status")
    if isinstance(raw_status, Mapping):
        status: dict[str, str] = {}
        for key in sorted(str(k) for k in raw_status):
            if len(status) >= MAX_ACTIVATION_STATUS_ENTRIES:
                break
            value = str(raw_status.get(key) or "")
            if value in ACTIVATION_STATES:
                status[key[:128]] = value
        snapshot["activation_status"] = status
    return snapshot


def publish_policy_runtime_snapshot(snapshot: Mapping[str, Any] | None) -> Token | None:
    """Publish the snapshot for this request.

    Writes onto the per-request ``AuditRequestState`` when one exists so sibling
    graph-node tasks observe it.  Always also sets the ContextVar fallback.
    """
    payload = dict(snapshot) if snapshot else None
    try:
        from app.agent.tool_execution_audit import current_audit_request_state

        state = current_audit_request_state()
        if state is not None:
            state.policy_snapshot = payload
    except Exception:  # noqa: BLE001 - facade state is never fatal
        pass
    return _policy_runtime_var.set(payload)


def reset_policy_runtime_context(token: Token | None) -> None:
    if token is None:
        return
    try:
        _policy_runtime_var.reset(token)
    except (ValueError, LookupError):  # pragma: no cover - defensive
        pass


def current_policy_runtime_context() -> dict[str, Any] | None:
    try:
        from app.agent.tool_execution_audit import current_audit_request_state

        state = current_audit_request_state()
        payload = getattr(state, "policy_snapshot", None) if state is not None else None
        if payload:
            return dict(payload)
    except Exception:  # noqa: BLE001
        pass
    value = _policy_runtime_var.get()
    return dict(value) if value else None


def policy_runtime_context_or_unknown() -> dict[str, Any]:
    """Return the actual snapshot, or an explicit unknown marker.

    An absent snapshot is never presented as a proven-empty policy state.
    """
    snapshot = current_policy_runtime_context()
    if snapshot is None:
        return {
            "schema": POLICY_RUNTIME_SCHEMA,
            "state_known": False,
            "reason": "NO_ACTIVE_PARENT_TURN_CONTEXT",
        }
    snapshot["state_known"] = True
    return snapshot
