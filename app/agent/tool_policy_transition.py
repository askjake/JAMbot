"""Pure checkpoint transition for one model-facing tool binding.

This module intentionally has no LangChain or graph dependency.  It makes the
activation/checkpoint contract regression-testable without constructing a
model, a graph, or an MCP connection.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.agent.tool_policy_state import (
    FIELD_BOUNDS,
    MAX_LAST_BOUND_TOOL_NAMES,
    TOOL_POLICY_VERSION,
    classify_activation_state,
    compute_policy_signature,
    current_registry_generation,
    normalize_name_list,
    reduce_activation_status,
)


def build_tool_policy_update(
    *,
    stored: Mapping[str, Any] | None,
    plan: Any,
    authorization_flags: Mapping[str, bool] | None,
    bound_tool_names: Sequence[str],
    activation_request_revision: int = 0,
    processed_activation_tool_call_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a bounded replacement/union update for checkpointed policy state."""
    previous = dict(stored or {})
    flags = {str(key): bool(value) for key, value in dict(authorization_flags or {}).items()}
    # The normalized plan already contains trusted checkpoint + current-turn +
    # management intent.  Re-unioning raw legacy checkpoint rows here would
    # resurrect phantom requests that the migration layer deliberately removed.
    requested_toolsets = normalize_name_list(
        getattr(plan, "requested_toolsets", ()) or (),
        FIELD_BOUNDS["requested_toolsets"],
    )
    requested = normalize_name_list(
        getattr(plan, "requested_extra_tools", ()) or (),
        FIELD_BOUNDS["requested_extra_tools"],
    )
    pending_registry = set(getattr(plan, "pending_registry_requests", ()) or ())
    unavailable = set(getattr(plan, "unavailable_extra_tools", ()) or ())
    unavailable.update(getattr(plan, "unhealthy_requests", ()) or ())
    requested_set = set(requested)
    exact_pending_registry = sorted(requested_set.intersection(pending_registry))
    exact_unavailable = sorted(requested_set.intersection(unavailable))
    available = sorted(requested_set - set(exact_pending_registry) - set(exact_unavailable))
    available_set = set(available)

    eligible = [
        value for value in (getattr(plan, "eligible_extra_tools", ()) or ())
        if value in available_set
    ]
    pending = [
        value for value in (getattr(plan, "pending_authorization_extra_tools", ()) or ())
        if value in available_set
    ]
    active = normalize_name_list(
        getattr(plan, "candidate_toolsets", ()) or (), FIELD_BOUNDS["active_toolsets"]
    )
    registry_generation = (
        str(getattr(plan, "inventory_signature", "") or "") or current_registry_generation()
    )

    activation: dict[str, str] = {}
    for extra in requested:
        activation[extra] = classify_activation_state(
            extra,
            available=available,
            eligible=eligible,
            pending=pending,
            pending_registry=exact_pending_registry,
            bound_tool_names=bound_tool_names,
            previously_eligible=previous.get("eligible_extra_tools", []),
        )
    active_set = set(active)
    for family in requested_toolsets:
        if family in pending_registry:
            activation[family] = "PENDING_REGISTRY"
        elif family in unavailable:
            activation[family] = "UNAVAILABLE_UPSTREAM"
        elif family in active_set:
            activation[family] = "BOUND"
        else:
            activation[family] = "REQUESTED"

    revision = max(
        int(previous.get("activation_request_revision") or 0),
        int(activation_request_revision or 0),
    )
    return {
        "active_toolsets": active,
        "requested_toolsets": requested_toolsets,
        "requested_extra_tools": requested,
        "processed_activation_tool_call_ids": normalize_name_list(
            [
                *previous.get("processed_activation_tool_call_ids", []),
                *processed_activation_tool_call_ids,
            ],
            FIELD_BOUNDS["processed_activation_tool_call_ids"],
        ),
        "eligible_extra_tools": normalize_name_list(eligible, FIELD_BOUNDS["eligible_extra_tools"]),
        "pending_authorization_extra_tools": normalize_name_list(
            pending, FIELD_BOUNDS["pending_authorization_extra_tools"]
        ),
        "pending_registry_requests": normalize_name_list(
            exact_pending_registry, FIELD_BOUNDS["pending_registry_requests"]
        ),
        "unavailable_extra_tools": normalize_name_list(
            exact_unavailable, FIELD_BOUNDS["unavailable_extra_tools"]
        ),
        "authorization_flags": flags,
        "mcop_children_forbidden": bool(
            getattr(plan, "mcop_children_forbidden", previous.get("mcop_children_forbidden", False))
        ),
        "last_bound_tool_names": normalize_name_list(bound_tool_names, MAX_LAST_BOUND_TOOL_NAMES),
        "last_activation_status": reduce_activation_status(None, activation),
        "activation_request_revision": revision,
        "continuity_task_scope": str(getattr(plan, "continuity_task_scope", "") or "")[:128],
        "continuity_methodology": str(getattr(plan, "methodology", "") or "")[:128],
        "continuity_environment": str(getattr(plan, "continuity_environment", "") or "")[:128],
        "continuity_revision": max(
            int(previous.get("continuity_revision") or 0),
            int(getattr(plan, "continuity_revision", 0) or 0),
        ),
        "tool_profile_signature": compute_policy_signature(
            active_toolsets=active,
            requested_toolsets=requested_toolsets,
            requested_extra_tools=requested,
            eligible_extra_tools=eligible,
            activation_request_revision=revision,
            registry_generation=registry_generation,
            authorization_flags=flags,
            mcop_children_forbidden=bool(
                getattr(plan, "mcop_children_forbidden", False)
            ),
        ),
        "tool_registry_generation": registry_generation,
        "tool_policy_version": TOOL_POLICY_VERSION,
    }
