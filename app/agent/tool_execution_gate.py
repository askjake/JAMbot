"""Independent tool execution authorization and hard-limit gate.

This module is framework-neutral.  It evaluates model-emitted calls against the
exact last model-facing binding and current server-side capability flags before
any executor is invoked.  Blocked calls receive one paired result each.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence

from app.agent.evidence_status import normalize_tool_outcome
from app.agent.identifier_validation import (
    IDENTIFIER_VALIDATION_FAILED,
    validate_tool_arguments,
)
from app.agent.tool_profiles import (
    EXACT_TOOL_CAPABILITIES,
    code_execution_capability,
    is_code_execution_tool,
    normalize_authorization_flags,
)

TOOL_EXECUTION_GATE_SCHEMA = "diship_tool_execution_gate.v1"
TOOL_EXECUTION_GATE_VERSION = "tool_execution_gate.v1"

# Hard ceilings are independent of upstream defaults.  They prevent a stale or
# malicious model call from expanding work beyond the verified source limits.
_TOOL_ARGUMENT_LIMITS: dict[str, dict[str, int | float]] = {
    "build_log_capsule": {
        "max_files": 60,
        "max_lines_per_file": 250_000,
        "max_events": 120_000,
        "top_templates": 80,
        "max_pois": 80,
    },
    "build_complete_log_capsule": {
        "max_files": 60,
        "max_lines_per_file": 250_000,
        "max_events": 120_000,
        "top_templates": 80,
        "max_pois": 80,
    },
    "build_incident_scene": {
        "max_expansion_events": 500,
    },
    "query_incident_scene": {"limit": 500},
    "expand_scene_region": {"max_events": 500, "radius_before": 250, "radius_after": 250},
    "list_incident_scenes": {"limit": 100},
}

_HEAVY_TOOL_NAMES = {
    "build_log_capsule",
    "build_complete_log_capsule",
    "build_incident_scene",
    "get_timeline",
    "create_log_bundle",
    "start_complete_corpus_job",
    "run_complete_corpus_job_batch",
    "run_complete_corpus_job_autopilot",
    "finalize_complete_corpus_job",
    "summarize_log_patterns",
    "quick_analyze",
    "quick_analyze_logs",
    "filter_log_lines",
    "compare_log_capsules",
    "discover_log_formats_and_templates",
    "discover_receiver_metadata",
    "backfill_case_metadata_from_logs",
    "build_baseline_evidence_manifest",
}

# Upstream tools whose own default is persist=True.  For these an omitted
# argument is pinned to the authorization state so nothing writes unbidden.
_PERSIST_DEFAULT_TRUE_TOOLS = {"build_log_capsule"}
# Upstream tools whose own default is persist=False.  Authorization is
# permission, not intent: persistence authorization alone must never convert
# an omitted persist argument into True.  Only an explicit persist=True
# requests persistence for these tools.
_PERSIST_DEFAULT_FALSE_TOOLS = {"build_incident_scene"}
_ALWAYS_PERSIST_TOOLS = {"build_complete_log_capsule", "create_log_bundle"}

# Mutating names are intentionally conservative.  Exact bound-name enforcement
# remains the first boundary; these patterns add authorization checks for
# server-side writes that happen to be model-facing.
_MUTATION_NAME_RE = re.compile(
    r"^(?:persist_|submit_|delete_|cleanup_|commit_|supersede_|regenerate_|materialize_|resolve_human_|reconcile_human_|create_human_review_|request_.*upload|upload_)",
    re.IGNORECASE,
)
_EXTERNAL_UPLOAD_RE = re.compile(r"(?:grasshopper|external).*upload|upload.*(?:grasshopper|external)", re.IGNORECASE)
_MANAGEMENT_REQUEST_EXEMPT = {"diship_backend_activate_tool_binding"}



_MODEL_HIDDEN_ARGUMENTS = {
    "allow_heavy",
    "heavy_auth_token",
    "operator_auth_token",
    "authorization_token",
    "mutation_auth_token",
}


def _sanitized_model_args_schema(args_schema: Any, tool_name: str) -> Any:
    """Return a Pydantic schema without server-controlled authorization fields."""
    fields = getattr(args_schema, "model_fields", None)
    if not isinstance(fields, Mapping):
        return args_schema
    retained = {name: field for name, field in fields.items() if name not in _MODEL_HIDDEN_ARGUMENTS}
    if len(retained) == len(fields):
        return args_schema
    try:
        from pydantic import create_model
        definitions = {
            name: (field.rebuild_annotation(), field)
            for name, field in retained.items()
        }
        digest = hashlib.sha256((tool_name + ":" + ",".join(sorted(retained))).encode()).hexdigest()[:10]
        return create_model(f"ModelFacing_{re.sub(r'[^A-Za-z0-9_]', '_', tool_name)}_{digest}", **definitions)
    except Exception:
        # Refuse to expose sensitive fields if a compatible schema cannot be
        # constructed.  A schema-less clone is safer than advertising tokens.
        return None


def prepare_model_facing_tool(tool: Any) -> Any:
    """Clone a tool for bind_tools() and hide all server-controlled auth args.

    The broad executor registry retains the original tool and schema.  Only the
    model-facing clone is reduced, so server-side rewrites can still inject
    allow_heavy without making tokens or switches model arguments.
    """
    try:
        clone = tool.model_copy(deep=False) if hasattr(tool, "model_copy") else copy.copy(tool)
    except Exception:
        clone = copy.copy(tool)
    original_schema = getattr(tool, "args_schema", None)
    sanitized = _sanitized_model_args_schema(original_schema, str(getattr(tool, "name", "tool")))
    if original_schema is not None and sanitized is None:
        # The caller will omit a tool whose sensitive schema could not be made
        # safe, instead of silently advertising authorization fields.
        return None
    if original_schema is not None:
        setattr(clone, "args_schema", sanitized)
    description = str(getattr(clone, "description", "") or "")
    if original_schema is not None and any(name in getattr(original_schema, "model_fields", {}) for name in _MODEL_HIDDEN_ARGUMENTS):
        note = "[authorization controls are supplied server-side and are not model arguments]"
        setattr(clone, "description", f"{note} {description}".strip())
    return clone

@dataclass(frozen=True)
class FallbackToolMessage:
    """Minimal attribute-compatible ToolMessage used without LangChain installed."""

    content: str
    tool_call_id: str
    name: str = ""
    type: str = "tool"


@dataclass(frozen=True)
class GateDecision:
    schema: str = TOOL_EXECUTION_GATE_SCHEMA
    gate_version: str = TOOL_EXECUTION_GATE_VERSION
    tool_call_id: str = ""
    tool_name: str = ""
    allowed: bool = False
    result_code: str = "TOOL_EXECUTION_BLOCKED"
    original_argument_names: tuple[str, ...] = ()
    effective_args: dict[str, Any] = field(default_factory=dict)
    rewritten_arguments: tuple[str, ...] = ()
    required_authorizations: tuple[str, ...] = ()
    missing_authorizations: tuple[str, ...] = ()
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _call_parts(call: Any) -> tuple[str, str, dict[str, Any]]:
    if isinstance(call, Mapping):
        call_id = str(call.get("id") or call.get("tool_call_id") or "")
        name = str(call.get("name") or "")
        args = call.get("args") or call.get("arguments") or {}
    else:
        call_id = str(getattr(call, "id", "") or getattr(call, "tool_call_id", ""))
        name = str(getattr(call, "name", "") or "")
        args = getattr(call, "args", {}) or getattr(call, "arguments", {}) or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    return call_id, name, dict(args) if isinstance(args, Mapping) else {}


def _raw_name(name: str) -> str:
    value = str(name or "")
    if ":" in value:
        value = value.split(":", 1)[1]
    return value


def _capability_metadata(name: str, args: Mapping[str, Any]) -> dict[str, Any]:
    raw = _raw_name(name)
    exact = None
    for qualified, metadata in EXACT_TOOL_CAPABILITIES.items():
        if qualified.endswith(":" + raw):
            exact = metadata
            break
    required = set(exact.get("required_authorizations", ()) if exact else ())
    capabilities = set(exact.get("capabilities", ()) if exact else ())
    risk = str(exact.get("risk", "read_only") if exact else "read_only")

    # D3B0: arbitrary code execution is never ordinary read-only.  This is the
    # one shared classifier consumed by the parent profile, MCOP child
    # narrowing, and this gate, so a tool cannot be privileged in one place and
    # read-only in another.
    code_exec = code_execution_capability(raw)
    if code_exec is not None:
        required.update(code_exec["required_authorizations"])
        capabilities.update(code_exec["capabilities"])
        capabilities.add("code_execution_classification:" + str(code_exec["classification"]))
        risk = str(code_exec["risk"])

    if raw in _HEAVY_TOOL_NAMES:
        required.add("heavy_tools_authorized")
        risk = "heavy_compute"
    if raw in _ALWAYS_PERSIST_TOOLS:
        required.add("persistence_authorized")
        capabilities.add("persistence_required")
        risk = "heavy_persistent"
    if exact is not None:
        mutating = "mutation_authorized" in required
        external_upload = bool({
            "external_request",
            "log_upload_request",
            "external_upload",
        }.intersection(capabilities))
    else:
        mutating = raw not in _MANAGEMENT_REQUEST_EXEMPT and bool(_MUTATION_NAME_RE.search(raw))
        external_upload = bool(_EXTERNAL_UPLOAD_RE.search(str(name or "")))
        if mutating or external_upload:
            required.update(("operator_authorized", "mutation_authorized"))
            capabilities.add("external_upload" if external_upload else "mutation")
            risk = "mutating"
    if bool(args.get("persist")):
        capabilities.add("persistence_requested")
    return {
        "risk": risk,
        "capabilities": tuple(sorted(capabilities)),
        "required_authorizations": tuple(sorted(required)),
        "mutating": mutating,
        "external_upload": external_upload,
    }


def required_authorizations_for_tool(
    name: str, args: Mapping[str, Any] | None = None
) -> tuple[str, ...]:
    """Public: authorization flags a tool requires before it may execute.

    Exposes the same classification the gate itself uses so callers can
    pre-filter a candidate tool list without duplicating capability logic
    or reaching into private helpers.
    """
    metadata = _capability_metadata(str(name or ""), dict(args or {}))
    return tuple(metadata.get("required_authorizations", ()))


def tool_is_permitted_by_flags(
    name: str, authorization_flags: Mapping[str, Any] | None = None
) -> bool:
    """Public: whether the given flags satisfy every requirement of a tool."""
    flags = dict(authorization_flags or {})
    return all(bool(flags.get(key)) for key in required_authorizations_for_tool(name))


def _bounded_args(name: str, args: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    raw = _raw_name(name)
    effective = dict(args)
    rewritten: list[str] = []
    for key, ceiling in _TOOL_ARGUMENT_LIMITS.get(raw, {}).items():
        if key not in effective:
            continue
        value = effective[key]
        if isinstance(value, bool):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric > float(ceiling):
            effective[key] = int(ceiling) if isinstance(ceiling, int) else ceiling
            rewritten.append(key)
    return effective, tuple(sorted(set(rewritten)))


UPSTREAM_UNAVAILABLE_RESULT_CODE = "BLOCKED_UPSTREAM_UNAVAILABLE"


def _upstream_family_block(tool_name: str) -> tuple[bool, dict[str, Any]]:
    """Report whether a tool belongs to a family whose health forbids execution.

    D3B1: a degraded last-known-good family may keep a stable advertised schema,
    but it must never be called blindly.  Only a bounded health classification
    is returned, never a transport exception body, URL, header, or credential.

    A family with no recorded health state is local and non-MCP, so it is not
    affected by this check.
    """
    try:
        from app.agent.agents.tools.registry import owning_mcp_family
        from app.agent.mcp_registry_health import family_execution_block
    except Exception:  # noqa: BLE001 - additive check must never break binding
        return False, {}
    try:
        family = owning_mcp_family(tool_name)
        if not family:
            return False, {}
        return family_execution_block(family)
    except Exception:  # noqa: BLE001
        return False, {}


def evaluate_tool_call(
    call: Any,
    *,
    last_bound_tool_names: Iterable[str],
    authorization_flags: Mapping[str, Any] | None = None,
    context_identifiers: Mapping[str, Any] | None = None,
    argument_sources: Mapping[str, str] | None = None,
    audit_event_id: str = "",
    execution_constraints: Mapping[str, Any] | None = None,
) -> GateDecision:
    call_id, name, args = _call_parts(call)
    bound = {str(item) for item in last_bound_tool_names if str(item)}
    flags = normalize_authorization_flags(authorization_flags)
    metadata = _capability_metadata(name, args)
    required = tuple(metadata["required_authorizations"])
    missing = tuple(key for key in required if not flags.get(key, False))
    effective, rewritten = _bounded_args(name, args)
    raw_name = _raw_name(name)
    identifier_validation = validate_tool_arguments(
        args,
        context_identifiers=context_identifiers,
        argument_sources=argument_sources,
        tool_call_id=call_id,
        audit_event_id=audit_event_id,
    )

    # Model arguments never carry authorization material.  The gate supplies
    # only the boolean allow_heavy switch after server-side authorization and
    # removes any model-emitted token field without inspecting its value.
    if "heavy_auth_token" in effective:
        effective.pop("heavy_auth_token", None)
        rewritten = tuple(sorted(set((*rewritten, "heavy_auth_token"))))
    if raw_name in _HEAVY_TOOL_NAMES and flags["heavy_tools_authorized"]:
        if effective.get("allow_heavy") is not True:
            effective["allow_heavy"] = True
            rewritten = tuple(sorted(set((*rewritten, "allow_heavy"))))

    # Upstream persistence contracts differ per tool.  Make the effective
    # behavior explicit so an omitted argument can never silently write an
    # artifact, and so authorization alone never implies persistence intent.
    if "persist" not in effective:
        if raw_name in _PERSIST_DEFAULT_TRUE_TOOLS:
            effective["persist"] = bool(flags["persistence_authorized"])
            rewritten = tuple(sorted(set((*rewritten, "persist"))))
        elif raw_name in _PERSIST_DEFAULT_FALSE_TOOLS:
            effective["persist"] = False
            rewritten = tuple(sorted(set((*rewritten, "persist"))))

    result_code = "TOOL_EXECUTION_ALLOWED"
    allowed = True
    from app.agent.execution_constraints import spawn_blocked_by_constraints
    if spawn_blocked_by_constraints(name, execution_constraints):
        allowed = False
        result_code = "BLOCKED_PARENT_ONLY_TURN"
    elif not name or name not in bound:
        allowed = False
        result_code = "TOOL_NOT_IN_LAST_BINDING"
    elif missing:
        allowed = False
        result_code = "TOOL_AUTHORIZATION_REQUIRED"
    elif not identifier_validation.allowed:
        allowed = False
        result_code = IDENTIFIER_VALIDATION_FAILED

    # D3B1: a call into a family whose upstream health is degraded, disabled,
    # invalid, or without a trusted baseline is blocked before any invocation.
    upstream_blocked, upstream_detail = _upstream_family_block(name)
    if allowed and upstream_blocked:
        allowed = False
        result_code = UPSTREAM_UNAVAILABLE_RESULT_CODE

    # Persistence can be safely downgraded without blocking an otherwise
    # authorized heavy read/analysis.  The upstream call sees persist=False.
    if allowed and bool(effective.get("persist")) and not flags["persistence_authorized"]:
        effective["persist"] = False
        rewritten = tuple(sorted(set((*rewritten, "persist"))))
        result_code = "TOOL_EXECUTION_ALLOWED_WITH_REWRITE"

    if allowed and rewritten and result_code == "TOOL_EXECUTION_ALLOWED":
        result_code = "TOOL_EXECUTION_ALLOWED_WITH_REWRITE"

    audit = {
        "schema": "diship_tool_execution_audit.v1",
        "tool_call_id": call_id,
        "tool_name": name,
        "allowed": allowed,
        "result_code": result_code,
        "risk": metadata["risk"],
        "capabilities": list(metadata["capabilities"]),
        "argument_names": sorted(args),
        "rewritten_argument_names": list(rewritten),
        "required_authorizations": list(required),
        "missing_authorizations": list(missing),
        "authorization_flags": flags,
        "argument_values_recorded": False,
        # Bounded upstream health classification only; never an error body.
        "upstream_family": str(upstream_detail.get("upstream_family", "")),
        "upstream_health": str(upstream_detail.get("upstream_health", "")),
        "upstream_source": str(upstream_detail.get("upstream_source", "")),
        "upstream_error_class": str(upstream_detail.get("upstream_error_class", "")),
        "upstream_execution_blocked": bool(upstream_blocked),
        "identifier_validation_status": (
            "PASS" if identifier_validation.allowed else "BLOCKED"
        ),
        "identifier_validations": list(identifier_validation.audit_records),
        "identifier_values_recorded": False,
        "execution_constraints": {
            "mcop_children_forbidden": bool(
                (execution_constraints or {}).get("mcop_children_forbidden", False)
            )
        },
    }
    return GateDecision(
        tool_call_id=call_id,
        tool_name=name,
        allowed=allowed,
        result_code=result_code,
        original_argument_names=tuple(sorted(args)),
        effective_args=effective,
        rewritten_arguments=rewritten,
        required_authorizations=required,
        missing_authorizations=missing,
        audit=audit,
    )


def paired_result_payload(decision: GateDecision, *, result: Any = None, error: str = "") -> dict[str, Any]:
    """Create one compact, serializable tool result for one emitted call."""
    status = normalize_tool_outcome(
        executed=bool(decision.allowed),
        error_type=error or None,
        result_code=("TOOL_EXECUTION_ERROR" if error else decision.result_code),
        result=result,
    )
    status_fields = {
        "result_status": status.result_status,
        "coverage": status.coverage,
        "negative_conclusion_allowed": status.negative_conclusion_allowed,
    }
    if decision.allowed and not error:
        return {
            "ok": True,
            "schema": "diship_gated_tool_result.v1",
            "result_code": decision.result_code,
            "tool_name": decision.tool_name,
            "tool_call_id": decision.tool_call_id,
            "rewritten_arguments": list(decision.rewritten_arguments),
            "result": result,
            **status_fields,
        }
    return {
        "ok": False,
        "schema": "diship_gated_tool_result.v1",
        "result_code": "TOOL_EXECUTION_ERROR" if error else decision.result_code,
        "tool_name": decision.tool_name,
        "tool_call_id": decision.tool_call_id,
        "missing_authorizations": list(decision.missing_authorizations),
        "required_action": "Request explicit server-side authorization or bind the tool before retrying." if not error else "Inspect the redacted executor error.",
        "error_type": error or None,
        "write_performed": False,
        **status_fields,
    }


def make_tool_message(payload: Mapping[str, Any], *, tool_message_class: type | None = None) -> Any:
    """Adapt a paired payload to LangChain ToolMessage without requiring it in tests."""
    content = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, default=str)
    kwargs = {
        "content": content,
        "tool_call_id": str(payload.get("tool_call_id") or ""),
        "name": str(payload.get("tool_name") or ""),
    }
    if tool_message_class is None:
        try:
            from langchain_core.messages import ToolMessage as tool_message_class  # type: ignore
        except Exception:
            return FallbackToolMessage(**kwargs)
    return tool_message_class(**kwargs)


async def _invoke_tool(tool: Any, args: dict[str, Any], config: Any = None) -> Any:
    if hasattr(tool, "ainvoke"):
        return await tool.ainvoke(args, config=config)
    if hasattr(tool, "invoke"):
        value = tool.invoke(args, config=config)
    elif callable(tool):
        value = tool(**args)
    else:
        raise TypeError("tool is not invokable")
    return await value if inspect.isawaitable(value) else value


async def execute_gated_tool_calls(
    calls: Sequence[Any],
    *,
    tools_by_name: Mapping[str, Any],
    last_bound_tool_names: Iterable[str],
    authorization_flags: Mapping[str, Any] | None = None,
    config: Any = None,
    context_identifiers: Mapping[str, Any] | None = None,
    argument_sources: Mapping[str, str] | None = None,
    audit_sink: Callable[[dict[str, Any]], Any | Awaitable[Any]] | None = None,
    tool_message_class: type | None = None,
    execution_constraints: Mapping[str, Any] | None = None,
) -> list[Any]:
    """Execute calls serially and return exactly one paired result per call."""
    messages: list[Any] = []
    for call in calls:
        decision = evaluate_tool_call(
            call,
            last_bound_tool_names=last_bound_tool_names,
            authorization_flags=authorization_flags,
            context_identifiers=context_identifiers,
            argument_sources=argument_sources,
            execution_constraints=execution_constraints,
        )
        if audit_sink is not None:
            emitted = audit_sink(dict(decision.audit))
            if inspect.isawaitable(emitted):
                await emitted
        if not decision.allowed:
            messages.append(make_tool_message(paired_result_payload(decision), tool_message_class=tool_message_class))
            continue
        tool = tools_by_name.get(decision.tool_name)
        if tool is None:
            unavailable = GateDecision(**{**decision.to_dict(), "allowed": False, "result_code": "BOUND_TOOL_NOT_EXECUTABLE"})
            messages.append(make_tool_message(paired_result_payload(unavailable), tool_message_class=tool_message_class))
            continue
        try:
            result = await _invoke_tool(tool, decision.effective_args, config=config)
            payload = paired_result_payload(decision, result=result)
        except Exception as exc:  # noqa: BLE001
            # Error values are intentionally reduced to type only.  Raw
            # exception text can contain URLs, headers, or credentials.
            payload = paired_result_payload(decision, error=type(exc).__name__)
        messages.append(make_tool_message(payload, tool_message_class=tool_message_class))
    return messages
