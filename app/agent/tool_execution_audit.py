"""Authoritative, redacted tool-execution audit evidence.

Schema: ``tool_execution_audit.v1``

Why this module exists
----------------------
Chat SSE deltas, assistant response text, and generic "Tools called" log lines
are **not** evidence that a tool executed.  They are model-authored or
best-effort.  This module is the server-owned source of truth for:

* the exact model-facing binding for the turn,
* the emitted tool-call identity,
* the execution-gate decision,
* server-side authorization booleans,
* argument rewrites and argument bounds (by field name),
* whether execution actually occurred,
* whether a paired ToolMessage was produced,
* the execution result code, duration, and safe error type.

Nothing in an audit event is derived from assistant prose.

Redaction contract (absolute)
-----------------------------
Events carry *names and enumerations only*.  A strict key allowlist is applied
by the writer so a future caller cannot widen a record by accident, and a
forbidden-key pattern drops anything that looks credential-bearing.

Stored:  argument field names, rewritten field names, bounded field names,
         boolean authorization flags, enumerated decisions/result codes,
         safe exception class names.

Never stored: raw prompt text, assistant text, tool argument values,
         tool-result bodies, authorization tokens, authorization hashes, token
         prefixes/suffixes, webhook URLs, API keys, database URLs, email
         addresses, raw thread IDs, model reasoning, or exception messages.

Authorization material is never hashed.  A hash of a secret is still a
secret-derived artifact, so token-shaped fields are dropped *by name* before an
event is constructed -- their values are never read, compared, or digested.

Thread digest
-------------
``thread_id_digest`` is a truncated SHA-256 over the fixed non-secret domain
label ``tool_execution_audit.v1:thread:`` concatenated with the server-generated
thread/session identifier.  Thread IDs are server-side conversation UUIDs, not
credentials.  The digest exists only to group events belonging to one
conversation without persisting the raw identifier.  No credential, prompt, or
user attribute is part of the digest input.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

TOOL_EXECUTION_AUDIT_SCHEMA = "tool_execution_audit.v1"
THREAD_DIGEST_DOMAIN = "tool_execution_audit.v1:thread:"

# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------
DECISION_ALLOWED = "ALLOWED"
DECISION_BLOCKED = "BLOCKED"
DECISION_REWRITTEN = "REWRITTEN"
DECISION_FAILED = "FAILED"

DECISIONS: frozenset[str] = frozenset(
    {DECISION_ALLOWED, DECISION_BLOCKED, DECISION_REWRITTEN, DECISION_FAILED}
)

# Execution scope: parent chat graph vs ephemeral MCOP child graph.
SCOPE_PARENT = "parent"
SCOPE_MCOP_CHILD = "mcop_child"
AUDIT_SCOPES: frozenset[str] = frozenset({SCOPE_PARENT, SCOPE_MCOP_CHILD})

# Enumerated outcomes of applying a parent policy snapshot to a child run.
SNAPSHOT_ACCEPTED = "SNAPSHOT_ACCEPTED"
SNAPSHOT_NARROWED = "SNAPSHOT_NARROWED"
SNAPSHOT_GENERATION_MISMATCH_NARROWED = "GENERATION_MISMATCH_NARROWED"
SNAPSHOT_MALFORMED_BLOCKED = "MALFORMED_SNAPSHOT_BLOCKED"
SNAPSHOT_ELEVATION_BLOCKED = "AUTHORIZATION_ELEVATION_BLOCKED"
SNAPSHOT_TOOL_OUTSIDE_BOUND_BLOCKED = "TOOL_OUTSIDE_PARENT_BOUND_BLOCKED"
SNAPSHOT_NO_PARENT_CONTEXT = "NO_PARENT_CONTEXT_FAIL_CLOSED"
CHILD_SNAPSHOT_STATUSES: frozenset[str] = frozenset({
    SNAPSHOT_ACCEPTED,
    SNAPSHOT_NARROWED,
    SNAPSHOT_GENERATION_MISMATCH_NARROWED,
    SNAPSHOT_MALFORMED_BLOCKED,
    SNAPSHOT_ELEVATION_BLOCKED,
    SNAPSHOT_TOOL_OUTSIDE_BOUND_BLOCKED,
    SNAPSHOT_NO_PARENT_CONTEXT,
})

RESULT_EXECUTED = "EXECUTED"
RESULT_BLOCKED_UNBOUND_TOOL = "BLOCKED_UNBOUND_TOOL"
RESULT_BLOCKED_HEAVY_UNAUTHORIZED = "BLOCKED_HEAVY_UNAUTHORIZED"
RESULT_BLOCKED_PERSISTENCE_UNAUTHORIZED = "BLOCKED_PERSISTENCE_UNAUTHORIZED"
RESULT_BLOCKED_MUTATION_UNAUTHORIZED = "BLOCKED_MUTATION_UNAUTHORIZED"
RESULT_BLOCKED_OPERATOR_UNAUTHORIZED = "BLOCKED_OPERATOR_UNAUTHORIZED"
RESULT_BLOCKED_TOOL_NOT_EXECUTABLE = "BLOCKED_TOOL_NOT_EXECUTABLE"
RESULT_BLOCKED_PARENT_ONLY_TURN = "BLOCKED_PARENT_ONLY_TURN"
RESULT_REWRITTEN_PERSIST_FALSE = "REWRITTEN_PERSIST_FALSE"
RESULT_REWRITTEN_LIMIT_BOUNDED = "REWRITTEN_LIMIT_BOUNDED"
RESULT_REWRITTEN_AUTHORIZATION_FIELD_DROPPED = "REWRITTEN_AUTHORIZATION_FIELD_DROPPED"
RESULT_FAILED_EXECUTOR_EXCEPTION = "FAILED_EXECUTOR_EXCEPTION"
RESULT_NO_TOOL_CALL = "NO_TOOL_CALL"

RESULT_CODES: frozenset[str] = frozenset(
    {
        RESULT_EXECUTED,
        RESULT_BLOCKED_UNBOUND_TOOL,
        RESULT_BLOCKED_HEAVY_UNAUTHORIZED,
        RESULT_BLOCKED_PERSISTENCE_UNAUTHORIZED,
        RESULT_BLOCKED_MUTATION_UNAUTHORIZED,
        RESULT_BLOCKED_OPERATOR_UNAUTHORIZED,
        RESULT_BLOCKED_TOOL_NOT_EXECUTABLE,
        RESULT_BLOCKED_PARENT_ONLY_TURN,
        RESULT_REWRITTEN_PERSIST_FALSE,
        RESULT_REWRITTEN_LIMIT_BOUNDED,
        RESULT_REWRITTEN_AUTHORIZATION_FIELD_DROPPED,
        RESULT_FAILED_EXECUTOR_EXCEPTION,
        RESULT_NO_TOOL_CALL,
    }
)

AUTHORIZATION_FLAG_KEYS: tuple[str, ...] = (
    "operator_authorized",
    "heavy_tools_authorized",
    "persistence_authorized",
    "mutation_authorized",
)

# --------------------------------------------------------------------------
# Redaction guards
# --------------------------------------------------------------------------
_ALLOWED_EVENT_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "event_id",
        "timestamp",
        "request_id",
        "thread_id_digest",
        "tool_call_id",
        "tool_name",
        "binding_signature",
        "tool_was_bound",
        "binding_known",
        "required_capabilities",
        "authorization_flags",
        "decision",
        "result_code",
        "executed",
        "paired_tool_result",
        "argument_field_names",
        "rewritten_fields",
        "bounded_fields",
        "duration_ms",
        "error_type",
        "enforcement_enabled",
        "graph_node",
        "scope",
        "parent_request_id",
        "child_run_id",
        "snapshot_status",
        "snapshot_policy_version",
        "snapshot_registry_generation",
        "current_registry_generation",
        "generation_match",
        "agent_execution_host",
        "tool_executor_host",
        "remote_service_host",
        "http_target",
        "source_client_address",
        "mcp_server_host",
        "repository_checkout_host",
        "repository_path",
        "repository_origin_host",
        "self_location_basis",
        "endpoint_can_establish_self_location",
    }
)

# Any key matching this pattern is dropped without reading its value.
_FORBIDDEN_KEY_RE = re.compile(
    r"token|secret|password|passwd|credential|webhook|api[_-]?key|cookie|bearer"
    r"|authorization_header|auth_header|signature_value|digest_value|hash",
    re.IGNORECASE,
)

# Field names emitted by a model that must never reach an audit record even as
# a *name*, because their presence in a name list is itself uninteresting and
# their similarity to real secrets invites accidental value logging later.
AUTHORIZATION_ARGUMENT_NAMES: frozenset[str] = frozenset(
    {
        "heavy_auth_token",
        "operator_auth_token",
        "authorization_token",
        "mutation_auth_token",
        "auth_token",
        "token",
    }
)

_MAX_NAME_LENGTH = 128
_MAX_LIST_ITEMS = 40
_MAX_LINE_BYTES = 4096


def _safe_name(value: Any) -> str:
    text = str(value if value is not None else "")
    text = text.replace("\n", " ").replace("\r", " ").strip()
    return text[:_MAX_NAME_LENGTH]


def _safe_name_list(values: Iterable[Any] | None) -> list[str]:
    """Return a bounded, de-duplicated, sorted list of safe field names.

    Authorization-shaped names are removed entirely; their values are never
    inspected.
    """
    if not values:
        return []
    out: set[str] = set()
    for value in values:
        name = _safe_name(value)
        if not name:
            continue
        if name in AUTHORIZATION_ARGUMENT_NAMES:
            continue
        if _FORBIDDEN_KEY_RE.search(name):
            continue
        out.add(name)
    return sorted(out)[:_MAX_LIST_ITEMS]


def normalized_authorization_flags(flags: Mapping[str, Any] | None) -> dict[str, bool]:
    """Return only the four boolean capability grants, defaulting to False."""
    source = flags or {}
    return {key: bool(source.get(key, False)) for key in AUTHORIZATION_FLAG_KEYS}


# --------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------
def new_request_id() -> str:
    return "req-" + uuid.uuid4().hex


def new_child_run_id() -> str:
    """Opaque identifier for one ephemeral MCOP child run."""
    return "child-" + uuid.uuid4().hex[:16]


def new_event_id() -> str:
    return "tea-" + uuid.uuid4().hex


def digest_thread_id(thread_id: Any) -> str:
    """One-way, truncated digest of a server-generated conversation identifier."""
    raw = str(thread_id or "")
    if not raw:
        return ""
    digest = hashlib.sha256((THREAD_DIGEST_DOMAIN + raw).encode("utf-8")).hexdigest()
    return "thr-" + digest[:24]


def binding_signature_for(tool_names: Iterable[str]) -> str:
    """Stable, non-secret signature of the exact model-facing binding."""
    names = sorted({_safe_name(name) for name in tool_names if _safe_name(name)})
    if not names:
        return "bind-empty"
    digest = hashlib.sha256("|".join(names).encode("utf-8")).hexdigest()
    return f"bind-{len(names):04d}-{digest[:16]}"


# --------------------------------------------------------------------------
# Server-owned request correlation
# --------------------------------------------------------------------------
@dataclass
class AuditRequestState:
    """Mutable, server-owned correlation state for one HTTP chat request.

    The object is intentionally mutable and shared by reference.  A ContextVar
    copy is taken when asyncio spawns child tasks, so graph nodes that run in
    sibling tasks still observe binding updates written by the model node.

    This is never exposed as a model-editable tool parameter.
    """

    request_id: str = field(default_factory=new_request_id)
    thread_id_digest: str = ""
    binding_signature: str = "bind-unknown"
    bound_tool_names: tuple[str, ...] = ()
    binding_known: bool = False
    authorization_flags: dict[str, bool] = field(
        default_factory=lambda: dict.fromkeys(AUTHORIZATION_FLAG_KEYS, False)
    )
    # Execution scope identity.  A child run carries the parent request id
    # and its own run id so evidence correlates without any raw thread id.
    scope: str = SCOPE_PARENT
    parent_request_id: str = ""
    child_run_id: str = ""
    # Phase D2: bounded parent-policy-snapshot correlation for a child run.
    snapshot_status: str = ""
    snapshot_policy_version: int = 0
    snapshot_registry_generation: str = ""
    current_registry_generation: str = ""
    generation_match: bool = True
    # Server-owned bounded tool-policy snapshot for this request.  It is
    # never an audit event field and never a model-editable argument; it
    # exists so sibling graph-node tasks observe the agent node's state.
    policy_snapshot: Any = None
    # Current task execution constraints are server-owned booleans. Only this
    # bounded non-secret mapping may be copied into gate/audit metadata.
    execution_constraints: dict[str, bool] = field(default_factory=dict)

    def record_binding(
        self,
        *,
        bound_tool_names: Iterable[str],
        authorization_flags: Mapping[str, Any] | None = None,
        thread_id: Any = None,
        execution_constraints: Mapping[str, Any] | None = None,
    ) -> None:
        names = tuple(sorted({_safe_name(n) for n in bound_tool_names if _safe_name(n)}))
        self.bound_tool_names = names
        self.binding_signature = binding_signature_for(names)
        self.binding_known = True
        if authorization_flags is not None:
            self.authorization_flags = normalized_authorization_flags(authorization_flags)
        if thread_id:
            self.thread_id_digest = digest_thread_id(thread_id)
        if execution_constraints is not None:
            self.execution_constraints = {
                "mcop_children_forbidden": bool(
                    execution_constraints.get("mcop_children_forbidden", False)
                )
            }


_audit_request_var: ContextVar[AuditRequestState | None] = ContextVar(
    "tool_execution_audit_request_state", default=None
)


def bind_audit_request_state(state: AuditRequestState | None = None) -> Token:
    return _audit_request_var.set(state or AuditRequestState())


def reset_audit_request_state(token: Token) -> None:
    try:
        _audit_request_var.reset(token)
    except (ValueError, LookupError):  # pragma: no cover - defensive
        pass


def current_audit_request_state() -> AuditRequestState | None:
    return _audit_request_var.get()


def require_audit_request_state() -> AuditRequestState:
    """Return the active correlation state, creating a detached one if absent.

    A detached state keeps audit coverage for non-HTTP entrypoints (cron,
    tests, background jobs) instead of silently dropping evidence.
    """
    state = _audit_request_var.get()
    if state is None:
        state = AuditRequestState()
        _audit_request_var.set(state)
    return state


def record_model_facing_binding(
    *,
    bound_tool_names: Iterable[str],
    authorization_flags: Mapping[str, Any] | None = None,
    thread_id: Any = None,
    execution_constraints: Mapping[str, Any] | None = None,
) -> AuditRequestState:
    state = require_audit_request_state()
    state.record_binding(
        bound_tool_names=bound_tool_names,
        authorization_flags=authorization_flags,
        thread_id=thread_id,
        execution_constraints=execution_constraints,
    )
    return state


def derive_child_audit_request_state(
    parent,
    *,
    child_run_id: str = "",
    bound_tool_names: Iterable[str] = (),
):
    """Derive an MCOP child correlation state from the parent state.

    Authorization is copied, never widened.  The child receives only safe
    identifiers: the parent request id and the already-digested thread value.
    No raw thread id, prompt, or authorization material crosses the boundary.
    """
    parent_flags = normalized_authorization_flags(
        getattr(parent, "authorization_flags", None) if parent is not None else None
    )
    state = AuditRequestState(
        thread_id_digest=_safe_name(
            getattr(parent, "thread_id_digest", "") if parent is not None else ""
        ),
        authorization_flags=dict(parent_flags),
        scope=SCOPE_MCOP_CHILD,
        parent_request_id=_safe_name(
            getattr(parent, "request_id", "") if parent is not None else ""
        ),
        child_run_id=_safe_name(child_run_id) or new_child_run_id(),
        execution_constraints={
            "mcop_children_forbidden": bool(
                getattr(parent, "execution_constraints", {}).get(
                    "mcop_children_forbidden", False
                ) if parent is not None else False
            )
        },
    )
    if bound_tool_names:
        state.record_binding(
            bound_tool_names=bound_tool_names, authorization_flags=parent_flags
        )
    return state


# --------------------------------------------------------------------------
# Gate result-code mapping
# --------------------------------------------------------------------------
_MISSING_AUTH_TO_RESULT: tuple[tuple[str, str], ...] = (
    ("heavy_tools_authorized", RESULT_BLOCKED_HEAVY_UNAUTHORIZED),
    ("mutation_authorized", RESULT_BLOCKED_MUTATION_UNAUTHORIZED),
    ("persistence_authorized", RESULT_BLOCKED_PERSISTENCE_UNAUTHORIZED),
    ("operator_authorized", RESULT_BLOCKED_OPERATOR_UNAUTHORIZED),
)


def classify_gate_decision(
    *,
    gate_result_code: str,
    allowed: bool,
    missing_authorizations: Sequence[str] = (),
    rewritten_fields: Sequence[str] = (),
    bounded_fields: Sequence[str] = (),
    authorization_field_dropped: bool = False,
) -> tuple[str, str]:
    """Map a repository gate result code onto the audit decision/result pair."""
    code = str(gate_result_code or "")
    missing = tuple(missing_authorizations or ())

    if not allowed:
        if code == "TOOL_NOT_IN_LAST_BINDING":
            return DECISION_BLOCKED, RESULT_BLOCKED_UNBOUND_TOOL
        if code == "BOUND_TOOL_NOT_EXECUTABLE":
            return DECISION_BLOCKED, RESULT_BLOCKED_TOOL_NOT_EXECUTABLE
        if code == "BLOCKED_PARENT_ONLY_TURN":
            return DECISION_BLOCKED, RESULT_BLOCKED_PARENT_ONLY_TURN
        for key, result in _MISSING_AUTH_TO_RESULT:
            if key in missing:
                return DECISION_BLOCKED, result
        return DECISION_BLOCKED, RESULT_BLOCKED_OPERATOR_UNAUTHORIZED

    if "persist" in tuple(rewritten_fields):
        return DECISION_REWRITTEN, RESULT_REWRITTEN_PERSIST_FALSE
    if bounded_fields:
        return DECISION_REWRITTEN, RESULT_REWRITTEN_LIMIT_BOUNDED
    if authorization_field_dropped:
        return DECISION_REWRITTEN, RESULT_REWRITTEN_AUTHORIZATION_FIELD_DROPPED
    return DECISION_ALLOWED, RESULT_EXECUTED


# --------------------------------------------------------------------------
# Event construction
# --------------------------------------------------------------------------
def build_event(
    *,
    request_id: str = "",
    thread_id_digest: str = "",
    tool_call_id: str = "",
    tool_name: str = "",
    binding_signature: str = "bind-unknown",
    tool_was_bound: bool = False,
    binding_known: bool = False,
    required_capabilities: Iterable[str] = (),
    authorization_flags: Mapping[str, Any] | None = None,
    decision: str = DECISION_BLOCKED,
    result_code: str = RESULT_BLOCKED_UNBOUND_TOOL,
    executed: bool = False,
    paired_tool_result: bool = False,
    argument_field_names: Iterable[str] = (),
    rewritten_fields: Iterable[str] = (),
    bounded_fields: Iterable[str] = (),
    duration_ms: float | int = 0,
    error_type: str | None = None,
    enforcement_enabled: bool = True,
    graph_node: str = "tools",
    scope: str = SCOPE_PARENT,
    parent_request_id: str = "",
    child_run_id: str = "",
    snapshot_status: str = "",
    snapshot_policy_version: int = 0,
    snapshot_registry_generation: str = "",
    current_registry_generation: str = "",
    generation_match: bool = True,
    agent_execution_host: str = "",
    tool_executor_host: str = "",
    remote_service_host: str = "",
    http_target: str = "",
    source_client_address: str = "",
    mcp_server_host: str = "",
    repository_checkout_host: str = "",
    repository_path: str = "",
    repository_origin_host: str = "",
    self_location_basis: str = "",
    endpoint_can_establish_self_location: bool = False,
) -> dict[str, Any]:
    """Build one redacted audit event.  Only allowlisted keys are produced."""
    event = {
        "schema_version": TOOL_EXECUTION_AUDIT_SCHEMA,
        "event_id": new_event_id(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "request_id": _safe_name(request_id),
        "thread_id_digest": _safe_name(thread_id_digest),
        "tool_call_id": _safe_name(tool_call_id),
        "tool_name": _safe_name(tool_name),
        "binding_signature": _safe_name(binding_signature),
        "tool_was_bound": bool(tool_was_bound),
        "binding_known": bool(binding_known),
        "required_capabilities": _safe_name_list(required_capabilities),
        "authorization_flags": normalized_authorization_flags(authorization_flags),
        "decision": str(decision) if str(decision) in DECISIONS else DECISION_BLOCKED,
        "result_code": str(result_code) if str(result_code) in RESULT_CODES else RESULT_BLOCKED_UNBOUND_TOOL,
        "executed": bool(executed),
        "paired_tool_result": bool(paired_tool_result),
        "argument_field_names": _safe_name_list(argument_field_names),
        "rewritten_fields": _safe_name_list(rewritten_fields),
        "bounded_fields": _safe_name_list(bounded_fields),
        "duration_ms": int(max(0, round(float(duration_ms or 0)))),
        "error_type": _safe_exception_name(error_type),
        "enforcement_enabled": bool(enforcement_enabled),
        "graph_node": _safe_name(graph_node),
        "scope": str(scope) if str(scope) in AUDIT_SCOPES else SCOPE_PARENT,
        "parent_request_id": _safe_name(parent_request_id),
        "child_run_id": _safe_name(child_run_id),
        "snapshot_status": (
            str(snapshot_status)
            if str(snapshot_status) in CHILD_SNAPSHOT_STATUSES
            else ""
        ),
        "snapshot_policy_version": int(snapshot_policy_version or 0),
        "snapshot_registry_generation": _safe_name(snapshot_registry_generation),
        "current_registry_generation": _safe_name(current_registry_generation),
        "generation_match": bool(generation_match),
        "agent_execution_host": _safe_name(agent_execution_host),
        "tool_executor_host": _safe_name(tool_executor_host),
        "remote_service_host": _safe_name(remote_service_host),
        "http_target": _safe_name(http_target),
        "source_client_address": _safe_name(source_client_address),
        "mcp_server_host": _safe_name(mcp_server_host),
        "repository_checkout_host": _safe_name(repository_checkout_host),
        "repository_path": _safe_name(repository_path),
        "repository_origin_host": _safe_name(repository_origin_host),
        "self_location_basis": _safe_name(self_location_basis),
        "endpoint_can_establish_self_location": False,
    }
    return sanitize_event(event)


_SAFE_EXCEPTION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{0,63}$")


def _safe_exception_name(value: Any) -> str | None:
    """Accept only a bare exception class name; reject anything message-like."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if not _SAFE_EXCEPTION_RE.match(text):
        return "UnsafeErrorTypeRedacted"
    return text


def sanitize_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Drop non-allowlisted and credential-shaped keys from an event."""
    clean: dict[str, Any] = {}
    for key, value in dict(event).items():
        name = str(key)
        if name not in _ALLOWED_EVENT_KEYS:
            continue
        if _FORBIDDEN_KEY_RE.search(name):
            continue
        if name == "authorization_flags":
            clean[name] = normalized_authorization_flags(value if isinstance(value, Mapping) else {})
            continue
        clean[name] = value
    return clean


def validate_event(event: Any) -> list[str]:
    """Return a list of schema problems.  An empty list means the record is valid."""
    problems: list[str] = []
    if not isinstance(event, Mapping):
        return ["record is not an object"]
    if event.get("schema_version") != TOOL_EXECUTION_AUDIT_SCHEMA:
        problems.append("schema_version mismatch")
    for key in ("event_id", "timestamp", "tool_name", "decision", "result_code"):
        if not str(event.get(key) or ""):
            problems.append(f"missing {key}")
    if str(event.get("decision") or "") not in DECISIONS:
        problems.append("decision not enumerated")
    if str(event.get("result_code") or "") not in RESULT_CODES:
        problems.append("result_code not enumerated")
    for key in ("executed", "paired_tool_result", "tool_was_bound"):
        if not isinstance(event.get(key), bool):
            problems.append(f"{key} is not boolean")
    flags = event.get("authorization_flags")
    if not isinstance(flags, Mapping):
        problems.append("authorization_flags is not an object")
    else:
        for key in AUTHORIZATION_FLAG_KEYS:
            if not isinstance(flags.get(key), bool):
                problems.append(f"authorization_flags.{key} is not boolean")
        for key in flags:
            if key not in AUTHORIZATION_FLAG_KEYS:
                problems.append(f"authorization_flags has unexpected key {key}")
    snapshot_status = event.get("snapshot_status")
    if snapshot_status:
        if str(snapshot_status) not in CHILD_SNAPSHOT_STATUSES:
            problems.append("snapshot_status not enumerated")
    if event.get("generation_match") is not None and not isinstance(
        event.get("generation_match"), bool
    ):
        problems.append("generation_match is not boolean")
    scope_value = event.get("scope")
    if scope_value is not None:
        if str(scope_value) not in AUDIT_SCOPES:
            problems.append("scope not enumerated")
        elif str(scope_value) == SCOPE_MCOP_CHILD and not str(event.get("child_run_id") or ""):
            problems.append("mcop_child event missing child_run_id")
    for key in event:
        if str(key) not in _ALLOWED_EVENT_KEYS:
            problems.append(f"non-allowlisted key {key}")
        if _FORBIDDEN_KEY_RE.search(str(key)):
            problems.append(f"forbidden key {key}")
    if event.get("decision") == DECISION_BLOCKED and event.get("executed") is True:
        problems.append("blocked event claims execution")
    if not isinstance(event.get("duration_ms"), int):
        problems.append("duration_ms is not an integer")
    error_type = event.get("error_type")
    if error_type is not None and not _SAFE_EXCEPTION_RE.match(str(error_type)):
        problems.append("error_type is not a bare exception class name")
    return problems


# --------------------------------------------------------------------------
# Bounded, runtime-only persistence sink
# --------------------------------------------------------------------------
_DEFAULT_DIR_NAME = os.path.join("var", "tool_execution_audit")
_ACTIVE_FILENAME = "tool_execution_audit.jsonl"
_ROTATED_GLOB = "tool_execution_audit.*.jsonl"

DEFAULT_MAX_BYTES = 5_000_000
DEFAULT_MAX_FILES = 10


def _repo_root() -> Path:
    # app/agent/tool_execution_audit.py -> app/agent -> app -> repo root
    return Path(__file__).resolve().parents[2]


def default_audit_dir() -> Path:
    override = os.getenv("DISHCHAT_TOOL_AUDIT_DIR")
    if override:
        return Path(override)
    return _repo_root() / _DEFAULT_DIR_NAME


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except (TypeError, ValueError):
        return default


def enforcement_enabled() -> bool:
    """Whether gate blocks are enforced (default) or only observed.

    ``DISHCHAT_TOOL_AUDIT_ENFORCE=0`` degrades to observe-only.  It exists as an
    operator rollback switch; auditing is unaffected either way.
    """
    raw = os.getenv("DISHCHAT_TOOL_AUDIT_ENFORCE")
    if raw is None:
        return True
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


class ToolExecutionAuditSink:
    """Append-only, size-bounded, permission-restricted JSONL audit sink.

    Guarantees:
      * directories are created 0700 and files 0600 where the platform allows;
      * one event is one line, written with a single ``O_APPEND`` write;
      * rotation is deterministic and bounded by size;
      * retention is bounded by rotated-file count;
      * a write failure never propagates to chat execution -- it increments a
        failure counter and emits at most one warning per backoff window;
      * the sink never logs event contents, so no recursive logging loop exists.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str] | None = None,
        *,
        max_bytes: int | None = None,
        max_files: int | None = None,
    ) -> None:
        self._directory = Path(directory) if directory is not None else default_audit_dir()
        self._max_bytes = int(max_bytes) if max_bytes is not None else _env_int(
            "DISHCHAT_TOOL_AUDIT_MAX_BYTES", DEFAULT_MAX_BYTES, minimum=1024
        )
        self._max_files = int(max_files) if max_files is not None else _env_int(
            "DISHCHAT_TOOL_AUDIT_MAX_FILES", DEFAULT_MAX_FILES, minimum=1
        )
        self._lock = threading.Lock()
        self._write_failures = 0
        self._last_warning_at = 0.0

    # -- introspection ----------------------------------------------------
    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def active_path(self) -> Path:
        return self._directory / _ACTIVE_FILENAME

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def max_files(self) -> int:
        return self._max_files

    @property
    def write_failure_count(self) -> int:
        """Audit-write failure metric.  Non-zero means evidence may be missing."""
        return self._write_failures

    # -- writing ----------------------------------------------------------
    def record(self, event: Mapping[str, Any]) -> bool:
        """Persist one event.  Returns True on success and never raises."""
        try:
            clean = sanitize_event(event)
            line = json.dumps(clean, sort_keys=True, ensure_ascii=True, default=str)
            encoded = (line + "\n").encode("utf-8")
            if len(encoded) > _MAX_LINE_BYTES:
                encoded = self._shrink(clean)
            with self._lock:
                self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                self._rotate_if_needed(len(encoded))
                fd = os.open(
                    self.active_path,
                    os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                    0o600,
                )
                try:
                    os.write(fd, encoded)
                finally:
                    os.close(fd)
                self._harden_permissions()
            return True
        except Exception as exc:  # noqa: BLE001 - audit must never break chat
            self._note_failure(exc)
            return False

    def _shrink(self, clean: dict[str, Any]) -> bytes:
        """Drop optional name lists until the record fits one bounded line."""
        reduced = dict(clean)
        for key in ("argument_field_names", "bounded_fields", "rewritten_fields", "required_capabilities"):
            reduced[key] = []
            candidate = (json.dumps(reduced, sort_keys=True, ensure_ascii=True, default=str) + "\n").encode("utf-8")
            if len(candidate) <= _MAX_LINE_BYTES:
                return candidate
        return (json.dumps(reduced, sort_keys=True, ensure_ascii=True, default=str) + "\n").encode("utf-8")[:_MAX_LINE_BYTES]

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        path = self.active_path
        try:
            current = path.stat().st_size
        except FileNotFoundError:
            return
        if current + incoming_bytes <= self._max_bytes:
            return
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        target = self._directory / f"tool_execution_audit.{stamp}.jsonl"
        suffix = 0
        while target.exists():
            suffix += 1
            target = self._directory / f"tool_execution_audit.{stamp}-{suffix:02d}.jsonl"
        os.replace(path, target)
        try:
            os.chmod(target, 0o600)
        except OSError:  # pragma: no cover - platform dependent
            pass
        self._prune()

    def _prune(self) -> None:
        rotated = sorted(
            self._directory.glob(_ROTATED_GLOB),
            key=lambda p: p.name,
        )
        excess = len(rotated) - self._max_files
        for path in rotated[: max(0, excess)]:
            try:
                path.unlink()
            except OSError:  # pragma: no cover
                pass

    def _harden_permissions(self) -> None:
        try:
            os.chmod(self._directory, 0o700)
            os.chmod(self.active_path, 0o600)
        except OSError:  # pragma: no cover - platform dependent
            pass

    def _note_failure(self, exc: BaseException) -> None:
        self._write_failures += 1
        now = time.monotonic()
        if now - self._last_warning_at < 60.0:
            return
        self._last_warning_at = now
        # Only the exception class name is logged: exception text can carry
        # paths, URLs, or user data.
        logger.warning(
            "tool_execution_audit write failed (count=%d type=%s); chat execution unaffected",
            self._write_failures,
            type(exc).__name__,
        )

    # -- reading ----------------------------------------------------------
    def files_newest_first(self) -> list[Path]:
        paths: list[Path] = []
        if self.active_path.exists():
            paths.append(self.active_path)
        paths.extend(sorted(self._directory.glob(_ROTATED_GLOB), key=lambda p: p.name, reverse=True))
        return paths

    def iter_events(self, *, max_records: int = 1000) -> Iterator[dict[str, Any]]:
        """Yield at most ``max_records`` parsed records, newest file first."""
        emitted = 0
        for path in self.files_newest_first():
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    lines = handle.readlines()
            except OSError:
                continue
            for raw in reversed(lines):
                if emitted >= max_records:
                    return
                text = raw.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except ValueError:
                    record = {"_malformed": True, "_raw_length": len(text)}
                emitted += 1
                yield record


_default_sink: ToolExecutionAuditSink | None = None
_default_sink_lock = threading.Lock()


def get_audit_sink() -> ToolExecutionAuditSink:
    global _default_sink
    with _default_sink_lock:
        if _default_sink is None:
            _default_sink = ToolExecutionAuditSink()
        return _default_sink


def set_audit_sink(sink: ToolExecutionAuditSink | None) -> None:
    """Replace the process-wide sink.  Intended for tests and operators."""
    global _default_sink
    with _default_sink_lock:
        _default_sink = sink


def record_event(event: Mapping[str, Any], *, sink: ToolExecutionAuditSink | None = None) -> bool:
    return (sink or get_audit_sink()).record(event)


MAX_QUERY_LIMIT = 500
DEFAULT_QUERY_LIMIT = 50


def query_events(
    *,
    sink: ToolExecutionAuditSink | None = None,
    request_id: str = "",
    thread_id_digest: str = "",
    tool_call_id: str = "",
    tool_name: str = "",
    decision: str = "",
    result_code: str = "",
    since: str = "",
    until: str = "",
    limit: int = DEFAULT_QUERY_LIMIT,
    scan_limit: int = 20_000,
) -> list[dict[str, Any]]:
    """Bounded operator query.  Argument values and result bodies cannot be
    returned because they were never stored."""
    bounded_limit = max(1, min(int(limit or DEFAULT_QUERY_LIMIT), MAX_QUERY_LIMIT))
    active = sink or get_audit_sink()
    out: list[dict[str, Any]] = []
    for record in active.iter_events(max_records=max(bounded_limit, int(scan_limit))):
        if request_id and record.get("request_id") != request_id:
            continue
        if thread_id_digest and record.get("thread_id_digest") != thread_id_digest:
            continue
        if tool_call_id and record.get("tool_call_id") != tool_call_id:
            continue
        if tool_name and record.get("tool_name") != tool_name:
            continue
        if decision and record.get("decision") != decision:
            continue
        if result_code and record.get("result_code") != result_code:
            continue
        stamp = str(record.get("timestamp") or "")
        if since and stamp < since:
            continue
        if until and stamp > until:
            continue
        out.append(record)
        if len(out) >= bounded_limit:
            break
    return out


def audit_storage_summary() -> dict[str, Any]:
    sink = get_audit_sink()
    return {
        "schema_version": TOOL_EXECUTION_AUDIT_SCHEMA,
        "directory": str(sink.directory),
        "active_file": str(sink.active_path),
        "max_bytes": sink.max_bytes,
        "max_rotated_files": sink.max_files,
        "write_failure_count": sink.write_failure_count,
        "enforcement_enabled": enforcement_enabled(),
        "argument_values_recorded": False,
        "tool_result_bodies_recorded": False,
        "authorization_material_recorded": False,
        "authorization_hashes_recorded": False,
    }
