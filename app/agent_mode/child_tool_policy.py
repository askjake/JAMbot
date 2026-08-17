"""Bounded parent-to-child tool policy for MCOP child conversations.

Phase D0 closed the child execution gap: children execute through the same
audited, gated executor as the parent. Phase D2 adds the *policy* half --
a bounded, immutable snapshot of the parent's effective authorization and tool
surface, captured at spawn, which the child may narrow but never broaden.

Persistence model (binding decision)
------------------------------------
The child stays ephemeral on ``MemorySaver`` for exactly one bounded run.
Durable authorization and tool-policy state lives only in the parent
checkpoint. The snapshot is an *upper bound*, not a recommendation.

What crosses the boundary
    effective authorization booleans, permitted tool families, permitted exact
    tools, registry generation, parent profile signature, parent request id, a
    safe parent-thread digest, the child run id, and a policy version.

What never crosses
    raw thread/chat/user ids, prompts, messages, tool objects, tool schemas,
    tool arguments, tool-result bodies, authorization material or hashes,
    webhook values, database credentials or connections, HTTP request objects,
    ContextVars, audit record bodies, and recursive MCOP spawn tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.agent.tool_execution_audit import (
    AUTHORIZATION_FLAG_KEYS,
    CHILD_SNAPSHOT_STATUSES,
    SCOPE_MCOP_CHILD,
    SNAPSHOT_ACCEPTED,
    SNAPSHOT_GENERATION_MISMATCH_NARROWED,
    SNAPSHOT_MALFORMED_BLOCKED,
    SNAPSHOT_NARROWED,
    SNAPSHOT_NO_PARENT_CONTEXT,
    bind_audit_request_state,
    current_audit_request_state,
    derive_child_audit_request_state,
    new_child_run_id,
    normalized_authorization_flags,
    reset_audit_request_state,
)

CHILD_TOOL_POLICY_SCHEMA = "child_tool_policy.v1"
CHILD_POLICY_VERSION = 1

# Deterministic truncation ceilings. Policy state is bounded, never unbounded.
MAX_NAME_LENGTH = 128
MAX_CHILD_TOOLSETS = 32
MAX_CHILD_EXTRA_TOOLS = 64
MAX_CHILD_BOUND_TOOL_NAMES = 128
MAX_ID_LENGTH = 128


def recursive_spawn_tool_names() -> frozenset[str]:
    """Authoritative MCOP spawn-tool exclusion set.

    Imported lazily so this module stays free of a circular import with
    ``app.agent_mode.child_conversation``.
    """
    try:
        from app.agent_mode.child_conversation import _MCOP_SPAWN_TOOL_NAMES

        return frozenset(_MCOP_SPAWN_TOOL_NAMES)
    except Exception:  # noqa: BLE001 - never let policy import break execution
        return frozenset()


def _clean_names(values: Iterable[Any] | None, limit: int) -> tuple[str, ...]:
    """Normalized, deterministically ordered, deterministically truncated set."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values or ():
        name = str(value or "").strip()[:MAX_NAME_LENGTH]
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    out.sort()
    return tuple(out[: max(0, int(limit))])


def _safe_id(value: Any) -> str:
    return str(value or "")[:MAX_ID_LENGTH]


def clamp_authorization_flags(
    parent_flags: Mapping[str, Any] | None,
    requested_flags: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    """Return ``requested AND parent``.

    A child may narrow authorization. A child can never elevate a false parent
    flag to true, regardless of what it requests.
    """
    parent = normalized_authorization_flags(parent_flags)
    if requested_flags is None:
        return dict(parent)
    requested = normalized_authorization_flags(requested_flags)
    return {
        key: bool(parent.get(key, False) and requested.get(key, False))
        for key in AUTHORIZATION_FLAG_KEYS
    }


@dataclass(frozen=True)
class ChildToolPolicy:
    """Immutable upper-bound policy snapshot handed to exactly one child run."""

    authorization_flags: tuple[tuple[str, bool], ...] = ()
    eligible_toolsets: tuple[str, ...] = ()
    eligible_extra_tools: tuple[str, ...] = ()
    registry_generation: str = ""
    # D3B1: content identity is what may narrow a child.  Refresh epoch and
    # health are recorded for audit only and never change permitted names.
    registry_content_signature: str = ""
    registry_refresh_epoch: int = 0
    registry_health_signature: str = ""
    profile_signature: str = ""
    parent_request_id: str = ""
    parent_thread_digest: str = ""
    child_run_id: str = ""
    truncated_fields: tuple[str, ...] = ()
    schema_version: str = CHILD_TOOL_POLICY_SCHEMA
    policy_version: int = CHILD_POLICY_VERSION
    snapshot_status: str = SNAPSHOT_ACCEPTED

    # ---- accessors -------------------------------------------------------
    @property
    def flags(self) -> dict[str, bool]:
        return {key: bool(value) for key, value in self.authorization_flags}

    def authorized(self, key: str) -> bool:
        return bool(self.flags.get(key, False))

    @property
    def permitted_toolsets(self) -> tuple[str, ...]:
        """Alias emphasising that the snapshot is an upper bound."""
        return self.eligible_toolsets

    @property
    def permitted_extra_tools(self) -> tuple[str, ...]:
        return self.eligible_extra_tools

    @property
    def content_signature(self) -> str:
        """Semantic inventory identity carried by this snapshot.

        A v1 snapshot stored the content signature in registry_generation, so
        that value remains the fallback and old snapshots stay comparable.
        """
        return self.registry_content_signature or self.registry_generation

    # ---- narrowing -------------------------------------------------------
    def narrow(self, **overrides: Any) -> "ChildToolPolicy":
        """Return a narrowed copy. Authorization and tools can only be reduced."""
        flags = self.flags
        if "authorization_flags" in overrides:
            flags = clamp_authorization_flags(flags, overrides.pop("authorization_flags"))
        toolsets = self.eligible_toolsets
        if "eligible_toolsets" in overrides:
            requested = _clean_names(overrides.pop("eligible_toolsets"), MAX_CHILD_TOOLSETS)
            toolsets = tuple(n for n in requested if n in set(self.eligible_toolsets))
        extras = self.eligible_extra_tools
        if "eligible_extra_tools" in overrides:
            requested = _clean_names(overrides.pop("eligible_extra_tools"), MAX_CHILD_EXTRA_TOOLS)
            extras = tuple(n for n in requested if n in set(self.eligible_extra_tools))
        status = str(overrides.pop("snapshot_status", "") or "")
        if status and status not in CHILD_SNAPSHOT_STATUSES:
            status = ""
        if overrides:
            raise ValueError("unsupported narrow() field: %s" % ",".join(sorted(overrides)))

        narrowed = (
            toolsets != self.eligible_toolsets
            or extras != self.eligible_extra_tools
            or flags != self.flags
        )
        return ChildToolPolicy(
            authorization_flags=tuple(sorted(flags.items())),
            eligible_toolsets=toolsets,
            eligible_extra_tools=extras,
            registry_generation=self.registry_generation,
            registry_content_signature=self.registry_content_signature,
            registry_refresh_epoch=self.registry_refresh_epoch,
            registry_health_signature=self.registry_health_signature,
            profile_signature=self.profile_signature,
            parent_request_id=self.parent_request_id,
            parent_thread_digest=self.parent_thread_digest,
            child_run_id=self.child_run_id,
            truncated_fields=self.truncated_fields,
            schema_version=self.schema_version,
            policy_version=self.policy_version,
            snapshot_status=status
            or (SNAPSHOT_NARROWED if narrowed else self.snapshot_status),
        )

    def as_safe_dict(self) -> dict[str, Any]:
        """Bounded, redacted representation safe for logs and audit context."""
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "authorization_flags": self.flags,
            "permitted_toolsets": list(self.eligible_toolsets),
            "permitted_extra_tools": list(self.eligible_extra_tools),
            "eligible_toolsets": list(self.eligible_toolsets),
            "eligible_extra_tools": list(self.eligible_extra_tools),
            "registry_generation": self.registry_generation,
            "registry_content_signature": self.content_signature,
            "registry_refresh_epoch": int(self.registry_refresh_epoch),
            "registry_health_signature": self.registry_health_signature,
            "profile_signature": self.profile_signature,
            "parent_request_id": self.parent_request_id,
            "parent_thread_digest": self.parent_thread_digest,
            "child_run_id": self.child_run_id,
            "scope": SCOPE_MCOP_CHILD,
            "snapshot_status": self.snapshot_status,
            "truncated_fields": list(self.truncated_fields),
            "authorization_material_stored": False,
        }


def validate_child_policy_snapshot(policy: Any) -> list[str]:
    """Return a list of problems. An empty list means the snapshot is usable."""
    problems: list[str] = []
    if not isinstance(policy, ChildToolPolicy):
        return ["snapshot is not a ChildToolPolicy"]
    if policy.schema_version != CHILD_TOOL_POLICY_SCHEMA:
        problems.append("schema_version mismatch")
    if not isinstance(policy.policy_version, int) or policy.policy_version < 1:
        problems.append("policy_version invalid")
    flags = policy.flags
    for key in AUTHORIZATION_FLAG_KEYS:
        if not isinstance(flags.get(key), bool):
            problems.append("authorization_flags.%s is not boolean" % key)
    for key in flags:
        if key not in AUTHORIZATION_FLAG_KEYS:
            problems.append("authorization_flags has unexpected key %s" % key)
    if not policy.child_run_id:
        problems.append("missing child_run_id")
    if len(policy.eligible_toolsets) > MAX_CHILD_TOOLSETS:
        problems.append("eligible_toolsets exceeds bound")
    if len(policy.eligible_extra_tools) > MAX_CHILD_EXTRA_TOOLS:
        problems.append("eligible_extra_tools exceeds bound")
    if policy.snapshot_status not in CHILD_SNAPSHOT_STATUSES:
        problems.append("snapshot_status not enumerated")
    for name in (*policy.eligible_toolsets, *policy.eligible_extra_tools):
        if len(name) > MAX_NAME_LENGTH:
            problems.append("name exceeds length bound")
            break
    return problems


def fail_closed_policy(child_run_id: str = "", status: str = SNAPSHOT_MALFORMED_BLOCKED) -> ChildToolPolicy:
    """The safe policy: no authorization, no permitted tools."""
    return ChildToolPolicy(
        authorization_flags=tuple(sorted(dict.fromkeys(AUTHORIZATION_FLAG_KEYS, False).items())),
        eligible_toolsets=(),
        eligible_extra_tools=(),
        child_run_id=_safe_id(child_run_id) or new_child_run_id(),
        snapshot_status=status if status in CHILD_SNAPSHOT_STATUSES else SNAPSHOT_MALFORMED_BLOCKED,
    )


def build_child_policy_snapshot(
    *,
    parent_state: Any = None,
    requested_authorization: Mapping[str, Any] | None = None,
    eligible_toolsets: Iterable[str] | None = None,
    eligible_extra_tools: Iterable[str] | None = None,
    registry_generation: str | None = None,
    profile_signature: str | None = None,
    child_run_id: str = "",
) -> ChildToolPolicy:
    """Build the bounded immutable snapshot for one child run.

    Sources of truth, in order:

    * authorization booleans come from the parent :class:`AuditRequestState`,
      which the parent model node populates with the D1-merged effective
      authorization for the current turn;
    * permitted tool families, permitted exact tools, registry generation and
      parent profile signature come from the D1 parent policy runtime snapshot
      unless explicitly supplied by the caller.

    Nothing is derived from assistant prose, compressed history, child task
    wording, or global process state. When no parent context exists at all the
    snapshot fails closed with every authorization boolean false.
    """
    if parent_state is None:
        parent_state = current_audit_request_state()

    status = SNAPSHOT_ACCEPTED
    runtime: Mapping[str, Any] = {}
    if (
        eligible_toolsets is None
        or eligible_extra_tools is None
        or registry_generation is None
        or profile_signature is None
    ):
        try:
            from app.agent.tool_policy_runtime import current_policy_runtime_context

            runtime = current_policy_runtime_context() or {}
        except Exception:  # noqa: BLE001 - absence is handled, never fatal
            runtime = {}

    if parent_state is None and not runtime:
        status = SNAPSHOT_NO_PARENT_CONTEXT

    parent_flags = normalized_authorization_flags(
        getattr(parent_state, "authorization_flags", None) if parent_state is not None else None
    )
    flags = clamp_authorization_flags(parent_flags, requested_authorization)

    raw_toolsets = list(
        eligible_toolsets if eligible_toolsets is not None else (runtime.get("active_toolsets") or ())
    )
    raw_extras = list(
        eligible_extra_tools
        if eligible_extra_tools is not None
        else (runtime.get("eligible_extra_tools") or ())
    )
    toolsets = _clean_names(raw_toolsets, MAX_CHILD_TOOLSETS)
    extras = _clean_names(raw_extras, MAX_CHILD_EXTRA_TOOLS)

    truncated: list[str] = []
    if len({str(v).strip() for v in raw_toolsets if str(v).strip()}) > len(toolsets):
        truncated.append("eligible_toolsets")
    if len({str(v).strip() for v in raw_extras if str(v).strip()}) > len(extras):
        truncated.append("eligible_extra_tools")

    generation = (
        registry_generation
        if registry_generation is not None
        else str(runtime.get("tool_registry_generation") or "")
    )
    signature = (
        profile_signature
        if profile_signature is not None
        else str(runtime.get("tool_profile_signature") or "")
    )

    content_signature_value = _safe_id(
        str(runtime.get("tool_registry_content_signature") or "") or generation
    )
    try:
        refresh_epoch_value = max(0, int(runtime.get("tool_registry_refresh_epoch") or 0))
    except (TypeError, ValueError):
        refresh_epoch_value = 0
    health_signature_value = _safe_id(str(runtime.get("tool_registry_health_signature") or ""))

    return ChildToolPolicy(
        authorization_flags=tuple(sorted(flags.items())),
        eligible_toolsets=toolsets,
        eligible_extra_tools=extras,
        registry_generation=_safe_id(generation),
        registry_content_signature=content_signature_value,
        registry_refresh_epoch=refresh_epoch_value,
        registry_health_signature=health_signature_value,
        profile_signature=_safe_id(signature),
        parent_request_id=_safe_id(getattr(parent_state, "request_id", "")) if parent_state is not None else "",
        parent_thread_digest=_safe_id(getattr(parent_state, "thread_id_digest", ""))
        if parent_state is not None
        else "",
        child_run_id=_safe_id(child_run_id) or new_child_run_id(),
        truncated_fields=tuple(truncated),
        snapshot_status=status,
    )


def current_registry_content_signature() -> str:
    """Live semantic inventory identity, or empty when it cannot be resolved."""
    try:
        from app.agent.tool_policy_state import current_registry_content_signature as _sig

        return _sig()
    except Exception:  # noqa: BLE001
        return ""


def current_registry_refresh_epoch() -> int:
    """Live process-local refresh epoch.  Never compared for narrowing."""
    try:
        from app.agent.tool_policy_state import current_registry_refresh_epoch as _epoch

        return int(_epoch())
    except Exception:  # noqa: BLE001
        return 0


def current_registry_health_signature() -> str:
    """Live health summary.  Never compared for narrowing or broadening."""
    try:
        from app.agent.tool_policy_state import current_registry_health_signature as _health

        return _health()
    except Exception:  # noqa: BLE001
        return ""


def current_registry_generation() -> str:
    """Backward-compatible alias for the semantic content signature."""
    return current_registry_content_signature()


def reconcile_snapshot_with_registry(
    policy: ChildToolPolicy,
    *,
    current_generation: str | None = None,
    current_content_signature: str | None = None,
    current_refresh_epoch: int | None = None,
    current_health_signature: str | None = None,
) -> tuple[ChildToolPolicy, bool]:
    """Revalidate a snapshot against the live registry.

    Returns ``(policy, content_match)``. On mismatch the snapshot permitted
    names remain a strict upper bound: names that no longer exist upstream are
    removed, and a newly appeared tool is **never** added merely because it is
    now available. If validation cannot complete the result fails closed.

    D3B1 comparison semantics:

    * only a semantic content-signature mismatch may revalidate and narrow;
    * a refresh-epoch-only difference must not narrow, because the effective
      inventory did not change;
    * a health-only difference must never broaden.  Execution against a family
      whose upstream health is not currently proven is blocked by the shared
      execution gate, not by widening or rewriting this snapshot.
    """
    if current_content_signature is not None:
        live = str(current_content_signature or "")
    elif current_generation is not None:
        live = str(current_generation or "")
    else:
        live = current_registry_content_signature()
    match = bool(live) and live == policy.content_signature
    if match or not policy.eligible_extra_tools:
        return policy, match

    try:
        from app.agent.tool_policy_state import resolve_upstream_availability

        available, _unavailable = resolve_upstream_availability(list(policy.eligible_extra_tools))
    except Exception:  # noqa: BLE001 - fail closed on validation failure
        return (
            policy.narrow(
                eligible_extra_tools=(),
                snapshot_status=SNAPSHOT_GENERATION_MISMATCH_NARROWED,
            ),
            match,
        )

    still_present = tuple(n for n in policy.eligible_extra_tools if n in set(available))
    if still_present == policy.eligible_extra_tools:
        return policy, match
    return (
        policy.narrow(
            eligible_extra_tools=still_present,
            snapshot_status=SNAPSHOT_GENERATION_MISMATCH_NARROWED,
        ),
        match,
    )


def narrow_policy_for_task(
    policy: ChildToolPolicy,
    *,
    task_required_toolsets: Sequence[str] = (),
    task_required_tools: Sequence[str] = (),
) -> ChildToolPolicy:
    """Reduce a snapshot to what the assigned task actually needs.

    Intersection only. An empty task requirement means "no additional
    narrowing for that dimension", never "grant everything".
    """
    overrides: dict[str, Any] = {}
    if task_required_toolsets:
        overrides["eligible_toolsets"] = list(task_required_toolsets)
    if task_required_tools:
        overrides["eligible_extra_tools"] = list(task_required_tools)
    if not overrides:
        return policy
    return policy.narrow(**overrides)


def child_safe_tools(
    tools: Iterable[Any],
    policy: ChildToolPolicy | None = None,
    *,
    task_required_names: Sequence[str] | None = None,
) -> list[Any]:
    """Intersect candidate child tools with the permitted policy surface.

    Applied in order:

    1. recursive MCOP spawn tools are always removed;
    2. when the policy declares permitted exact tools, anything outside that
       upper bound is removed;
    3. when the policy declares permitted families, the binding is restricted
       to the concrete tool names those families expose, so a family absent
       from the parent upper bound cannot appear in the child binding;
    4. when a task requirement is supplied, anything outside it is removed;
    5. any tool whose required capabilities are not all authorized by the
       policy is removed. This is defense in depth: the execution gate would
       block such a call anyway, but an unauthorized tool should never reach a
       model schema in the first place.
    """
    excluded = recursive_spawn_tool_names()

    allowed: set[str] | None = None
    if policy is not None and policy.eligible_extra_tools:
        allowed = set()
        for qualified in policy.eligible_extra_tools:
            allowed.add(qualified)
            if ":" in qualified:
                allowed.add(qualified.split(":", 1)[1])

    # Family upper bound. Tool objects do not carry an owning family, so the
    # permitted families are expanded into their concrete tool names and the
    # binding is restricted to that union. Without this the family bound would
    # only be enforced on the policy object and not on the real binding.
    family_allowed: set[str] | None = None
    if policy is not None and policy.eligible_toolsets:
        try:
            from app.agent.tool_execution_policy import get_tools_for_toolsets

            family_allowed = set()
            for candidate in get_tools_for_toolsets(list(policy.eligible_toolsets)) or ():
                candidate_name = str(getattr(candidate, "name", "") or "")
                if candidate_name:
                    family_allowed.add(candidate_name)
        except Exception:  # noqa: BLE001 - fail closed on expansion failure
            family_allowed = set()

    required: set[str] | None = None
    if task_required_names:
        required = set()
        for name in task_required_names:
            required.add(str(name))
            if ":" in str(name):
                required.add(str(name).split(":", 1)[1])

    flags = policy.flags if policy is not None else dict.fromkeys(AUTHORIZATION_FLAG_KEYS, False)
    try:
        from app.agent.tool_execution_gate import tool_is_permitted_by_flags
    except Exception:  # noqa: BLE001
        tool_is_permitted_by_flags = None  # type: ignore[assignment]

    out: list[Any] = []
    for tool in tools or ():
        name = str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "")
        if not name or name in excluded:
            continue
        if allowed is not None and name not in allowed:
            continue
        if (
            family_allowed is not None
            and name not in family_allowed
            and not (allowed is not None and name in allowed)
        ):
            continue
        if required is not None and name not in required:
            continue
        if tool_is_permitted_by_flags is not None and not tool_is_permitted_by_flags(name, flags):
            continue
        out.append(tool)
    return out


class child_audit_scope:
    """Context manager binding a child-scoped audit correlation state.

    The parent state object is never mutated, so a child run cannot overwrite
    the parent binding, parent request id, or parent authorization flags.
    """

    def __init__(
        self,
        policy: ChildToolPolicy,
        *,
        bound_tool_names: Iterable[str] = (),
        generation_match: bool = True,
        current_generation: str = "",
    ) -> None:
        self._policy = policy
        self._bound = _clean_names(bound_tool_names, MAX_CHILD_BOUND_TOOL_NAMES)
        self._generation_match = bool(generation_match)
        self._current_generation = _safe_id(current_generation)
        self._token = None
        self.state = None

    def __enter__(self):
        parent = current_audit_request_state()
        state = derive_child_audit_request_state(
            parent,
            child_run_id=self._policy.child_run_id,
            bound_tool_names=self._bound,
        )
        # Authorization is the clamped snapshot value, never the raw parent value.
        state.authorization_flags = dict(self._policy.flags)
        if self._policy.parent_thread_digest:
            state.thread_id_digest = self._policy.parent_thread_digest
        if self._policy.parent_request_id:
            state.parent_request_id = self._policy.parent_request_id
        state.snapshot_status = self._policy.snapshot_status
        state.snapshot_policy_version = int(self._policy.policy_version)
        state.snapshot_registry_generation = self._policy.registry_generation
        state.current_registry_generation = self._current_generation or self._policy.registry_generation
        state.generation_match = self._generation_match
        self.state = state
        self._token = bind_audit_request_state(state)
        return state

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._token is not None:
            reset_audit_request_state(self._token)
            self._token = None
        return False
