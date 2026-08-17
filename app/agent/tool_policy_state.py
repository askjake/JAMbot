"""Checkpointed parent tool-policy state (Phase D1).

Why this module exists
----------------------
Before D1, parent authorization was re-derived on every turn by walking the
retained message window (``_authorization_flags_for_window``).  That coupled a
security decision to message retention: compressing or truncating an old grant
message silently revoked authorization, and an old quoted grant could silently
re-grant it.

D1 replaces that with explicit, thread-scoped checkpoint state:

    stored authorization state
    + delta parsed from the current user message only
    = effective authorization for this turn

Nothing here scans history. Old checkpoints that predate these fields load with
every authorization boolean false, and no grant is ever inferred during that
migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Iterable, Mapping, Sequence

from app.agent.tool_profiles import (
    AUTHORIZATION_DEFAULTS,
    canonical_extra_tool,
    normalize_authorization_flags,
    profile_signature,
)

TOOL_POLICY_STATE_SCHEMA = "diship_tool_policy_state.v1"
TOOL_POLICY_VERSION = 3

# --------------------------------------------------------------------------
# Deterministic bounds.  Policy state is bounded; it never grows forever.
# --------------------------------------------------------------------------
MAX_NAME_LENGTH = 128
MAX_ACTIVE_TOOLSETS = 48
MAX_REQUESTED_TOOLSETS = 32
MAX_REQUESTED_EXTRA_TOOLS = 64
MAX_PROCESSED_ACTIVATION_TOOL_CALL_IDS = 64
MAX_ELIGIBLE_EXTRA_TOOLS = 64
MAX_PENDING_EXTRA_TOOLS = 64
MAX_UNAVAILABLE_EXTRA_TOOLS = 64
MAX_LAST_BOUND_TOOL_NAMES = 256
MAX_ACTIVATION_STATUS_ENTRIES = 64

FIELD_BOUNDS: dict[str, int] = {
    "active_toolsets": MAX_ACTIVE_TOOLSETS,
    "requested_toolsets": MAX_REQUESTED_TOOLSETS,
    "requested_extra_tools": MAX_REQUESTED_EXTRA_TOOLS,
    "processed_activation_tool_call_ids": MAX_PROCESSED_ACTIVATION_TOOL_CALL_IDS,
    "eligible_extra_tools": MAX_ELIGIBLE_EXTRA_TOOLS,
    "pending_authorization_extra_tools": MAX_PENDING_EXTRA_TOOLS,
    "pending_registry_requests": MAX_PENDING_EXTRA_TOOLS,
    "unavailable_extra_tools": MAX_UNAVAILABLE_EXTRA_TOOLS,
    "last_bound_tool_names": MAX_LAST_BOUND_TOOL_NAMES,
}

# --------------------------------------------------------------------------
# Authorization delta
# --------------------------------------------------------------------------
GRANT = "GRANT"
REVOKE = "REVOKE"
UNCHANGED = "UNCHANGED"
GRANT_STATES: frozenset[str] = frozenset({GRANT, REVOKE, UNCHANGED})

AUTHORIZATION_KEYS: tuple[str, ...] = tuple(sorted(AUTHORIZATION_DEFAULTS))


@dataclass(frozen=True)
class AuthorizationDelta:
    """Typed per-capability outcome of parsing exactly one user message."""

    operator_authorized: str = UNCHANGED
    heavy_tools_authorized: str = UNCHANGED
    persistence_authorized: str = UNCHANGED
    mutation_authorized: str = UNCHANGED

    def as_dict(self) -> dict[str, str]:
        return {key: getattr(self, key, UNCHANGED) for key in AUTHORIZATION_KEYS}

    def is_empty(self) -> bool:
        return all(value == UNCHANGED for value in self.as_dict().values())

    def changed_keys(self) -> tuple[str, ...]:
        return tuple(k for k, v in sorted(self.as_dict().items()) if v != UNCHANGED)


def parse_authorization_delta(current_user_text: str | None) -> AuthorizationDelta:
    """Parse a typed delta from the CURRENT user message only.

    Delegates phrase detection to the existing shared parser
    (``app.agent.tool_authorization.parse_authorization_updates``) so there is
    exactly one authorization grammar in the codebase.  This function only
    converts that boolean-update mapping into an explicit tri-state delta:
    a present ``True`` is GRANT, a present ``False`` is REVOKE, and an absent
    key is UNCHANGED.
    """
    from app.agent.tool_authorization import parse_authorization_updates

    updates = parse_authorization_updates(current_user_text) or {}
    fields: dict[str, str] = {}
    for key in AUTHORIZATION_KEYS:
        if key in updates:
            fields[key] = GRANT if bool(updates[key]) else REVOKE
        else:
            fields[key] = UNCHANGED
    return AuthorizationDelta(**fields)


def merge_authorization_state(
    previous: Mapping[str, Any] | None,
    delta: AuthorizationDelta | Mapping[str, str] | None,
) -> dict[str, bool]:
    """Apply a typed delta to stored state.

    UNCHANGED keeps the stored boolean, GRANT sets true, REVOKE sets false.
    Missing stored state defaults to false; no grant is ever inferred.
    """
    merged = normalize_authorization_flags(previous)
    if delta is None:
        return merged
    values = delta.as_dict() if isinstance(delta, AuthorizationDelta) else dict(delta)
    for key in AUTHORIZATION_KEYS:
        state = str(values.get(key, UNCHANGED) or UNCHANGED)
        if state == GRANT:
            merged[key] = True
        elif state == REVOKE:
            merged[key] = False
    return {key: bool(merged.get(key, False)) for key in AUTHORIZATION_KEYS}


# --------------------------------------------------------------------------
# Activation lifecycle
# --------------------------------------------------------------------------
ACTIVATION_REQUESTED = "REQUESTED"
ACTIVATION_PENDING = "PENDING_AUTHORIZATION"
ACTIVATION_PENDING_REGISTRY = "PENDING_REGISTRY"
ACTIVATION_UNAVAILABLE = "UNAVAILABLE_UPSTREAM"
ACTIVATION_ELIGIBLE = "ELIGIBLE"
ACTIVATION_BOUND = "BOUND"
ACTIVATION_REVOKED = "REVOKED"
ACTIVATION_BLOCKED = "BLOCKED_POLICY"

ACTIVATION_STATES: frozenset[str] = frozenset({
    ACTIVATION_REQUESTED,
    ACTIVATION_PENDING,
    ACTIVATION_PENDING_REGISTRY,
    ACTIVATION_UNAVAILABLE,
    ACTIVATION_ELIGIBLE,
    ACTIVATION_BOUND,
    ACTIVATION_REVOKED,
    ACTIVATION_BLOCKED,
})


def _raw_tool_name(extra_tool: str) -> str:
    value = str(extra_tool or "")
    return value.split(":", 1)[1] if ":" in value else value


def classify_activation_state(
    extra_tool: str,
    *,
    available: Collection[str] = (),
    eligible: Collection[str] = (),
    pending: Collection[str] = (),
    pending_registry: Collection[str] = (),
    bound_tool_names: Collection[str] = (),
    previously_eligible: Collection[str] = (),
    blocked: Collection[str] = (),
) -> str:
    """The single authoritative activation classifier.

    Precedence is deliberate and fixed:

        BLOCKED_POLICY > PENDING_REGISTRY > UNAVAILABLE_UPSTREAM > BOUND
                       > ELIGIBLE > PENDING_AUTHORIZATION > REVOKED > REQUESTED

    ``UNAVAILABLE_UPSTREAM`` outranks ``PENDING_AUTHORIZATION`` on purpose: a
    tool that does not exist upstream is not waiting on an authorization
    decision, and must never be reported as pending, submitted, or in flight.
    """
    name = canonical_extra_tool(str(extra_tool or ""))
    raw = _raw_tool_name(name)

    if name in set(blocked) or raw in set(blocked):
        return ACTIVATION_BLOCKED
    if name in set(pending_registry) or raw in set(pending_registry):
        return ACTIVATION_PENDING_REGISTRY
    if name not in set(available):
        return ACTIVATION_UNAVAILABLE
    if raw in set(bound_tool_names) or name in set(bound_tool_names):
        return ACTIVATION_BOUND
    if name in set(eligible):
        return ACTIVATION_ELIGIBLE
    if name in set(pending):
        return ACTIVATION_PENDING
    if name in set(previously_eligible):
        return ACTIVATION_REVOKED
    return ACTIVATION_REQUESTED


# --------------------------------------------------------------------------
# Reducers
# --------------------------------------------------------------------------
def _clean_name(value: Any) -> str:
    name = str(value or "").strip()
    return name[:MAX_NAME_LENGTH]


def normalize_name_list(values: Iterable[Any] | None, limit: int) -> list[str]:
    """Normalized, deduplicated, stably ordered, deterministically truncated."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values or ():
        name = _clean_name(value)
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    out.sort()
    return out[: max(0, int(limit))]


def make_ordered_set_reducer(limit: int):
    """Bounded, idempotent, order-stable union reducer."""

    def _reduce(left: Any, right: Any) -> list[str]:
        if right is None:
            return normalize_name_list(left, limit)
        return normalize_name_list([*(left or ()), *(right or ())], limit)

    return _reduce


def make_replacement_reducer(limit: int | None = None):
    """Replace-on-write reducer for fields that must not accumulate."""

    def _reduce(left: Any, right: Any) -> list[str]:
        if right is None:
            return normalize_name_list(left, limit or MAX_LAST_BOUND_TOOL_NAMES)
        return normalize_name_list(right, limit or MAX_LAST_BOUND_TOOL_NAMES)

    return _reduce


def reduce_authorization_flags(left: Any, right: Any) -> dict[str, bool]:
    """Authorization is replaced by the already-merged effective state.

    The delta merge happens in the graph node, not in the reducer, so that a
    revocation cannot be re-widened by a later reducer pass.
    """
    if right is None:
        return normalize_authorization_flags(left)
    return normalize_authorization_flags(right)


def reduce_activation_status(left: Any, right: Any) -> dict[str, str]:
    """Bounded mapping replacement with enumerated value clamping."""
    source = right if right is not None else left
    if not isinstance(source, Mapping):
        return {}
    out: dict[str, str] = {}
    for key in sorted(str(k) for k in source):
        if len(out) >= MAX_ACTIVATION_STATUS_ENTRIES:
            break
        value = str(source.get(key) or "")
        out[_clean_name(key)] = value if value in ACTIVATION_STATES else ACTIVATION_REQUESTED
    return out


def reduce_scalar(left: Any, right: Any) -> str:
    if right is None:
        return _clean_name(left)
    return _clean_name(right)


def reduce_bool(left: Any, right: Any) -> bool:
    """Replace a checkpointed boolean; missing writes preserve prior state."""
    return bool(left) if right is None else bool(right)


def reduce_version(left: Any, right: Any) -> int:
    value = right if right is not None else left
    try:
        return int(value)
    except (TypeError, ValueError):
        return TOOL_POLICY_VERSION


def reduce_revision(left: Any, right: Any) -> int:
    """Monotonic bounded activation-request revision."""
    try:
        old = max(0, int(left or 0))
    except (TypeError, ValueError):
        old = 0
    try:
        new = max(0, int(right if right is not None else old))
    except (TypeError, ValueError):
        new = old
    return min(max(old, new), 2_147_483_647)


# --------------------------------------------------------------------------
# Defaults and migration
# --------------------------------------------------------------------------
POLICY_STATE_KEYS: tuple[str, ...] = (
    "active_toolsets",
    "requested_toolsets",
    "requested_extra_tools",
    "processed_activation_tool_call_ids",
    "eligible_extra_tools",
    "pending_authorization_extra_tools",
    "pending_registry_requests",
    "unavailable_extra_tools",
    "authorization_flags",
    "mcop_children_forbidden",
    "last_bound_tool_names",
    "tool_profile_signature",
    "tool_registry_generation",
    # D3B1 v2 fields: semantic content identity, process-local refresh epoch,
    # and current health are recorded separately.  Only the content signature
    # participates in profile identity.
    "tool_registry_content_signature",
    "tool_registry_refresh_epoch",
    "tool_registry_health_signature",
    "last_activation_status",
    "activation_request_revision",
    "continuity_task_scope",
    "continuity_methodology",
    "continuity_environment",
    "continuity_revision",
    "tool_policy_version",
)


def default_tool_policy_state() -> dict[str, Any]:
    """Safe defaults for a thread with no D1 state.

    Every authorization boolean is false.  This is the value an old checkpoint
    loads as; nothing is inferred from its message history.
    """
    return {
        "active_toolsets": [],
        "requested_toolsets": [],
        "requested_extra_tools": [],
        "processed_activation_tool_call_ids": [],
        "eligible_extra_tools": [],
        "pending_authorization_extra_tools": [],
        "pending_registry_requests": [],
        "unavailable_extra_tools": [],
        "authorization_flags": dict.fromkeys(AUTHORIZATION_KEYS, False),
        "mcop_children_forbidden": False,
        "last_bound_tool_names": [],
        "tool_profile_signature": "",
        "tool_registry_generation": "",
        "tool_registry_content_signature": "",
        "tool_registry_refresh_epoch": 0,
        "tool_registry_health_signature": "",
        "last_activation_status": {},
        "activation_request_revision": 0,
        "continuity_task_scope": "",
        "continuity_methodology": "",
        "continuity_environment": "",
        "continuity_revision": 0,
        "tool_policy_version": TOOL_POLICY_VERSION,
    }


def load_tool_policy_state(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read D1 policy state from graph state, tolerating old checkpoints.

    Unknown or missing keys fall back to safe defaults.  Unrecognized extra
    keys are ignored rather than raising, so a checkpoint written by a newer
    reader does not crash an older one.
    """
    loaded = default_tool_policy_state()
    if not isinstance(state, Mapping):
        return loaded
    for key in (
        "active_toolsets",
        "requested_toolsets",
        "requested_extra_tools",
        "processed_activation_tool_call_ids",
        "eligible_extra_tools",
        "pending_authorization_extra_tools",
        "pending_registry_requests",
        "unavailable_extra_tools",
    ):
        if state.get(key) is not None:
            loaded[key] = normalize_name_list(state.get(key), FIELD_BOUNDS[key])
    if state.get("last_bound_tool_names") is not None:
        loaded["last_bound_tool_names"] = normalize_name_list(
            state.get("last_bound_tool_names"), MAX_LAST_BOUND_TOOL_NAMES
        )
    if state.get("authorization_flags") is not None:
        loaded["authorization_flags"] = normalize_authorization_flags(state.get("authorization_flags"))
    if state.get("mcop_children_forbidden") is not None:
        loaded["mcop_children_forbidden"] = bool(state.get("mcop_children_forbidden"))
    if state.get("last_activation_status") is not None:
        loaded["last_activation_status"] = reduce_activation_status(None, state.get("last_activation_status"))
    for key in (
        "tool_profile_signature",
        "tool_registry_generation",
        "tool_registry_content_signature",
        "tool_registry_health_signature",
        "continuity_task_scope",
        "continuity_methodology",
        "continuity_environment",
    ):
        if state.get(key) is not None:
            loaded[key] = _clean_name(state.get(key))
    # D3B1 migration.  A v1 checkpoint carries only tool_registry_generation,
    # which already held the semantic inventory content signature.  It is read
    # as content identity and is never treated as a trusted refresh epoch.
    # Authorization state and requested tools are untouched by this migration,
    # and nothing is rebuilt by scanning old messages.
    if not loaded["tool_registry_content_signature"] and loaded["tool_registry_generation"]:
        loaded["tool_registry_content_signature"] = loaded["tool_registry_generation"]
    raw_epoch = state.get("tool_registry_refresh_epoch")
    if raw_epoch is not None:
        try:
            loaded["tool_registry_refresh_epoch"] = max(0, int(raw_epoch))
        except (TypeError, ValueError):
            loaded["tool_registry_refresh_epoch"] = 0
    # LangGraph initializes an empty Annotated[int, ...] channel to 0 rather
    # than None, so a falsy version means "absent", not "version zero".
    # Version 0 is not a valid policy version.
    if state.get("activation_request_revision") is not None:
        loaded["activation_request_revision"] = reduce_revision(
            None, state.get("activation_request_revision")
        )
    if state.get("continuity_revision") is not None:
        loaded["continuity_revision"] = reduce_revision(
            None, state.get("continuity_revision")
        )
    if state.get("tool_policy_version"):
        loaded["tool_policy_version"] = reduce_version(None, state.get("tool_policy_version"))
    return loaded


# --------------------------------------------------------------------------
# Upstream availability
# --------------------------------------------------------------------------
def resolve_upstream_availability(extra_tools: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split requested exact tools into (available, unavailable) upstream.

    Availability is decided by the real registry inventory for the owning
    family.  A registry failure is treated as "not proven available" so an
    absent tool is never reported as pending authorization.
    """
    available: list[str] = []
    unavailable: list[str] = []
    by_owner: dict[str, list[str]] = {}
    for value in extra_tools or ():
        name = canonical_extra_tool(str(value or ""))
        if not name:
            continue
        owner = name.split(":", 1)[0] if ":" in name else ""
        by_owner.setdefault(owner, []).append(name)

    for owner, names in by_owner.items():
        present: set[str] = set()
        try:
            from app.agent.tool_execution_policy import get_tools_for_toolsets

            for tool in get_tools_for_toolsets([owner]) or ():
                tool_name = str(getattr(tool, "name", "") or "")
                if tool_name:
                    present.add(tool_name)
        except Exception:  # noqa: BLE001 - fail closed to "unavailable"
            present = set()
        for name in names:
            (available if _raw_tool_name(name) in present else unavailable).append(name)

    return sorted(set(available)), sorted(set(unavailable))


def current_registry_content_signature() -> str:
    """Semantic content identity of the effective tool inventory.

    This is the only registry value permitted to influence a tool-profile
    signature.  It is stable across refresh epochs, restarts, and transient
    upstream failures that still hold a validated last-known-good baseline.
    """
    try:
        from app.agent.agents.tools.registry import get_registry_content_signature

        return _clean_name(get_registry_content_signature())
    except Exception:  # noqa: BLE001
        return ""


def current_registry_refresh_epoch() -> int:
    """Process-local monotonic refresh counter.  Never a semantic identity."""
    try:
        from app.agent.agents.tools.registry import get_registry_refresh_epoch

        return int(get_registry_refresh_epoch())
    except Exception:  # noqa: BLE001
        return 0


def current_registry_health_signature() -> str:
    """Current per-family health summary.  Never a semantic identity."""
    try:
        from app.agent.agents.tools.registry import get_registry_health_signature

        return _clean_name(get_registry_health_signature())
    except Exception:  # noqa: BLE001
        return ""


def current_registry_generation() -> str:
    """Backward-compatible alias for the semantic content signature."""
    return current_registry_content_signature()


# --------------------------------------------------------------------------
# Deterministic signature
# --------------------------------------------------------------------------
def compute_policy_signature(
    *,
    active_toolsets: Sequence[str],
    eligible_extra_tools: Sequence[str],
    registry_generation: str,
    authorization_flags: Mapping[str, Any] | None = None,
    requested_toolsets: Sequence[str] = (),
    requested_extra_tools: Sequence[str] = (),
    activation_request_revision: int = 0,
    mcop_children_forbidden: bool = False,
    policy_version: int = TOOL_POLICY_VERSION,
) -> str:
    """Deterministic signature over policy inputs only.

    Deliberately excludes timestamps, request ids, thread ids, chat ids,
    prompts, authorization material, audit ids, and Python object identity.
    Ordering is normalized so equivalent sets in any input order agree.
    """
    material = {
        "schema": TOOL_POLICY_STATE_SCHEMA,
        "policy_version": int(policy_version),
        "active_toolsets": sorted({_clean_name(v) for v in active_toolsets or () if _clean_name(v)}),
        "requested_toolsets": sorted({_clean_name(v) for v in requested_toolsets or () if _clean_name(v)}),
        "requested_extra_tools": sorted({_clean_name(v) for v in requested_extra_tools or () if _clean_name(v)}),
        "eligible_extra_tools": sorted({_clean_name(v) for v in eligible_extra_tools or () if _clean_name(v)}),
        "activation_request_revision": max(0, int(activation_request_revision or 0)),
        "registry_generation": _clean_name(registry_generation),
        "authorization_flags": normalize_authorization_flags(authorization_flags),
        "mcop_children_forbidden": bool(mcop_children_forbidden),
    }
    return profile_signature(material)
