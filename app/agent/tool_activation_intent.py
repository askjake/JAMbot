"""Typed, bounded activation intent for dynamic tool families.

Activation requests can originate from the current user turn, from an executed
management-facade ToolMessage, or from checkpointed state.  This module keeps
those sources structurally distinct and resolves names only against a bounded
registry snapshot.  Assistant-authored JSON is deliberately not accepted as a
management result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from app.agent.directive_text import sanitize_current_turn_directives

SOURCE_CURRENT_USER_PROMPT = "CURRENT_USER_PROMPT"
SOURCE_MANAGEMENT_FACADE_RESULT = "MANAGEMENT_FACADE_RESULT"
SOURCE_CHECKPOINTED_STATE = "CHECKPOINTED_STATE"
SOURCE_NONE = "NONE"
VALID_SOURCES = frozenset({
    SOURCE_CURRENT_USER_PROMPT,
    SOURCE_MANAGEMENT_FACADE_RESULT,
    SOURCE_CHECKPOINTED_STATE,
    SOURCE_NONE,
})

MAX_REQUESTED_TOOLSETS = 32

_CAPABILITY_NAME_UNIVERSE_CACHE: "frozenset[str] | None" = None


def _capability_name_universe() -> "frozenset[str]":
    """Names that are per-tool *capabilities*, never registry *toolset*
    families -- requesting one as a toolset is a caller error (most often
    the model itself confusing "I need X capability" with "activate X
    toolset") and can never resolve via registry discovery, no matter how
    long it is polled.
    """
    names: set[str] = {
        # Heuristic capability tags from app.agent_mode.task_capability_plan
        "read_only", "repository_access", "repository_clone", "mutation",
        "remote_service_access",
        # Declared write/network/repo requirements from TaskCapabilityPlan
        "filesystem_write", "network_egress",
        # Persistence-adjacent capability satisfiers treated as equivalent
        # to filesystem_write in evaluate_child_capability_plan()
        "persistent_artifacts", "bundle_write", "capsule_persist", "scene_persist",
    }
    try:
        from app.agent import tool_profiles as _tp
        for attr_name in dir(_tp):
            value = getattr(_tp, attr_name, None)
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Mapping):
                        caps = item.get("capabilities")
                        if isinstance(caps, (list, tuple)):
                            names.update(str(c) for c in caps if c)
    except Exception:
        pass
    return frozenset(names)


def _is_known_capability_name(name: str) -> bool:
    global _CAPABILITY_NAME_UNIVERSE_CACHE
    if _CAPABILITY_NAME_UNIVERSE_CACHE is None:
        _CAPABILITY_NAME_UNIVERSE_CACHE = _capability_name_universe()
    return name in _CAPABILITY_NAME_UNIVERSE_CACHE
MAX_REQUESTED_EXACT_TOOLS = 64
MAX_AMBIGUOUS_REQUESTS = 16
MAX_NAME_LENGTH = 128

# A qualified name is deterministic.  An unqualified name is considered only
# when it follows an explicit activation verb; ordinary prose or copied logs do
# not silently request tools.
_QUALIFIED_RE = re.compile(r"\b([a-z][a-z0-9_]{1,63}):([A-Za-z][A-Za-z0-9_-]{1,127})\b")
_ACTIVATE_TOOLSET_RE = re.compile(
    r"\b(?:activate|enable|bind|request|load)\s+(?:the\s+)?(?:toolset\s+|family\s+)?"
    r"([a-z][a-z0-9_]{2,63})\b",
    re.IGNORECASE,
)
_ACTIVATE_BARE_TOOL_RE = re.compile(
    r"\b(?:bind|activate|enable|request|load|use)\s+(?:the\s+)?(?:exact\s+)?(?:tool\s+)?"
    r"([A-Za-z][A-Za-z0-9_-]{2,127})\b",
    re.IGNORECASE,
)

_ACTIVATION_VERB_RE = re.compile(r"\b(?:bind|activate|enable|request|load|use)\b", re.IGNORECASE)
_NEGATED_ACTIVATION_RE = re.compile(
    r"(?:\bdo\s+not|\bdon't|\bnever|\bnot|\bno)\s+(?:\w+\s+){0,3}$",
    re.IGNORECASE,
)
_BARE_AFTER_VERB_RE = re.compile(
    r"^\s*(?:the\s+)?(?:exact\s+)?(?:toolset\s+|family\s+|tool\s+)?"
    r"([A-Za-z][A-Za-z0-9_-]{2,127})\b",
    re.IGNORECASE,
)
_GENERIC_ACTIVATION_WORDS = frozenset({
    "exactly", "following", "these", "this", "tool", "tools", "toolset", "family", "the"
})


def _activation_match_is_negated(line: str, start: int) -> bool:
    clause_start = max(
        line.rfind(".", 0, start), line.rfind(";", 0, start),
        line.rfind("!", 0, start), line.rfind("?", 0, start),
    )
    prefix = line[clause_start + 1:start][-80:]
    return bool(_NEGATED_ACTIVATION_RE.search(prefix))


def _activation_directive_bodies(text: str | None) -> tuple[str, ...]:
    """Return positive activation directive bodies, including bounded lists."""
    clean = sanitize_current_turn_directives(text)
    bodies: list[str] = []
    list_mode = False
    blank_budget = 1
    for line in clean.splitlines():
        matches = [
            match for match in _ACTIVATION_VERB_RE.finditer(line)
            if not _activation_match_is_negated(line, match.start())
        ]
        if matches:
            for match in matches:
                body = line[match.end():]
                bodies.append(body)
                tail = body.strip().lower()
                if tail.endswith(":") or re.search(r"\b(?:exactly|following)\s*:?$", tail):
                    list_mode = True
                    blank_budget = 1
            continue
        stripped = line.strip()
        if list_mode and not stripped:
            if blank_budget > 0:
                blank_budget -= 1
                continue
            list_mode = False
            continue
        if list_mode and (_QUALIFIED_RE.search(line) or re.match(r"^\s*[-*]\s+", line)):
            bodies.append(line)
            continue
        if stripped:
            list_mode = False
    return tuple(bodies[:32])

_ALLOWED_HEALTH = frozenset({
    "HEALTHY",
    "DEGRADED_LAST_KNOWN_GOOD",
    # Local/static families do not always have an MCP health record.
    "LOCAL",
})


def _clean_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip())[:MAX_NAME_LENGTH].strip("_")


def _clean_family(value: Any) -> str:
    return _clean_name(value).lower()


def _dedupe(values: Iterable[Any], limit: int) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = str(value or "").strip()[:MAX_NAME_LENGTH]
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(sorted(out)[: max(0, int(limit))])


@dataclass(frozen=True)
class RegistryToolIndex:
    """Value-only, bounded view of the effective registry."""

    tools_by_family: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    family_health: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_material(
        cls,
        *,
        inventory: Sequence[Mapping[str, Any]] = (),
        family_status: Mapping[str, Any] | None = None,
    ) -> "RegistryToolIndex":
        grouped: dict[str, set[str]] = {}
        for row in list(inventory or ())[:4096]:
            if not isinstance(row, Mapping) or not bool(row.get("enabled", True)):
                continue
            family = _clean_family(row.get("toolset"))
            tool = _clean_name(row.get("tool_name"))
            if not family or not tool or tool.startswith("<"):
                continue
            grouped.setdefault(family, set()).add(tool)
        health = {
            _clean_family(key): str(value or "").upper()
            for key, value in dict(family_status or {}).items()
            if _clean_family(key)
        }
        # A static/local family present in inventory but absent from MCP health is
        # still a real registry family, not pending discovery.
        for family in grouped:
            health.setdefault(family, "LOCAL")
        return cls(
            tools_by_family={key: tuple(sorted(values)) for key, values in sorted(grouped.items())},
            family_health=dict(sorted(health.items())),
        )

    @classmethod
    def live(cls) -> "RegistryToolIndex":
        try:
            from app.agent.agents.tools.registry import get_mcp_registry_status, get_tool_inventory
            from app.agent.agents.tools import get_tools_set

            try:
                inventory_rows = get_tool_inventory(include_uninitialized=True)
            except TypeError:
                # Compatibility with test fixtures and older registry adapters
                # whose inventory function predates the keyword argument.
                inventory_rows = get_tool_inventory()
            raw_inventory = inventory_rows or ()
            if isinstance(raw_inventory, Mapping):
                # Accept both the canonical row list and the older
                # {toolsets: {family: [{name: ...}]}} materialization.
                normalized_inventory: list[Mapping[str, Any]] = []
                toolsets = raw_inventory.get("toolsets") or {}
                if isinstance(toolsets, Mapping):
                    for family, rows in toolsets.items():
                        for row in list(rows or ())[:512]:
                            if isinstance(row, Mapping):
                                normalized_inventory.append({
                                    "toolset": family,
                                    "tool_name": row.get("tool_name") or row.get("name"),
                                    "enabled": row.get("enabled", True),
                                })
                inventory = normalized_inventory
            else:
                inventory = list(raw_inventory)
            status = get_mcp_registry_status() or {}
            families = status.get("families") or {}
            health = {
                str(name): str((details or {}).get("health") or "")
                for name, details in dict(families).items()
            }
            # The model-facing registry adapter is also the most direct source
            # of an effective family surface.  Supplement the inventory from it
            # so a just-refreshed dynamic MCP family (and isolated registry
            # fixtures) can participate before a separate inventory snapshot is
            # rebuilt.  This is bounded and still validated by family health.
            known_families: set[str] = set(str(name) for name in dict(families))
            raw_toolsets = status.get("toolsets") or ()
            if isinstance(raw_toolsets, Mapping):
                known_families.update(str(name) for name in raw_toolsets)
            elif isinstance(raw_toolsets, (list, tuple, set, frozenset)):
                known_families.update(str(name) for name in raw_toolsets)
            for family in sorted(known_families)[:128]:
                try:
                    tools = list(get_tools_set(family) or ())[:512]
                except Exception:
                    tools = []
                for tool in tools:
                    name = str(getattr(tool, "name", "") or "")
                    if name:
                        inventory.append({
                            "toolset": family,
                            "tool_name": name,
                            "enabled": True,
                        })
            return cls.from_material(inventory=inventory, family_status=health)
        except Exception:
            return cls()

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self.tools_by_family))

    def family_known(self, family: str) -> bool:
        return _clean_family(family) in self.tools_by_family

    def family_usable(self, family: str) -> bool:
        clean = _clean_family(family)
        return clean in self.tools_by_family and str(self.family_health.get(clean) or "LOCAL").upper() in _ALLOWED_HEALTH

    def tools(self, family: str) -> tuple[str, ...]:
        return tuple(self.tools_by_family.get(_clean_family(family), ()))

    def qualified_candidates(self, raw_tool_name: str) -> tuple[str, ...]:
        raw = _clean_name(raw_tool_name)
        return tuple(sorted(
            f"{family}:{raw}"
            for family, tools in self.tools_by_family.items()
            if raw in set(tools)
        ))

    def resolve(self, value: str) -> tuple[str, str, tuple[str, ...]]:
        """Return ``(canonical, status, candidates)``.

        Status is one of RESOLVED, AMBIGUOUS, UNAVAILABLE, PENDING_REGISTRY, or
        FAMILY_UNHEALTHY.  Only RESOLVED names may become model-facing.
        """
        text = str(value or "").strip()
        if not text:
            return "", "UNAVAILABLE", ()
        if ":" in text:
            owner, raw = text.split(":", 1)
            owner, raw = _clean_family(owner), _clean_name(raw)
            canonical = f"{owner}:{raw}" if owner and raw else ""
            if not owner or not raw:
                return canonical, "UNAVAILABLE", ()
            if not self.family_known(owner):
                return canonical, "PENDING_REGISTRY", ()
            if raw not in set(self.tools(owner)):
                return canonical, "UNAVAILABLE", ()
            if not self.family_usable(owner):
                return canonical, "FAMILY_UNHEALTHY", ()
            return canonical, "RESOLVED", (canonical,)
        candidates = self.qualified_candidates(text)
        if len(candidates) == 1:
            owner = candidates[0].split(":", 1)[0]
            if not self.family_usable(owner):
                return candidates[0], "FAMILY_UNHEALTHY", candidates
            return candidates[0], "RESOLVED", candidates
        if len(candidates) > 1:
            return "", "AMBIGUOUS", candidates
        return "", "UNAVAILABLE", ()


@dataclass(frozen=True)
class ToolActivationIntent:
    requested_toolsets: tuple[str, ...] = ()
    requested_exact_tools: tuple[str, ...] = ()
    source: str = SOURCE_NONE
    request_revision: int = 0
    unavailable_requests: tuple[str, ...] = ()
    pending_registry_requests: tuple[str, ...] = ()
    unhealthy_requests: tuple[str, ...] = ()
    ambiguous_requests: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (
            self.requested_toolsets
            or self.requested_exact_tools
            or self.unavailable_requests
            or self.pending_registry_requests
            or self.unhealthy_requests
            or self.ambiguous_requests
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_toolsets": list(self.requested_toolsets),
            "requested_exact_tools": list(self.requested_exact_tools),
            "source": self.source,
            "request_revision": int(self.request_revision),
            "unavailable_requests": list(self.unavailable_requests),
            "pending_registry_requests": list(self.pending_registry_requests),
            "unhealthy_requests": list(self.unhealthy_requests),
            "ambiguous_requests": {key: list(value) for key, value in sorted(self.ambiguous_requests.items())},
        }


def _intent_from_raw(
    *,
    requested_toolsets: Iterable[Any] = (),
    requested_tools: Iterable[Any] = (),
    source: str,
    request_revision: int = 0,
    registry_index: RegistryToolIndex | None = None,
) -> ToolActivationIntent:
    registry = registry_index or RegistryToolIndex.live()
    toolsets: list[str] = []
    exact: list[str] = []
    unavailable: list[str] = []
    pending: list[str] = []
    unhealthy: list[str] = []
    ambiguous: dict[str, tuple[str, ...]] = {}

    for value in requested_toolsets:
        family = _clean_family(value)
        if not family:
            continue
        # D3B-FIX(2026-08-18): a request for a *tool capability* name (e.g.
        # "filesystem_write", "network_egress") masquerading as a toolset
        # family used to fall into `pending` -> PENDING_REGISTRY, which is
        # reserved for families that are merely not loaded *yet*. Capability
        # names are never registry families and will never resolve, so they
        # sat as PENDING_REGISTRY forever: activation_intent_from_checkpoint()
        # re-derives requested_toolsets from checkpointed state every turn,
        # so the same unresolvable name was re-requested every turn
        # indefinitely (observed: {"filesystem_write": "PENDING_REGISTRY"}
        # on every turn of a session, activation_request_revision frozen).
        # Route these to a terminal `unavailable` bucket instead, and do not
        # record them as a durable toolset request.
        if not registry.family_known(family) and _is_known_capability_name(family):
            unavailable.append(family)
            continue
        # The request is durable independently of current registry health.  A
        # transient discovery outage must not erase operator intent.
        toolsets.append(family)
        if registry.family_usable(family):
            continue
        if registry.family_known(family):
            unhealthy.append(family)
        else:
            pending.append(family)

    for value in requested_tools:
        original = str(value or "").strip()
        canonical, status, candidates = registry.resolve(original)
        # Qualified requests (and uniquely resolved bare requests) remain
        # checkpointable even while registry health/availability changes.
        if canonical and (":" in original or status in {"RESOLVED", "FAMILY_UNHEALTHY"}):
            exact.append(canonical)
        if status == "RESOLVED" and canonical:
            continue
        if status == "AMBIGUOUS":
            ambiguous[original[:MAX_NAME_LENGTH]] = tuple(candidates[:16])
        elif status == "PENDING_REGISTRY":
            pending.append(canonical or original)
        elif status == "FAMILY_UNHEALTHY":
            unhealthy.append(canonical or original)
        else:
            unavailable.append(canonical or original)

    return ToolActivationIntent(
        requested_toolsets=_dedupe(toolsets, MAX_REQUESTED_TOOLSETS),
        requested_exact_tools=_dedupe(exact, MAX_REQUESTED_EXACT_TOOLS),
        source=source if source in VALID_SOURCES else SOURCE_NONE,
        request_revision=max(0, int(request_revision or 0)),
        unavailable_requests=_dedupe(unavailable, MAX_REQUESTED_EXACT_TOOLS),
        pending_registry_requests=_dedupe(pending, MAX_REQUESTED_EXACT_TOOLS),
        unhealthy_requests=_dedupe(unhealthy, MAX_REQUESTED_EXACT_TOOLS),
        ambiguous_requests=dict(list(sorted(ambiguous.items()))[:MAX_AMBIGUOUS_REQUESTS]),
    )


def activation_intent_from_request(
    *,
    requested_toolsets: Iterable[Any] = (),
    requested_tools: Iterable[Any] = (),
    source: str = SOURCE_MANAGEMENT_FACADE_RESULT,
    request_revision: int = 0,
    registry_index: RegistryToolIndex | None = None,
) -> ToolActivationIntent:
    """Build a typed intent from a trusted structured request."""
    return _intent_from_raw(
        requested_toolsets=requested_toolsets,
        requested_tools=requested_tools,
        source=source,
        request_revision=request_revision,
        registry_index=registry_index,
    )


def activation_intent_from_checkpoint(
    policy_state: Mapping[str, Any] | None,
    *,
    registry_index: RegistryToolIndex | None = None,
) -> ToolActivationIntent:
    """Restore bounded trusted intent without scanning message history.

    Checkpoints written before the explicit pending-registry provenance field
    may contain incidental ``owner:name`` tokens.  Unknown legacy rows are
    dropped fail-closed.  New explicit unknown requests remain sticky because
    the transition records them in ``pending_registry_requests``.
    """
    state = dict(policy_state or {})
    registry = registry_index or RegistryToolIndex.live()
    trusted_pending = {
        str(value) for value in state.get("pending_registry_requests") or () if str(value)
    }
    trusted_tools: list[str] = []
    for value in state.get("requested_extra_tools") or ():
        original = str(value or "").strip()
        canonical, status, _candidates = registry.resolve(original)
        if status == "PENDING_REGISTRY" and (canonical or original) not in trusted_pending:
            continue
        trusted_tools.append(canonical or original)
    return _intent_from_raw(
        requested_toolsets=state.get("requested_toolsets") or (),
        requested_tools=trusted_tools,
        source=SOURCE_CHECKPOINTED_STATE,
        request_revision=int(state.get("activation_request_revision") or 0),
        registry_index=registry,
    )


def activation_intent_from_prompt(
    prompt: str | None,
    *,
    registry_index: RegistryToolIndex | None = None,
    request_revision: int = 0,
) -> ToolActivationIntent:
    """Extract only explicit, positive current-turn activation directives.

    Qualified tokens appearing in URLs, examples, copied logs, schemas, quoted
    prompts, or negated instructions are ordinary data and never become sticky
    requests.  A genuinely explicit unknown family remains PENDING_REGISTRY.
    """

    registry = registry_index or RegistryToolIndex.live()
    bodies = _activation_directive_bodies(prompt)
    qualified: list[str] = []
    toolsets: list[str] = []
    bare_tools: list[str] = []

    for body in bodies:
        # Multiline exact-tool lists commonly use Markdown bullets.  Strip only
        # the list marker; the directive sanitizer and explicit list-mode gate
        # have already established that this body belongs to a positive
        # activation request.
        directive_body = re.sub(r"^\s*[-*]\s+", "", body)
        for owner, name in _QUALIFIED_RE.findall(directive_body):
            clean_owner = _clean_family(owner)
            # Unknown qualified names are checkpointable only when they use the
            # explicit dynamic-family convention. This preserves legitimate
            # ``newserver_mcp:tool`` discovery while rejecting semantic
            # non-families such as mailto:, key:, owner:, and toolset:.
            if registry.family_known(clean_owner) or clean_owner.endswith("_mcp"):
                qualified.append(f"{clean_owner}:{_clean_name(name)}")
        match = _BARE_AFTER_VERB_RE.search(directive_body)
        if not match:
            continue
        token = _clean_name(match.group(1))
        if not token or token.lower() in _GENERIC_ACTIVATION_WORDS:
            continue
        suffix = directive_body[match.end(1):]
        if suffix.lstrip().startswith(":"):
            continue
        if token.lower().endswith("_mcp") or registry.family_known(token):
            toolsets.append(_clean_family(token))
        else:
            bare_tools.append(token)

    return _intent_from_raw(
        requested_toolsets=toolsets,
        requested_tools=[*qualified, *bare_tools],
        source=SOURCE_CURRENT_USER_PROMPT,
        request_revision=request_revision,
        registry_index=registry,
    )


def _find_activation_payload(value: Any) -> Mapping[str, Any] | None:
    """Find one activation schema inside supported structured tool content."""
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump()
        except Exception:
            pass
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if isinstance(value, Mapping):
        if str(value.get("schema") or "") == "diship_backend_tool_activation_request.v1":
            return value
        for child in value.values():
            found = _find_activation_payload(child)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _find_activation_payload(child)
            if found is not None:
                return found
    return None


def _tool_message_payload(message: Any) -> Mapping[str, Any] | None:
    # Structural evidence boundary: only an actual ToolMessage object (or a
    # framework object whose declared type is "tool") may contribute. An
    # AIMessage containing tool-like JSON is ignored.
    type_name = type(message).__name__
    declared_type = str(getattr(message, "type", "") or "").lower()
    if type_name != "ToolMessage" and declared_type != "tool":
        return None
    # Some LangChain ToolMessage versions omit ``name`` while retaining the
    # authoritative tool_call_id and result payload.  Reject a conflicting
    # explicit name, but allow an absent name when the executed-tool message
    # contains the exact management schema.
    message_name = str(getattr(message, "name", "") or "")
    if message_name and message_name != "diship_backend_activate_tool_binding":
        return None
    content = getattr(message, "content", None)
    return _find_activation_payload(content)


def activation_intent_from_management_messages(
    messages: Sequence[Any] | None,
    *,
    registry_index: RegistryToolIndex | None = None,
    request_revision: int = 0,
) -> ToolActivationIntent:
    registry = registry_index or RegistryToolIndex.live()
    intents: list[ToolActivationIntent] = []
    for message in list(messages or ())[-64:]:
        payload = _tool_message_payload(message)
        if payload is None:
            continue
        intents.append(_intent_from_raw(
            requested_toolsets=payload.get("requested_toolsets") or (),
            requested_tools=payload.get("requested_extra_tools") or (),
            source=SOURCE_MANAGEMENT_FACADE_RESULT,
            request_revision=request_revision,
            registry_index=registry,
        ))
    return merge_activation_intents(*intents)


def merge_activation_intents(*intents: ToolActivationIntent) -> ToolActivationIntent:
    real = [intent for intent in intents if isinstance(intent, ToolActivationIntent)]
    if not real:
        return ToolActivationIntent()
    sources = [intent.source for intent in real if intent.source != SOURCE_NONE]
    source = sources[-1] if sources else SOURCE_NONE
    revision = max((int(intent.request_revision or 0) for intent in real), default=0)
    checkpoint_sets = {
        value for intent in real if intent.source == SOURCE_CHECKPOINTED_STATE
        for value in intent.requested_toolsets
    }
    checkpoint_tools = {
        value for intent in real if intent.source == SOURCE_CHECKPOINTED_STATE
        for value in intent.requested_exact_tools
    }
    noncheckpoint_sets = {
        value for intent in real if intent.source != SOURCE_CHECKPOINTED_STATE
        for value in intent.requested_toolsets
    }
    noncheckpoint_tools = {
        value for intent in real if intent.source != SOURCE_CHECKPOINTED_STATE
        for value in intent.requested_exact_tools
    }
    # Replaying a retained management ToolMessage does not advance state.  The
    # revision changes only when the accepted request set gains a new member.
    if (noncheckpoint_sets - checkpoint_sets) or (noncheckpoint_tools - checkpoint_tools):
        revision += 1
    ambiguous: dict[str, tuple[str, ...]] = {}
    for intent in real:
        ambiguous.update(dict(intent.ambiguous_requests or {}))
    return ToolActivationIntent(
        requested_toolsets=_dedupe(
            (name for intent in real for name in intent.requested_toolsets),
            MAX_REQUESTED_TOOLSETS,
        ),
        requested_exact_tools=_dedupe(
            (name for intent in real for name in intent.requested_exact_tools),
            MAX_REQUESTED_EXACT_TOOLS,
        ),
        source=source,
        request_revision=revision,
        unavailable_requests=_dedupe(
            (name for intent in real for name in intent.unavailable_requests),
            MAX_REQUESTED_EXACT_TOOLS,
        ),
        pending_registry_requests=_dedupe(
            (name for intent in real for name in intent.pending_registry_requests),
            MAX_REQUESTED_EXACT_TOOLS,
        ),
        unhealthy_requests=_dedupe(
            (name for intent in real for name in intent.unhealthy_requests),
            MAX_REQUESTED_EXACT_TOOLS,
        ),
        ambiguous_requests=dict(list(sorted(ambiguous.items()))[:MAX_AMBIGUOUS_REQUESTS]),
    )
