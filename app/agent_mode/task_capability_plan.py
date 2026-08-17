"""Typed capability contract and preflight for MCOP child tasks.

A child may narrow an already-authorized parent surface, but it may never invent
missing capabilities.  The decision is value-only and can be made before any
model call or child graph construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any, Iterable, Mapping, Sequence

MAX_REQUIRED_TOOLSETS = 24
MAX_REQUIRED_TOOLS = 48
MAX_REQUIRED_CAPABILITIES = 48
MAX_ARTIFACT_TYPES = 24
MAX_NAME_LENGTH = 128

BLOCKED_CHILD_MISSING_CAPABILITY = "BLOCKED_CHILD_MISSING_CAPABILITY"
PREFLIGHT_PASS = "CHILD_CAPABILITY_PREFLIGHT_PASS"


def _clean(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.:/-]+", "_", str(value or "").strip())[:MAX_NAME_LENGTH].strip("_")


def _bounded(values: Iterable[Any], limit: int) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = _clean(value)
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return tuple(sorted(out)[:limit])


def parse_string_list(value: Any, *, limit: int) -> tuple[str, ...]:
    """Parse a JSON array, CSV string, or sequence without executing content."""
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = []
            if isinstance(parsed, list):
                return _bounded(parsed, limit)
        return _bounded((part.strip() for part in text.split(",")), limit)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return _bounded(value, limit)
    return ()


@dataclass(frozen=True)
class TaskCapabilityPlan:
    required_toolsets: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    expected_artifact_types: tuple[str, ...] = ()
    write_required: bool = False
    network_required: bool = False
    repository_access_required: bool = False
    plan_required: bool = True

    @classmethod
    def from_values(
        cls,
        *,
        required_toolsets: Any = (),
        required_tools: Any = (),
        required_capabilities: Any = (),
        expected_artifact_types: Any = (),
        write_required: bool = False,
        network_required: bool = False,
        repository_access_required: bool = False,
        plan_required: bool = True,
    ) -> "TaskCapabilityPlan":
        artifacts = parse_string_list(expected_artifact_types, limit=MAX_ARTIFACT_TYPES)
        return cls(
            required_toolsets=parse_string_list(required_toolsets, limit=MAX_REQUIRED_TOOLSETS),
            required_tools=parse_string_list(required_tools, limit=MAX_REQUIRED_TOOLS),
            required_capabilities=parse_string_list(required_capabilities, limit=MAX_REQUIRED_CAPABILITIES),
            expected_artifact_types=artifacts,
            write_required=bool(write_required or artifacts),
            network_required=bool(network_required),
            repository_access_required=bool(repository_access_required),
            plan_required=bool(plan_required),
        )

    @property
    def declared(self) -> bool:
        return bool(
            self.required_toolsets
            or self.required_tools
            or self.required_capabilities
            or self.expected_artifact_types
            or self.write_required
            or self.network_required
            or self.repository_access_required
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChildCapabilityDecision:
    allowed: bool
    status: str
    missing_tools: tuple[str, ...] = ()
    missing_toolsets: tuple[str, ...] = ()
    missing_capabilities: tuple[str, ...] = ()
    parent_bound_tools: tuple[str, ...] = ()
    child_candidate_tools: tuple[str, ...] = ()
    required_plan: Mapping[str, Any] = field(default_factory=dict)
    llm_call_permitted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _raw(name: str) -> str:
    return str(name or "").split(":", 1)[-1]


def capabilities_for_tool(name: str) -> frozenset[str]:
    """Return conservative capabilities for one concrete tool name."""
    canonical = str(name or "")
    raw = _raw(canonical)
    capabilities: set[str] = {"tool_execution"}
    try:
        from app.agent.tool_profiles import EXACT_TOOL_CAPABILITIES, code_execution_capability

        code = code_execution_capability(canonical)
        if code:
            capabilities.update(str(value) for value in code.get("capabilities", ()))
        for qualified, metadata in EXACT_TOOL_CAPABILITIES.items():
            if qualified == canonical or qualified.endswith(":" + raw):
                capabilities.update(str(value) for value in metadata.get("capabilities", ()))
    except Exception:
        pass

    low = raw.lower()
    if low.startswith(("list_", "get_", "read_", "search_", "query_", "validate_", "compare_")):
        capabilities.add("read_only")
    if raw in {"agent_run_shell", "agent_git_clone"}:
        capabilities.add("repository_access")
    if raw == "agent_git_clone":
        capabilities.add("repository_clone")
    if "upload" in low or "submit" in low or "persist" in low or "write" in low or "create_" in low:
        capabilities.add("mutation")
    if ":" in canonical or low.startswith(("s3_", "qos_", "rtr_", "grasshopper_")):
        capabilities.add("remote_service_access")
    return frozenset(capabilities)


def _name_is_present(required: str, candidate_names: set[str]) -> bool:
    return required in candidate_names or _raw(required) in candidate_names


def evaluate_child_capability_plan(
    *,
    plan: TaskCapabilityPlan,
    parent_policy: Any = None,
    candidate_tool_names: Sequence[str] = (),
) -> ChildCapabilityDecision:
    """Evaluate the declared child plan against the actual usable binding."""
    candidates = _bounded(candidate_tool_names, 256)
    candidate_set = set(candidates) | {_raw(name) for name in candidates}
    parent_tools: tuple[str, ...] = ()
    parent_toolsets: set[str] = set()
    if parent_policy is not None:
        parent_tools = _bounded(getattr(parent_policy, "eligible_extra_tools", ()) or (), 256)
        parent_toolsets = set(getattr(parent_policy, "eligible_toolsets", ()) or ())

    missing_tools = tuple(sorted(
        required for required in plan.required_tools
        if not _name_is_present(required, candidate_set)
    ))
    missing_toolsets = tuple(sorted(set(plan.required_toolsets) - parent_toolsets))

    available_capabilities: set[str] = set()
    for name in candidates:
        available_capabilities.update(capabilities_for_tool(name))
    required_capabilities = set(plan.required_capabilities)
    if plan.write_required:
        required_capabilities.add("filesystem_write")
    if plan.network_required:
        required_capabilities.add("network_egress")
    if plan.repository_access_required:
        required_capabilities.add("repository_access")

    # A remote MCP call is a valid bounded network capability even though the
    # HTTP transport is hidden behind the tool executor.
    if "network_egress" in required_capabilities and "remote_service_access" in available_capabilities:
        required_capabilities.remove("network_egress")
    # Persistent artifacts/bundle writes satisfy a declared write requirement.
    if "filesystem_write" in required_capabilities and available_capabilities.intersection({
        "filesystem_write", "persistent_artifacts", "bundle_write", "capsule_persist", "scene_persist"
    }):
        required_capabilities.remove("filesystem_write")

    missing_capabilities = tuple(sorted(required_capabilities - available_capabilities))
    no_declared_plan = bool(plan.plan_required and not plan.declared)
    no_usable_tools = not candidates
    allowed = not (missing_tools or missing_toolsets or missing_capabilities or no_declared_plan or no_usable_tools)
    if no_declared_plan:
        missing_capabilities = tuple(sorted(set(missing_capabilities) | {"DECLARED_CAPABILITY_PLAN"}))
    if no_usable_tools:
        missing_capabilities = tuple(sorted(set(missing_capabilities) | {"USABLE_CHILD_TOOL_BINDING"}))
    return ChildCapabilityDecision(
        allowed=allowed,
        status=PREFLIGHT_PASS if allowed else BLOCKED_CHILD_MISSING_CAPABILITY,
        missing_tools=missing_tools,
        missing_toolsets=missing_toolsets,
        missing_capabilities=missing_capabilities,
        parent_bound_tools=parent_tools,
        child_candidate_tools=candidates,
        required_plan=plan.to_dict(),
        llm_call_permitted=allowed,
    )
