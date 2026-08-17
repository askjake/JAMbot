"""Cache-stable dynamic tool profiles for the active methodology-scoped agent.

This module is deliberately independent of LangChain.  It computes a
serializable, deterministic binding policy; the active registry and graph adapt
that policy to runtime tool objects.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from app.agent.tool_activation_intent import (
    RegistryToolIndex,
    activation_intent_from_prompt,
)

from app.agent.continuity_policy import ContinuityCheckpoint, resolve_continuity
from app.agent.execution_constraints import (
    merge_mcop_children_forbidden,
    parse_mcop_constraint_delta,
)
from app.agent.operational_workflows import (
    REPO_CHECKOUT_LOCAL_DEPLOY,
    detect_operational_workflow,
    operational_tool_owner,
    operational_tools_in,
)

TOOL_PROFILE_SCHEMA = "diship_tool_profile.v1"
TOOL_PROFILE_POLICY_VERSION = "dynamic_tool_profile_policy.v1"

AUTHORIZATION_DEFAULTS: dict[str, bool] = {
    "operator_authorized": False,
    "heavy_tools_authorized": False,
    "persistence_authorized": False,
    "mutation_authorized": False,
}

# Permanently small, cache-stable families.  Domain families are additive.
STABLE_CORE_TOOLSETS: tuple[str, ...] = (
    "search",
    "agent_mode",
    "backend_facades",
)

# The full S3 inventory remains in the executor registry.  Only this compact
# read/triage subset is model-facing until an exact extra is requested.
S3_CURATED_CORE_TOOLS: tuple[str, ...] = (
    "get_tool_info",
    "get_heavy_auth_status",
    "list_dates",
    "list_files",
    "search_logs",
    "get_summary",
    "list_log_capsules",
    "list_incident_scenes",
)

# Tool capability metadata is used by model-facing eligibility and later by the
# independent execution gate.  Build tools are eligible after heavy
# authorization; persistence is separately enforced from actual arguments.
EXACT_TOOL_CAPABILITIES: dict[str, dict[str, Any]] = {
    "s3_stb_logs:build_log_capsule": {
        "risk": "heavy_compute",
        "required_authorizations": ("heavy_tools_authorized",),
        "capabilities": ("capsule_build", "capsule_persist"),
    },
    "s3_stb_logs:build_complete_log_capsule": {
        "risk": "heavy_persistent",
        "required_authorizations": ("heavy_tools_authorized", "persistence_authorized"),
        "capabilities": ("capsule_build", "capsule_persist"),
    },
    "s3_stb_logs:build_incident_scene": {
        "risk": "heavy_compute",
        "required_authorizations": ("heavy_tools_authorized",),
        "capabilities": ("scene_build", "scene_persist", "scene_render"),
    },
    "s3_stb_logs:query_incident_scene": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("scene_query",),
    },
    "s3_stb_logs:expand_scene_region": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("scene_query", "evidence_expand"),
    },
    "s3_stb_logs:render_incident_scene": {
        "risk": "bounded_compute",
        "required_authorizations": (),
        "capabilities": ("scene_render",),
    },
    "s3_stb_logs:compare_incident_scenes": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("scene_compare",),
    },
    "s3_stb_logs:validate_incident_scene": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("scene_validate",),
    },
    "s3_stb_logs:get_incident_scene_manifest": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("scene_query",),
    },
    "s3_stb_logs:get_timeline": {
        "risk": "heavy_compute",
        "required_authorizations": ("heavy_tools_authorized",),
        "capabilities": ("timeline_build",),
    },
    "s3_stb_logs:summarize_log_patterns": {
        "risk": "heavy_compute",
        "required_authorizations": ("heavy_tools_authorized",),
        "capabilities": ("capsule_build",),
    },
    "s3_stb_logs:filter_log_lines": {
        "risk": "heavy_compute",
        "required_authorizations": ("heavy_tools_authorized",),
        "capabilities": ("raw_evidence_filter",),
    },
    "s3_stb_logs:compare_log_capsules": {
        "risk": "heavy_compute",
        "required_authorizations": ("heavy_tools_authorized",),
        "capabilities": ("capsule_compare",),
    },
    "s3_stb_logs:create_log_bundle": {
        "risk": "heavy_persistent",
        "required_authorizations": ("heavy_tools_authorized", "persistence_authorized"),
        "capabilities": ("bundle_write",),
    },
    "grasshopper_mcp:grasshopper_health": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("service_health",),
    },
    "grasshopper_mcp:grasshopper_get_log_type_catalog": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("log_type_catalog",),
    },
    "grasshopper_mcp:grasshopper_get_resource": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("resource_read",),
    },
    "grasshopper_mcp:grasshopper_list_file_groups": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("uploadable_inventory_read",),
    },
    "grasshopper_mcp:grasshopper_list_files": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("uploadable_inventory_read",),
    },
    "grasshopper_mcp:grasshopper_list_uploadable_files": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("uploadable_inventory_read",),
    },
    "grasshopper_mcp:grasshopper_plan_profile_upload": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("log_upload_plan",),
    },
    "grasshopper_mcp:grasshopper_get_upload_history": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("upload_history_read", "status_introspection"),
    },
    "grasshopper_mcp:grasshopper_get_upload_request_status": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("upload_status_read", "status_introspection"),
    },
    "grasshopper_mcp:grasshopper_classify_profile_upload_request": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("upload_request_classification",),
    },
    "grasshopper_mcp:grasshopper_probe_upload_status_endpoints": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("experimental_status_probe", "speculative_endpoint_probe"),
    },
    "grasshopper_mcp:grasshopper_introspect_upload_api": {
        "risk": "read_only",
        "required_authorizations": (),
        "capabilities": ("upload_api_introspection",),
    },
    "grasshopper_mcp:grasshopper_upload_profile_logs": {
        "risk": "external_mutation",
        "required_authorizations": (
            "operator_authorized",
            "mutation_authorized",
            "persistence_authorized",
        ),
        "capabilities": (
            "log_upload_request",
            "external_request",
            "persistent_tracker",
        ),
    },
    "grasshopper_mcp:grasshopper_upload_file": {
        "risk": "external_mutation",
        "required_authorizations": (
            "operator_authorized", "mutation_authorized", "persistence_authorized",
        ),
        "capabilities": ("external_upload", "persistent_external_write"),
    },
    "grasshopper_mcp:grasshopper_batch_upload": {
        "risk": "external_mutation",
        "required_authorizations": (
            "operator_authorized", "mutation_authorized", "persistence_authorized",
        ),
        "capabilities": ("external_upload", "persistent_external_write"),
    },
    "grasshopper_mcp:grasshopper_upload": {
        "risk": "external_mutation",
        "required_authorizations": (
            "operator_authorized", "mutation_authorized", "persistence_authorized",
        ),
        "capabilities": ("external_upload", "persistent_external_write"),
    },
    "grasshopper_mcp:grasshopper_ccshare_upload": {
        "risk": "external_mutation",
        "required_authorizations": (
            "operator_authorized", "mutation_authorized", "persistence_authorized",
        ),
        "capabilities": ("external_upload", "persistent_external_write"),
    },
}

# ---------------------------------------------------------------------------
# D3B0: arbitrary-code-execution capability class.
#
# A tool that runs model-supplied Python or shell is not "read_only" merely
# because it is sandboxed by working directory.  The executed process inherits
# the service user, can write the filesystem, can reach the network, and can
# leave persistent artifacts.  Classification below is derived from the actual
# implementations in app/agent_mode/tools.py, not from tool naming.
# ---------------------------------------------------------------------------

CODE_EXECUTION_RISK = "code_execution"

# D3B2A: methodologies with no standing need for arbitrary code execution.
# In a generic chat a privileged executor tool must not become model-facing
# merely because authorization happens to be granted for something else.
GENERIC_CODE_EXECUTION_METHODOLOGIES: frozenset[str] = frozenset({
    "generic_engineering",
    "no_tool_response",
})

# Exact, implementation-derived requirements.  Keys are raw tool names because
# these executor tools are bound by family rather than as qualified extras.
CODE_EXECUTION_TOOL_CAPABILITIES: dict[str, dict[str, Any]] = {
    # Writes a model-supplied script then runs it with subprocess (300s).
    # Arbitrary code: filesystem write, network, process spawn, persistence.
    "agent_run_python": {
        "risk": CODE_EXECUTION_RISK,
        "required_authorizations": (
            "operator_authorized",
            "heavy_tools_authorized",
            "mutation_authorized",
            "persistence_authorized",
        ),
        "capabilities": (
            "arbitrary_code_execution",
            "filesystem_write",
            "network_egress",
            "process_spawn",
            "persistent_artifacts",
        ),
    },
    # shell=True with a command whitelist that includes bash/git/aws/kubectl.
    # A whitelist of interpreters is still arbitrary execution.
    "agent_run_shell": {
        "risk": CODE_EXECUTION_RISK,
        "required_authorizations": (
            "operator_authorized",
            "heavy_tools_authorized",
            "mutation_authorized",
            "persistence_authorized",
        ),
        "capabilities": (
            "arbitrary_code_execution",
            "shell_execution",
            "filesystem_write",
            "network_egress",
            "process_spawn",
        ),
    },
    # Spawns python -m venv then pip install --upgrade (network install,
    # persistent interpreter tree).  Not model-supplied code, so no
    # arbitrary_code_execution capability, but still privileged.
    "agent_create_venv": {
        "risk": CODE_EXECUTION_RISK,
        "required_authorizations": (
            "operator_authorized",
            "heavy_tools_authorized",
            "mutation_authorized",
            "persistence_authorized",
        ),
        "capabilities": (
            "package_install",
            "filesystem_write",
            "network_egress",
            "process_spawn",
            "persistent_artifacts",
        ),
    },
    # Spawns git, fetches an operator-supplied remote, writes a persistent
    # checkout.  Bounded 60s, so not classified heavy compute.
    "agent_git_clone": {
        "risk": CODE_EXECUTION_RISK,
        "required_authorizations": (
            "operator_authorized",
            "mutation_authorized",
            "persistence_authorized",
        ),
        "capabilities": (
            "filesystem_write",
            "network_egress",
            "process_spawn",
            "persistent_artifacts",
        ),
    },
}

# Fail-closed detection for code-execution tools that are not yet enumerated.
# An unknown tool whose name denotes code or shell execution must require
# privileged authorization rather than defaulting to read_only.
CODE_EXECUTION_NAME_RE = re.compile(
    r"(?:^|_)(?:"
    r"run_python|python_run|run_shell|shell_run|run_command|run_cmd|"
    r"run_code|code_run|run_script|script_run|run_bash|bash_run|"
    r"exec_code|code_exec|execute_code|code_execute|exec_python|exec_shell|"
    r"eval_code|code_eval|eval_python|"
    r"code_interpreter|interpreter_exec|exec_interpreter|"
    r"notebook_exec|exec_notebook|sandbox_exec|exec_sandbox|"
    r"spawn_process|process_spawn|subprocess_run"
    r")(?:$|_)",
    re.IGNORECASE,
)

# Requirements applied to a fail-closed (unenumerated) code-execution match.
CODE_EXECUTION_FALLBACK_REQUIREMENTS: tuple[str, ...] = (
    "operator_authorized",
    "heavy_tools_authorized",
    "mutation_authorized",
    "persistence_authorized",
)


def code_execution_capability(name: str) -> dict[str, Any] | None:
    """Return the code-execution capability record for a tool, or None.

    Accepts either a raw tool name or a qualified ``family:tool`` name.  This
    is the single source of truth consumed by the execution gate, parent
    profile eligibility, and MCOP child narrowing.
    """
    text = str(name or "").strip()
    if not text:
        return None
    raw = text.split(":", 1)[1] if ":" in text else text
    exact = CODE_EXECUTION_TOOL_CAPABILITIES.get(raw)
    if exact is not None:
        return {
            "risk": str(exact["risk"]),
            "required_authorizations": tuple(exact["required_authorizations"]),
            "capabilities": tuple(exact["capabilities"]),
            "classification": "enumerated",
        }
    if CODE_EXECUTION_NAME_RE.search(raw):
        return {
            "risk": CODE_EXECUTION_RISK,
            "required_authorizations": tuple(CODE_EXECUTION_FALLBACK_REQUIREMENTS),
            "capabilities": ("arbitrary_code_execution", "unclassified_code_execution"),
            "classification": "fail_closed_pattern",
        }
    return None


def is_code_execution_tool(name: str) -> bool:
    """Public: whether a tool is in the arbitrary-code-execution class."""
    return code_execution_capability(name) is not None

_INTENT_EXTRAS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(?:build|create|generate|persist)\b.{0,40}\bincident\s+scene\b|\bincident\s+scene\b.{0,40}\b(?:build|create|generate|persist)\b", re.I), ("s3_stb_logs:build_incident_scene",)),
    (re.compile(r"\bcompare\b.{0,25}\bincident\s+scenes?\b|\bincident\s+scenes?\b.{0,25}\bcompare\b", re.I), ("s3_stb_logs:compare_incident_scenes",)),
    (re.compile(r"\bquery\b.{0,25}\bincident\s+scene\b", re.I), ("s3_stb_logs:query_incident_scene",)),
    (re.compile(r"\bexpand\b.{0,30}\b(?:incident\s+)?scene\b|\btargeted\s+evidence\s+expansion\b", re.I), ("s3_stb_logs:expand_scene_region",)),
    (re.compile(r"\brender\b.{0,25}\bincident\s+scene\b|\bscene\.svg\b", re.I), ("s3_stb_logs:render_incident_scene",)),
    (re.compile(r"\bvalidate\b.{0,25}\bincident\s+scene\b", re.I), ("s3_stb_logs:validate_incident_scene",)),
    (re.compile(r"\b(?:build|create|persist)\b.{0,30}\b(?:log\s+)?capsule\b|\b(?:log\s+)?capsule\b.{0,30}\b(?:build|create|persist)\b", re.I), ("s3_stb_logs:build_log_capsule",)),
)

_EXACT_EXTRA_RE = re.compile(r"\b([a-z][a-z0-9_]*):([A-Za-z_][A-Za-z0-9_-]*)\b")


def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in result:
            result.append(value)
    return tuple(result)


def normalize_authorization_flags(flags: Mapping[str, Any] | None = None) -> dict[str, bool]:
    result = dict(AUTHORIZATION_DEFAULTS)
    for key in result:
        if flags and key in flags:
            result[key] = bool(flags[key])
    return result


def canonical_extra_tool(value: str, default_owner: str = "s3_stb_logs") -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if ":" not in text:
        # D3B2A: an operational executor tool belongs to its real owning
        # family.  Defaulting it to the S3 owner would fabricate a tool
        # identity that no family can resolve.
        operational_owner = operational_tool_owner(text)
        if operational_owner:
            return f"{operational_owner}:{text}"
        return f"{default_owner}:{text}"
    owner, name = text.split(":", 1)
    owner = re.sub(r"[^a-z0-9_]+", "_", owner.lower()).strip("_")
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return f"{owner}:{name}" if owner and name else ""


def _static_requested_extra_tools(prompt: str | None) -> tuple[str, ...]:
    """Legacy deterministic intent rules that do not require registry access."""
    text = str(prompt or "")
    requested: list[str] = []
    # Exact qualified requests are extracted only by ToolActivationIntent,
    # which requires a positive activation directive.  Merely mentioning a
    # known tool in examples, logs, negated instructions, or schemas is data.
    for pattern, tools in _INTENT_EXTRAS:
        if pattern.search(text):
            requested.extend(tools)
    workflow = detect_operational_workflow(text)
    if workflow is not None:
        requested.extend(workflow.required_exact_tools)
    return tuple(sorted(set(requested)))


def extract_requested_extra_tools(
    prompt: str | None,
    *,
    registry_index: RegistryToolIndex | None = None,
) -> tuple[str, ...]:
    """Extract static and live-registry exact-tool requests.

    Qualified dynamic names and explicit unique bare names are resolved against
    the current registry. Ambiguous, unknown, or unhealthy requests are not
    silently assigned to ``s3_stb_logs``.
    """
    intent = activation_intent_from_prompt(prompt, registry_index=registry_index)
    return tuple(sorted(set([
        *_static_requested_extra_tools(prompt),
        *intent.requested_exact_tools,
    ])))


_TOOLSET_ACTIVATE_RE = re.compile(
    r"(?:activate|bind|enable|use(?:\s+the)?)\s+"
    r"([a-z][a-z0-9_]{2,63})\b(?!:)"
    r"(?:\s+(?:toolset|tools|family))?",
    re.IGNORECASE,
)


def extract_requested_toolsets(
    prompt: str | None,
    registry_snapshot: frozenset[str] = frozenset(),
    *,
    registry_index: RegistryToolIndex | None = None,
) -> tuple[str, ...]:
    """Compatibility extractor for explicit whole-family requests.

    Dynamic production resolution is performed by ``ToolActivationIntent``.
    This helper preserves the older pure-function API used by policy tests and
    callers that have only a bounded family-name snapshot.
    """
    if registry_index is not None:
        return activation_intent_from_prompt(
            prompt, registry_index=registry_index
        ).requested_toolsets
    text = str(prompt or "")
    known = {str(value).strip().lower() for value in registry_snapshot or ()}
    result: list[str] = []
    for match in _TOOLSET_ACTIVATE_RE.finditer(text):
        name = str(match.group(1) or "").lower().strip()
        if not name or (known and name not in known):
            continue
        if name not in result:
            result.append(name)
    return tuple(result)


def extra_tool_eligibility(extra_tool: str, authorization_flags: Mapping[str, Any] | None = None) -> tuple[bool, tuple[str, ...]]:
    canonical = canonical_extra_tool(extra_tool)
    metadata = EXACT_TOOL_CAPABILITIES.get(canonical)
    if metadata is None:
        code_exec = code_execution_capability(canonical)
        if code_exec is not None:
            required = tuple(code_exec.get("required_authorizations", ()))
        else:
            # Dynamic MCP tools use the same server-side capability classifier as
            # the execution gate (not an optimistic read-only default).
            try:
                from app.agent.tool_execution_gate import required_authorizations_for_tool

                required = tuple(required_authorizations_for_tool(canonical))
            except Exception:
                required = ()
    else:
        required = tuple(metadata.get("required_authorizations", ()))
    flags = normalize_authorization_flags(authorization_flags)
    missing = tuple(key for key in required if not flags.get(key, False))
    return (not missing, missing)


def profile_signature(material: Mapping[str, Any]) -> str:
    """Deterministic tool-profile identity.

    D3B1 removes the JSON default-coercion hook, so a callable, connection
    object, Pydantic model class, or memory address can no longer be silently
    stringified into profile identity.  An unsupported value now raises a safe
    deterministic canonicalization error instead.
    """
    from app.agent import registry_canonical as canonical

    return canonical.content_signature(dict(material))


@dataclass(frozen=True)
class ToolProfile:
    schema: str = TOOL_PROFILE_SCHEMA
    policy_version: str = TOOL_PROFILE_POLICY_VERSION
    methodology: str = "generic_engineering"
    active_toolsets: tuple[str, ...] = ()
    requested_toolsets: tuple[str, ...] = ()
    eligible_toolsets: tuple[str, ...] = ()
    unavailable_toolsets: tuple[str, ...] = ()
    requested_extra_tools: tuple[str, ...] = ()
    eligible_extra_tools: tuple[str, ...] = ()
    unavailable_extra_tools: tuple[str, ...] = ()
    pending_registry_requests: tuple[str, ...] = ()
    unhealthy_requests: tuple[str, ...] = ()
    ambiguous_extra_tools: dict[str, tuple[str, ...]] = field(default_factory=dict)
    pending_authorization_extra_tools: tuple[str, ...] = ()
    missing_authorizations: dict[str, tuple[str, ...]] = field(default_factory=dict)
    curated_tools_by_toolset: dict[str, tuple[str, ...]] = field(default_factory=dict)
    authorization_flags: dict[str, bool] = field(default_factory=lambda: dict(AUTHORIZATION_DEFAULTS))
    operational_workflow: str = ""
    operational_required_tools: tuple[str, ...] = ()
    code_execution_allowlist: tuple[str, ...] | None = None
    inventory_signature: str = ""
    activation_request_revision: int = 0
    continuity_task_scope: str = ""
    continuity_environment: str = ""
    continuity_restored: bool = False
    continuity_revision: int = 0
    mcop_children_forbidden: bool = False
    signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_tool_profile(
    *,
    methodology: str,
    methodology_toolsets: Sequence[str],
    prompt: str | None = None,
    prior_active_toolsets: Sequence[str] = (),
    prior_extra_tools: Sequence[str] = (),
    prior_requested_toolsets: Sequence[str] = (),
    authorization_flags: Mapping[str, Any] | None = None,
    requested_extra_tools: Sequence[str] = (),
    requested_toolsets: Sequence[str] = (),
    inventory_signature: str = "",
    prior_operational_workflow: str = "",
    registry_index: RegistryToolIndex | None = None,
    available_toolsets: Sequence[str] | None = None,
    available_extra_tools: Sequence[str] | None = None,
    prior_activation_request_revision: int = 0,
    activation_request_revision: int = 0,
    continuity_task_scope: str = "",
    continuity_environment: str = "",
    continuity_restored: bool = False,
    continuity_revision: int = 0,
    prior_mcop_children_forbidden: bool = False,
) -> ToolProfile:
    flags = normalize_authorization_flags(authorization_flags)
    mcop_children_forbidden = merge_mcop_children_forbidden(
        prior_mcop_children_forbidden, parse_mcop_constraint_delta(prompt)
    )
    registry = registry_index or RegistryToolIndex.live()
    prompt_intent = activation_intent_from_prompt(prompt, registry_index=registry)

    requested_sets = _dedupe([
        *prior_requested_toolsets,
        *requested_toolsets,
        *prompt_intent.requested_toolsets,
    ])
    requested = _dedupe([
        *prior_extra_tools,
        *_static_requested_extra_tools(prompt),
        *prompt_intent.requested_exact_tools,
        *(canonical_extra_tool(value) for value in requested_extra_tools),
    ])
    prior_requested_sets_normalized = _dedupe(prior_requested_toolsets)
    prior_requested_tools_normalized = _dedupe(
        canonical_extra_tool(value) for value in prior_extra_tools
    )
    prior_revision = max(0, int(prior_activation_request_revision or 0))
    effective_activation_revision = max(
        prior_revision,
        max(0, int(activation_request_revision or 0)),
    )
    if (
        set(requested_sets) != set(prior_requested_sets_normalized)
        or set(requested) != set(prior_requested_tools_normalized)
    ) and effective_activation_revision <= prior_revision:
        effective_activation_revision = prior_revision + 1

    unavailable: list[str] = list(prompt_intent.unavailable_requests)
    pending_registry: list[str] = list(prompt_intent.pending_registry_requests)
    unhealthy: list[str] = list(prompt_intent.unhealthy_requests)
    usable_requested_sets: list[str] = []
    available_toolset_set = (
        None if available_toolsets is None else set(str(value) for value in available_toolsets)
    )
    for family in requested_sets:
        family_usable = (
            family in available_toolset_set
            if available_toolset_set is not None
            else registry.family_usable(family)
        )
        if family_usable:
            usable_requested_sets.append(family)
        elif available_toolset_set is not None or registry.family_known(family):
            unhealthy.append(family)
        else:
            # Persist the request while discovery is incomplete; do not claim
            # availability and do not collapse it into authorization pending.
            pending_registry.append(family)

    owners: list[str] = []
    registry_available_requested: list[str] = []
    available_extra_set = (
        None if available_extra_tools is None else set(str(value) for value in available_extra_tools)
    )
    for value in requested:
        if ":" not in value:
            continue
        owner, raw = value.split(":", 1)
        static_known = value in EXACT_TOOL_CAPABILITIES or bool(operational_tool_owner(raw))
        family_known = registry.family_known(owner)
        tool_known = raw in set(registry.tools(owner)) if family_known else False
        explicitly_available = available_extra_set is not None and value in available_extra_set
        explicitly_unavailable = available_extra_set is not None and value not in available_extra_set
        if explicitly_available:
            owners.append(owner)
            registry_available_requested.append(value)
        elif explicitly_unavailable:
            unavailable.append(value)
        elif family_known and not registry.family_usable(owner):
            # A checkpointed/static tool declaration cannot bypass a current
            # unhealthy-family result. Preserve the request, but withhold both
            # the owner family and exact tool until health recovers.
            unhealthy.append(value)
        elif tool_known and registry.family_usable(owner):
            owners.append(owner)
            registry_available_requested.append(value)
        elif static_known and (
            (not family_known or registry.family_usable(owner))
            and (
                value in EXACT_TOOL_CAPABILITIES
                or owner in set(methodology_toolsets)
                or bool(operational_tool_owner(raw))
            )
        ):
            # Enumerated architectural tools retain their exact capability
            # contract when a healthy stripped registry snapshot omits a row.
            # The unhealthy-family branch above still fails closed.
            owners.append(owner)
            registry_available_requested.append(value)
        elif family_known:
            unavailable.append(value)
        else:
            pending_registry.append(value)

    requested_dynamic_families = set(requested_sets) | {
        value.split(":", 1)[0] for value in requested if ":" in value
    }
    retained_prior_toolsets = [
        name for name in prior_active_toolsets
        if name not in requested_dynamic_families
    ]
    active = _dedupe([
        *STABLE_CORE_TOOLSETS,
        *retained_prior_toolsets,
        *methodology_toolsets,
        *usable_requested_sets,
        *owners,
    ])

    eligible: list[str] = []
    pending: list[str] = []
    missing: dict[str, tuple[str, ...]] = {}
    available_requested_set = set(registry_available_requested)
    for extra in requested:
        if extra not in available_requested_set:
            continue
        ok, absent = extra_tool_eligibility(extra, flags)
        if ok:
            eligible.append(extra)
        else:
            pending.append(extra)
            missing[extra] = absent

    detected = detect_operational_workflow(prompt)
    operational_requested = operational_tools_in(requested)
    workflow = ""
    if detected is not None:
        workflow = detected.workflow
    elif str(prior_operational_workflow or "").strip():
        workflow = str(prior_operational_workflow).strip()
    elif operational_requested:
        workflow = REPO_CHECKOUT_LOCAL_DEPLOY

    code_execution_allowlist: tuple[str, ...] | None = None
    if workflow or methodology in GENERIC_CODE_EXECUTION_METHODOLOGIES:
        operational_set = set(operational_requested)
        code_execution_allowlist = tuple(sorted(
            value.split(":", 1)[1] if ":" in value else value
            for value in eligible
            if value in operational_set
        ))

    # Least privilege: an exact-tool request activates its owner only with an
    # exact curated list.  A whole-family request intentionally omits the
    # curation entry and receives the bounded normal family surface, still
    # filtered by the shared authorization classifier.
    curated: dict[str, tuple[str, ...]] = {}
    explicit_families = set(usable_requested_sets)
    owner_exact: dict[str, list[str]] = {}
    for value in eligible:
        if ":" in value:
            owner, raw = value.split(":", 1)
            owner_exact.setdefault(owner, []).append(raw)
    if "s3_stb_logs" in active and "s3_stb_logs" not in explicit_families:
        curated["s3_stb_logs"] = _dedupe([
            *S3_CURATED_CORE_TOOLS,
            *owner_exact.get("s3_stb_logs", ()),
        ])
    for owner in sorted(set(owners) - explicit_families - {"s3_stb_logs"}):
        curated[owner] = _dedupe(owner_exact.get(owner, ()))

    signature_material = {
        "schema": TOOL_PROFILE_SCHEMA,
        "policy_version": TOOL_PROFILE_POLICY_VERSION,
        "methodology": methodology,
        "active_toolsets": active,
        "requested_toolsets": requested_sets,
        "eligible_toolsets": tuple(sorted(usable_requested_sets)),
        "unavailable_toolsets": tuple(sorted(set([
            *pending_registry,
            *unhealthy,
        ]).intersection(set(requested_sets)))),
        "eligible_extra_tools": tuple(sorted(eligible)),
        "pending_authorization_extra_tools": tuple(sorted(pending)),
        "unavailable_extra_tools": tuple(sorted(set(unavailable))),
        "pending_registry_requests": tuple(sorted(set(pending_registry))),
        "unhealthy_requests": tuple(sorted(set(unhealthy))),
        "ambiguous_extra_tools": {
            key: tuple(value) for key, value in sorted(prompt_intent.ambiguous_requests.items())
        },
        "curated_tools_by_toolset": {key: tuple(value) for key, value in sorted(curated.items())},
        "authorization_flags": flags,
        "operational_workflow": workflow,
        "code_execution_allowlist": code_execution_allowlist or (),
        "inventory_signature": inventory_signature,
        "activation_request_revision": effective_activation_revision,
        "continuity_task_scope": continuity_task_scope,
        "continuity_environment": continuity_environment,
        "continuity_restored": bool(continuity_restored),
        "continuity_revision": int(continuity_revision or 0),
        "mcop_children_forbidden": bool(mcop_children_forbidden),
    }
    return ToolProfile(
        methodology=methodology,
        active_toolsets=active,
        requested_toolsets=tuple(sorted(requested_sets)),
        eligible_toolsets=tuple(sorted(usable_requested_sets)),
        unavailable_toolsets=tuple(sorted(
            set(requested_sets).intersection(set([*pending_registry, *unhealthy]))
        )),
        requested_extra_tools=tuple(sorted(requested)),
        eligible_extra_tools=tuple(sorted(eligible)),
        unavailable_extra_tools=tuple(sorted(set(unavailable))),
        pending_registry_requests=tuple(sorted(set(pending_registry))),
        unhealthy_requests=tuple(sorted(set(unhealthy))),
        ambiguous_extra_tools={
            key: tuple(value) for key, value in sorted(prompt_intent.ambiguous_requests.items())
        },
        pending_authorization_extra_tools=tuple(sorted(pending)),
        missing_authorizations=missing,
        curated_tools_by_toolset=curated,
        authorization_flags=flags,
        operational_workflow=workflow,
        operational_required_tools=tuple(sorted(operational_requested)),
        code_execution_allowlist=code_execution_allowlist,
        inventory_signature=inventory_signature,
        activation_request_revision=effective_activation_revision,
        continuity_task_scope=continuity_task_scope,
        continuity_environment=continuity_environment,
        continuity_restored=bool(continuity_restored),
        continuity_revision=int(continuity_revision or 0),
        mcop_children_forbidden=bool(mcop_children_forbidden),
        signature=profile_signature(signature_material),
    )


def tool_profile_selftest() -> dict[str, Any]:
    base = build_tool_profile(
        methodology="receiver_reboot_dvr_playback",
        methodology_toolsets=("s3_stb_logs", "rtr_alerts_mcp"),
        prompt="Investigate R1911746693 and build an incident scene",
    )
    authorized = build_tool_profile(
        methodology="receiver_reboot_dvr_playback",
        methodology_toolsets=("s3_stb_logs", "rtr_alerts_mcp"),
        prompt="Investigate R1911746693 and build an incident scene",
        prior_active_toolsets=base.active_toolsets,
        prior_extra_tools=base.requested_extra_tools,
        authorization_flags={"heavy_tools_authorized": True},
    )
    checks = {
        "heavy_pending_without_authorization": "s3_stb_logs:build_incident_scene" in base.pending_authorization_extra_tools,
        "heavy_absent_from_curated_without_authorization": "build_incident_scene" not in base.curated_tools_by_toolset.get("s3_stb_logs", ()),
        "promotion_after_authorization": "s3_stb_logs:build_incident_scene" in authorized.eligible_extra_tools,
        "curated_s3_core_bounded": len(base.curated_tools_by_toolset.get("s3_stb_logs", ())) == len(S3_CURATED_CORE_TOOLS),
        "thread_additive": set(base.active_toolsets).issubset(set(authorized.active_toolsets)),
        "deterministic_signature": base.signature == build_tool_profile(
            methodology="receiver_reboot_dvr_playback",
            methodology_toolsets=("s3_stb_logs", "rtr_alerts_mcp"),
            prompt="Investigate R1911746693 and build an incident scene",
        ).signature,
    }
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks}
