from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.agent.methodology import select_methodology
from app.agent.continuity_policy import ContinuityCheckpoint, resolve_continuity
from app.agent.operational_workflows import (
    REPO_CHECKOUT_LOCAL_DEPLOY,
    workflow_from_requested_tools,
)
from app.agent.tool_profiles import ToolProfile, build_tool_profile

logger = logging.getLogger(__name__)

# Keep the legacy broad list in one place so only the model-facing binding path is
# narrowed. ToolNode may still be able to execute already-emitted calls.
BROAD_TOOLSETS: tuple[str, ...] = (
    "search",
    "agent_mode",
    "netra_mcp",
    "beta_report",
    "grasshopper_mcp",
    "viewership",
    "s3_stb_logs",
    "s3_stb_logs_prod",
    "stbhealth_mcp",
    "stbhealth_popups_mcp",
    "rca_mcp",
    "rtr_alerts_mcp",
    "qos_mcp",
    "epg_mcp",
    "net_detective_mcp",
    "qodo_context_mcp",
    "dish_code_tools",
    "dva_mcp",
    "headless_browser_mcp",
    "dish_internal",
    "backend_facades",
)

# Default first-pass toolsets for non-investigation prompts. This deliberately
# excludes every MCP domain tool by default so local Ollama is not forced to pick
# from a catalog it does not need.
GENERIC_INITIAL_TOOLSETS: tuple[str, ...] = ("search", "agent_mode", "dish_internal")

# Methodology name selected when a prompt carries no domain signal.
GENERIC_ENGINEERING = "generic_engineering"

# D3B2A: methodologies whose scope is a subset of local repository/runtime
# work.  While an operational workflow is active these must not displace it,
# which is exactly what produced the observed repo_code_review /
# generic_engineering oscillation mid-deployment.  A genuinely different
# domain (receiver RCA, log acquisition) still transitions intentionally.
OPERATIONAL_WORKFLOW_STABLE_METHODOLOGIES: frozenset[str] = frozenset({
    "generic_engineering",
    "repo_code_review",
    "local_repository_audit",
    "backend_runtime_debug",
    "performance_scalability_review",
    "artifact_generation",
})

INITIAL_TOOLSETS_BY_METHODOLOGY: dict[str, tuple[str, ...]] = {
    # --- STB / receiver investigation ---
    "receiver_reboot_dvr_playback": ("s3_stb_logs", "rtr_alerts_mcp"),
    "popup_signal_loss_investigation": ("s3_stb_logs", "rtr_alerts_mcp"),
    "dvr_playback_instability_investigation": ("s3_stb_logs", "rtr_alerts_mcp"),
    "qos_ota_switchback_investigation": ("qos_mcp", "rtr_alerts_mcp"),
    "viewership_rtr_investigation": ("viewership", "rtr_alerts_mcp"),
    "qos_switchback_investigation": ("qos_mcp", "rtr_alerts_mcp"),
    "epg_schedule_metadata_check": ("epg_mcp",),
    "dva_stb_firmware_workflow": ("dva_mcp", "stbhealth_mcp"),
    # --- Log acquisition ---
    "log_acquisition_preview": ("s3_stb_logs",),
    "log_acquisition_execute": ("s3_stb_logs", "grasshopper_mcp"),
    # --- SSH and remote-host ---
    "no_tool_response": (),
    "ssh_host_key_audit": ("agent_mode",),
    "ssh_remote_service_inspection": ("agent_mode",),
    "ssh_artifact_retrieval": ("agent_mode",),
    "remote_ssh_connectivity_check": ("agent_mode",),
    "remote_host_network_usb_triage": ("agent_mode",),
    # --- Backend / Ollama readiness ---
    "ollama_backend_readiness": ("agent_mode", "dish_internal"),
    # --- Tool introspection ---
    "tool_binding_audit": ("agent_mode",),
    "tool_execution_proof": ("agent_mode",),
    "tool_systematic_validation": ("agent_mode",),
    # --- Efficiency / repo ---
    "agent_efficiency_dashboard_audit": ("agent_mode", "dish_code_tools"),
    "local_repository_audit": ("agent_mode", "dish_code_tools"),
    "repo_code_review": ("agent_mode", "dish_code_tools", "qodo_context_mcp"),
    # D3B2A: repository checkout + isolated local deployment.  Deliberately
    # narrow: the owning family only, so no unrelated family is injected.
    "repo_checkout_local_deploy": ("agent_mode",),
    "backend_runtime_debug": ("agent_mode", "dish_code_tools", "log_assist"),
    "performance_scalability_review": ("agent_mode",),
    "timezone_window_resolution": (),
    # --- Research/generation ---
    "web_internal_research": ("internal_search", "search", "gdrive_mcp", "jira_mcp", "confluence_mcp"),
    "artifact_generation": ("agent_mode",),
    "generic_engineering": GENERIC_INITIAL_TOOLSETS,
}

FOLLOWUP_TOOLSETS_BY_METHODOLOGY: dict[str, tuple[str, ...]] = {
    "receiver_reboot_dvr_playback": ("s3_stb_logs", "rtr_alerts_mcp", "stbhealth_mcp", "rca_mcp"),
    "popup_signal_loss_investigation": ("s3_stb_logs", "rtr_alerts_mcp", "stbhealth_popups_mcp"),
    "qos_ota_switchback_investigation": ("qos_mcp", "rtr_alerts_mcp"),
    "viewership_rtr_investigation": ("viewership", "rtr_alerts_mcp"),
}

DATA_INVESTIGATION_METHODOLOGIES = frozenset(
    {
        "receiver_reboot_dvr_playback",
        "popup_signal_loss_investigation",
        "dvr_playback_instability_investigation",
        "qos_ota_switchback_investigation",
        "viewership_rtr_investigation",
        "qos_switchback_investigation",
        "epg_schedule_metadata_check",
        "dva_stb_firmware_workflow",
        "backend_runtime_debug",
        "log_acquisition_preview",
        "log_acquisition_execute",
        "ollama_backend_readiness",
        "remote_host_network_usb_triage",
        "ssh_remote_service_inspection",
        "ssh_host_key_audit",
        "ssh_artifact_retrieval",
        "remote_ssh_connectivity_check",
        "tool_binding_audit",
        "tool_execution_proof",
        "tool_systematic_validation",
        "local_repository_audit",
        "agent_efficiency_dashboard_audit",
        "repo_checkout_local_deploy",
    }
)

EXECUTION_ACTION_KEYWORDS = (
    "investigate",
    "check",
    "pull logs",
    "pull s3",
    "query",
    "analyze",
    "analyse",
    "determine root cause",
    "root cause",
    "compare",
    "smoke test",
    "validate",
    "identify exactly",
    "classify",
)

PSEUDO_TOOL_RESPONSE_PATTERNS = (
    r"\bfunc\s+\w+\s*\(",
    r"\bdef\s+\w+\s*\(",
    r"\bfunction\s+\w+\s*\(",
    r"type\s+\w+\s+struct\s*\{",
    r"interface\s+\w+\s*\{",
    r"tool_calls?\s*[:=]",
    r"\"tool_calls\"\s*:",
    r"\"name\"\s*:\s*\"[A-Za-z0-9_]+\"\s*,\s*\"args\"",
    r"here(?:'s| is)\s+(?:the\s+)?(?:code|function definition|schema)",
    r"available tools?\s*:",
    r"example schema",
    r"json schema",
    r"pseudo[- ]?code",
)

RECEIVER_RE = re.compile(r"\bR\d{6,}\b", re.I)
SOFTWARE_RE = re.compile(r"\bU\d{3,}\b", re.I)
TIME_PHRASE_RE = re.compile(
    r"\b(?:around\s+)?(?:\d{1,2}:\d{2}\s*(?:am|pm)|\d{1,2}\s*(?:am|pm)|yesterday|last\s+\w+day|July\s+\d{1,2}\s*[-–]\s*\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2})\b",
    re.I,
)


@dataclass(frozen=True)
class ToolChoicePlan:
    methodology: str
    required_toolsets: tuple[str, ...] = ()
    candidate_toolsets: tuple[str, ...] = ()
    candidate_tools: tuple[str, ...] = ()
    first_tool: str = ""
    why_first_tool: str = ""
    required_inputs: dict[str, Any] = field(default_factory=dict)
    forbidden_tools: tuple[str, ...] = ()
    missing_required_toolsets: tuple[str, ...] = ()
    data_investigation: bool = False
    has_prior_tool_results: bool = False
    tool_profile_signature: str = ""
    inventory_signature: str = ""
    requested_toolsets: tuple[str, ...] = ()
    eligible_toolsets: tuple[str, ...] = ()
    unavailable_toolsets: tuple[str, ...] = ()
    requested_extra_tools: tuple[str, ...] = ()
    eligible_extra_tools: tuple[str, ...] = ()
    pending_authorization_extra_tools: tuple[str, ...] = ()
    unavailable_extra_tools: tuple[str, ...] = ()
    pending_registry_requests: tuple[str, ...] = ()
    unhealthy_requests: tuple[str, ...] = ()
    ambiguous_extra_tools: dict[str, tuple[str, ...]] = field(default_factory=dict)
    activation_request_revision: int = 0
    continuity_task_scope: str = ""
    continuity_environment: str = ""
    continuity_restored: bool = False
    continuity_revision: int = 0
    mcop_children_forbidden: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dedupe_keep_order(items: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return tuple(out)


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", tool))


def _tool_names(tools: Iterable[Any]) -> tuple[str, ...]:
    return tuple(_tool_name(t) for t in tools if getattr(t, "name", None))


def _extract_required_inputs(prompt: str) -> dict[str, Any]:
    receivers = _dedupe_keep_order(m.group(0).upper() for m in RECEIVER_RE.finditer(prompt or ""))
    software = _dedupe_keep_order(m.group(0).upper() for m in SOFTWARE_RE.finditer(prompt or ""))
    windows = _dedupe_keep_order(m.group(0) for m in TIME_PHRASE_RE.finditer(prompt or ""))
    data: dict[str, Any] = {}
    if receivers:
        data["receivers"] = list(receivers)
    if software:
        data["software_versions"] = list(software)
    if windows:
        data["time_windows"] = list(windows)
    if "july 1-7, 2026" in (prompt or "").lower() or "july 1–7, 2026" in (prompt or "").lower():
        data["date_window"] = "July 1-7, 2026"
    return data


def _first_tool_hint(methodology: str) -> tuple[str, str, tuple[str, ...]]:
    if methodology == "receiver_reboot_dvr_playback":
        return (
            "s3 log bundle/search/read tool",
            "Start from receiver-scoped STB logs around the reported DVR playback reboot window before correlating RTR alerts.",
            ("s3", "log", "bundle", "search", "read"),
        )
    if methodology == "popup_signal_loss_investigation":
        return (
            "s3 log bundle/search/read tool",
            "Find the exact popup code/string in Joey/Hopper logs before looking up popup definitions.",
            ("s3", "log", "popup", "search", "read"),
        )
    if methodology == "qos_ota_switchback_investigation":
        return (
            "qos_get_coverage",
            "QoS coverage establishes whether the receiver/session window exists before device/session/switchback analysis.",
            ("qos_get_coverage", "coverage"),
        )
    if methodology == "viewership_rtr_investigation":
        return (
            "viewership top-services/watch-hours query",
            "The first evidence must preserve the partner's July 1-7, 2026 watch-hours window before correlating RTR anomalies.",
            ("viewership", "watch", "top", "service"),
        )
    if methodology == "remote_host_network_usb_triage":
        return (
            "agent_run_shell or cluster_inspect or dish_internal_tool",
            "Use SSH-based read-only commands (ip link, dmesg, lsusb, journalctl) to diagnose NIC/USB/bridge state before reporting.",
            ("shell", "ip", "link", "inspect", "dmesg", "lsusb"),
        )
    if methodology == "log_acquisition_preview":
        return (
            "s3 list_dates or list_files tool",
            "Inventory existing S3 log dates for the receiver before attempting any Grasshopper acquisition planning.",
            ("s3", "list", "dates", "files", "inventory"),
        )
    if methodology == "ollama_backend_readiness":
        return (
            "health or readiness endpoint / bound tool",
            "Check the backend health endpoint or use the highest-signal bound readiness tool first.",
            ("health", "readiness", "status", "selftest"),
        )
    if methodology in ("ssh_host_key_audit", "ssh_remote_service_inspection", "ssh_artifact_retrieval",
                       "remote_ssh_connectivity_check"):
        return (
            "agent_run_shell (SSH)",
            "Use a read-only SSH command via agent_run_shell to collect the required remote evidence.",
            ("shell", "ssh", "remote"),
        )
    if methodology == "repo_checkout_local_deploy":
        return (
            "agent_git_clone",
            "Clone the requested repository with the configured SSH identity, then inspect it before executing anything.",
            ("git_clone", "clone", "shell"),
        )
    return ("highest-signal bound tool", "Use the most specialized bound tool for the task before generic explanation.", ())


def _choose_first_actual_tool(methodology: str, tools: Sequence[Any]) -> str:
    hint, _why, patterns = _first_tool_hint(methodology)
    names = list(_tool_names(tools))
    if not names:
        return hint
    lowered = [(name, name.lower()) for name in names]
    for pattern in patterns:
        p = pattern.lower()
        for original, low in lowered:
            if p in low:
                return original
    return names[0]


def is_data_investigation_prompt(prompt: str | None, methodology: str | None = None) -> bool:
    text = (prompt or "").lower()
    selected = methodology or select_methodology(text)["name"]
    if selected in DATA_INVESTIGATION_METHODOLOGIES:
        return True
    return any(k in text for k in EXECUTION_ACTION_KEYWORDS)


def contains_pseudo_tool_response(text: str | None) -> bool:
    body = text or ""
    if not body.strip():
        return False
    return any(re.search(pattern, body, flags=re.I | re.S) for pattern in PSEUDO_TOOL_RESPONSE_PATTERNS)


def selected_toolsets_for_prompt(prompt: str | None, *, has_prior_tool_results: bool = False) -> tuple[str, ...]:
    methodology = select_methodology(prompt)["name"]
    if has_prior_tool_results and methodology in FOLLOWUP_TOOLSETS_BY_METHODOLOGY:
        return FOLLOWUP_TOOLSETS_BY_METHODOLOGY[methodology]
    return INITIAL_TOOLSETS_BY_METHODOLOGY.get(methodology, GENERIC_INITIAL_TOOLSETS)


def get_tools_for_toolsets(
    toolsets: Iterable[str],
    *,
    curated_tools_by_toolset: dict[str, tuple[str, ...]] | None = None,
    model_facing: bool = False,
    authorization_flags: dict[str, Any] | None = None,
    code_execution_allowlist: Sequence[str] | None = None,
    forbidden_tool_names: Sequence[str] = (),
) -> list[Any]:
    from app.agent.agents.tools import get_tools_set, get_tools_set_filtered
    from app.agent.tool_execution_gate import (
        prepare_model_facing_tool,
        tool_is_permitted_by_flags,
    )
    from app.agent.tool_profiles import extra_tool_eligibility, is_code_execution_tool

    tools: list[Any] = []
    seen: set[str] = set()
    forbidden = {str(name) for name in forbidden_tool_names if str(name)}
    curated_tools_by_toolset = curated_tools_by_toolset or {}
    for toolset in toolsets:
        try:
            if toolset in curated_tools_by_toolset:
                candidates = get_tools_set_filtered(toolset, curated_tools_by_toolset[toolset])
            else:
                candidates = get_tools_set(toolset)
            for tool in candidates:
                candidate = prepare_model_facing_tool(tool) if model_facing else tool
                if candidate is None:
                    logger.error("Omitting unsafe model-facing tool schema for toolset=%s name=%s", toolset, _tool_name(tool))
                    continue
                name = _tool_name(candidate)
                if name in forbidden:
                    logger.info(
                        "Withholding tool due to current task execution constraint: "
                        "toolset=%s name=%s", toolset, name
                    )
                    continue
                # D3B0: a family may contain arbitrary-code-execution tools.
                # Family membership alone must not make them model-facing; the
                # shared capability classifier decides, and unknown
                # code-execution names fail closed.
                # D3B2A: when an operational workflow is active, the
                # code-execution class is additionally narrowed to the exact
                # tools that workflow actually requires.
                if (
                    model_facing
                    and name
                    and code_execution_allowlist is not None
                    and is_code_execution_tool(name)
                    and name not in set(code_execution_allowlist)
                ):
                    logger.info(
                        "Withholding code-execution tool outside the active operational workflow: toolset=%s name=%s",
                        toolset,
                        name,
                    )
                    continue
                qualified_name = f"{toolset}:{name}" if name else ""
                if model_facing and qualified_name:
                    eligible_for_binding, missing_for_binding = extra_tool_eligibility(
                        qualified_name, authorization_flags or {}
                    )
                    if missing_for_binding and not eligible_for_binding:
                        logger.info(
                            "Withholding authorization-gated tool from model-facing binding: "
                            "toolset=%s name=%s missing=%s",
                            toolset,
                            name,
                            list(missing_for_binding),
                        )
                        continue
                if (
                    model_facing
                    and name
                    and is_code_execution_tool(name)
                    and not tool_is_permitted_by_flags(name, authorization_flags or {})
                ):
                    logger.info(
                        "Withholding code-execution tool from model-facing binding: "
                        "toolset=%s name=%s (authorization not granted)",
                        toolset,
                        name,
                    )
                    continue
                if name and name not in seen:
                    tools.append(candidate)
                    seen.add(name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Toolset %s unavailable while building scoped binding: %s", toolset, exc)
    return tools


def get_all_executor_tools() -> list[Any]:
    return get_tools_for_toolsets(BROAD_TOOLSETS)


def build_profile_for_prompt(
    prompt: str | None,
    *,
    has_prior_tool_results: bool = False,
    prior_active_toolsets: Sequence[str] = (),
    prior_extra_tools: Sequence[str] = (),
    prior_requested_toolsets: Sequence[str] = (),
    authorization_flags: dict[str, Any] | None = None,
    requested_extra_tools: Sequence[str] = (),
    requested_toolsets: Sequence[str] = (),
    prior_activation_request_revision: int = 0,
    activation_request_revision: int = 0,
    registry_index: Any | None = None,
    prior_continuity_methodology: str = "",
    prior_continuity_task_scope: str = "",
    prior_continuity_environment: str = "",
    prior_continuity_revision: int = 0,
    current_environment: str = "",
    prior_mcop_children_forbidden: bool = False,
) -> ToolProfile:
    methodology = select_methodology(prompt)["name"]
    methodology_toolsets = selected_toolsets_for_prompt(prompt, has_prior_tool_results=has_prior_tool_results)
    continuity = resolve_continuity(
        current_prompt=prompt,
        selected_methodology=methodology,
        checkpoint=ContinuityCheckpoint(
            task_scope=prior_continuity_task_scope,
            methodology=prior_continuity_methodology,
            authoritative_environment=prior_continuity_environment,
            revision=prior_continuity_revision,
        ),
        current_environment=current_environment,
    )
    methodology = continuity.methodology
    if continuity.restored_from_checkpoint and methodology in INITIAL_TOOLSETS_BY_METHODOLOGY:
        methodology_toolsets = INITIAL_TOOLSETS_BY_METHODOLOGY[methodology]
    # D3B2A: an active operational workflow must survive a generic
    # follow-up.  "Continue with the deployment." selects
    # generic_engineering, whose default toolsets would inject an unrelated
    # family and destabilise the workflow methodology mid-task.
    prior_operational_workflow = workflow_from_requested_tools(
        [*prior_extra_tools, *requested_extra_tools]
    )
    if (
        prior_operational_workflow
        and methodology in OPERATIONAL_WORKFLOW_STABLE_METHODOLOGIES
        and not continuity.stale_checkpoint_rejected
    ):
        # Preserve an established local repository/deployment workflow across
        # generic or adjacent engineering follow-ups, even for legacy callers
        # that have not yet checkpointed the explicit continuity fields.  A
        # genuinely different domain methodology is not in the stable set and
        # therefore cannot be overwritten by this compatibility bridge.
        methodology = REPO_CHECKOUT_LOCAL_DEPLOY
        methodology_toolsets = INITIAL_TOOLSETS_BY_METHODOLOGY[REPO_CHECKOUT_LOCAL_DEPLOY]
    elif (
        methodology == GENERIC_ENGINEERING
        and prior_active_toolsets
    ):
        # D3B3: generalise the D3B2A workflow-stability rule to every active
        # workflow, not just operational ones.  GENERIC_INITIAL_TOOLSETS exists
        # to give a *first-pass* non-investigation prompt a small starting set.
        # A content-free continuation ("Continue.", "Proceed.", "Check the
        # result.") of an already-established investigation carries no domain
        # signal, so adding the generic defaults on top only injects an
        # unrelated family - observed as dish_internal appearing mid
        # receiver-log investigation.  The established profile is preserved by
        # prior_active_toolsets; contribute nothing further.
        methodology_toolsets = ()
    try:
        from app.agent.agents.tools.registry import get_tool_inventory_signature
        inventory_signature = get_tool_inventory_signature()
    except Exception:
        inventory_signature = ""
    return build_tool_profile(
        methodology=methodology,
        methodology_toolsets=methodology_toolsets,
        prompt=prompt,
        prior_active_toolsets=prior_active_toolsets,
        prior_extra_tools=prior_extra_tools,
        prior_requested_toolsets=prior_requested_toolsets,
        authorization_flags=authorization_flags,
        requested_extra_tools=requested_extra_tools,
        requested_toolsets=requested_toolsets,
        inventory_signature=inventory_signature,
        registry_index=registry_index,
        prior_activation_request_revision=prior_activation_request_revision,
        activation_request_revision=activation_request_revision,
        prior_operational_workflow=prior_operational_workflow,
        continuity_task_scope=continuity.task_scope,
        continuity_environment=continuity.authoritative_environment,
        continuity_restored=continuity.restored_from_checkpoint,
        continuity_revision=continuity.next_revision,
        prior_mcop_children_forbidden=prior_mcop_children_forbidden,
    )


def get_scoped_tools_for_prompt(
    prompt: str | None,
    *,
    has_prior_tool_results: bool = False,
    prior_active_toolsets: Sequence[str] = (),
    prior_extra_tools: Sequence[str] = (),
    prior_requested_toolsets: Sequence[str] = (),
    authorization_flags: dict[str, Any] | None = None,
    requested_extra_tools: Sequence[str] = (),
    requested_toolsets: Sequence[str] = (),
    prior_activation_request_revision: int = 0,
    activation_request_revision: int = 0,
    registry_index: Any | None = None,
    prior_continuity_methodology: str = "",
    prior_continuity_task_scope: str = "",
    prior_continuity_environment: str = "",
    prior_continuity_revision: int = 0,
    current_environment: str = "",
    prior_mcop_children_forbidden: bool = False,
) -> tuple[list[Any], ToolChoicePlan]:
    profile = build_profile_for_prompt(
        prompt,
        has_prior_tool_results=has_prior_tool_results,
        prior_active_toolsets=prior_active_toolsets,
        prior_extra_tools=prior_extra_tools,
        prior_requested_toolsets=prior_requested_toolsets,
        authorization_flags=authorization_flags,
        requested_extra_tools=requested_extra_tools,
        requested_toolsets=requested_toolsets,
        prior_activation_request_revision=prior_activation_request_revision,
        activation_request_revision=activation_request_revision,
        registry_index=registry_index,
        prior_continuity_methodology=prior_continuity_methodology,
        prior_continuity_task_scope=prior_continuity_task_scope,
        prior_continuity_environment=prior_continuity_environment,
        prior_continuity_revision=prior_continuity_revision,
        current_environment=current_environment,
        prior_mcop_children_forbidden=prior_mcop_children_forbidden,
    )
    tools = get_tools_for_toolsets(
        profile.active_toolsets,
        curated_tools_by_toolset=profile.curated_tools_by_toolset,
        model_facing=True,
        authorization_flags=dict(profile.authorization_flags or {}),
        code_execution_allowlist=profile.code_execution_allowlist,
        forbidden_tool_names=(
            ("agent_spawn_task", "agent_spawn_parallel")
            if profile.mcop_children_forbidden else ()
        ),
    )
    plan = build_tool_choice_plan(
        prompt,
        candidate_toolsets=profile.active_toolsets,
        candidate_tools=_tool_names(tools),
        has_prior_tool_results=has_prior_tool_results,
        profile=profile,
    )
    return tools, plan


def build_tool_choice_plan(
    prompt: str | None,
    *,
    candidate_toolsets: Iterable[str] | None = None,
    candidate_tools: Iterable[str] | None = None,
    has_prior_tool_results: bool = False,
    profile: ToolProfile | None = None,
) -> ToolChoicePlan:
    selected = select_methodology(prompt, available_tool_families=candidate_toolsets)
    methodology = selected["name"]
    # D3B3: the profile is authoritative for methodology identity.  When
    # build_profile_for_prompt deliberately holds a sticky operational workflow
    # (D3B2A) that a content-free follow-up such as "Continue." would otherwise
    # discard, the plan must report the same methodology the binding was
    # actually built from.  Reporting select_methodology()'s raw answer made the
    # policy telemetry and management facade claim generic_engineering while the
    # binding was still repo_checkout_local_deploy.
    if profile is not None and getattr(profile, "operational_workflow", "") and profile.methodology:
        methodology = profile.methodology
    hint, why, _patterns = _first_tool_hint(methodology)
    names = tuple(candidate_tools or ())
    # Use actual bound name if present; keep the methodology-mandated hint when it
    # names a missing-but-expected tool such as qos_get_coverage.
    first_tool = hint
    if names:
        class _T:
            def __init__(self, name: str):
                self.name = name
        first_tool = _choose_first_actual_tool(methodology, [_T(n) for n in names])
        if methodology == "qos_ota_switchback_investigation" and "qos_get_coverage" not in names:
            first_tool = "qos_get_coverage"
    return ToolChoicePlan(
        methodology=methodology,
        required_toolsets=tuple(selected.get("required_tool_families") or ()),
        candidate_toolsets=tuple(candidate_toolsets or ()),
        candidate_tools=names[:20],
        first_tool=first_tool,
        why_first_tool=why,
        required_inputs=_extract_required_inputs(prompt or ""),
        forbidden_tools=tuple(selected.get("forbidden_broad_tools_unless_justified") or ()),
        missing_required_toolsets=tuple(selected.get("missing_required_tool_families") or ()),
        data_investigation=is_data_investigation_prompt(prompt, methodology),
        has_prior_tool_results=has_prior_tool_results,
        tool_profile_signature=profile.signature if profile else "",
        inventory_signature=profile.inventory_signature if profile else "",
        requested_toolsets=profile.requested_toolsets if profile else (),
        eligible_toolsets=profile.eligible_toolsets if profile else (),
        unavailable_toolsets=profile.unavailable_toolsets if profile else (),
        requested_extra_tools=profile.requested_extra_tools if profile else (),
        eligible_extra_tools=profile.eligible_extra_tools if profile else (),
        pending_authorization_extra_tools=profile.pending_authorization_extra_tools if profile else (),
        unavailable_extra_tools=profile.unavailable_extra_tools if profile else (),
        pending_registry_requests=profile.pending_registry_requests if profile else (),
        unhealthy_requests=profile.unhealthy_requests if profile else (),
        ambiguous_extra_tools=dict(profile.ambiguous_extra_tools) if profile else {},
        activation_request_revision=(
            int(profile.activation_request_revision or 0) if profile else 0
        ),
        continuity_task_scope=profile.continuity_task_scope if profile else "",
        continuity_environment=profile.continuity_environment if profile else "",
        continuity_restored=bool(profile.continuity_restored) if profile else False,
        continuity_revision=int(profile.continuity_revision or 0) if profile else 0,
        mcop_children_forbidden=bool(profile.mcop_children_forbidden) if profile else False,
    )


def rank_tools_for_retry(methodology: str, tools: Sequence[Any], limit: int = 3) -> list[Any]:
    if len(tools) <= limit:
        return list(tools)
    _hint, _why, patterns = _first_tool_hint(methodology)
    scored: list[tuple[int, int, Any]] = []
    for idx, tool in enumerate(tools):
        low = _tool_name(tool).lower()
        score = 0
        for rank, pattern in enumerate(patterns):
            if pattern.lower() in low:
                score += 100 - rank
        if methodology == "qos_ota_switchback_investigation":
            for rank, pattern in enumerate(("qos_get_coverage", "qos_lookup_devices", "qos_get_sessions", "qos_detect_switchback_sequences", "rtr", "alert")):
                if pattern in low:
                    score += 200 - rank
        elif methodology == "popup_signal_loss_investigation":
            for rank, pattern in enumerate(("s3", "log", "popup", "rtr", "alert")):
                if pattern in low:
                    score += 100 - rank
        elif methodology == "receiver_reboot_dvr_playback":
            for rank, pattern in enumerate(("s3", "log", "rtr", "alert", "rca")):
                if pattern in low:
                    score += 100 - rank
        elif methodology == "viewership_rtr_investigation":
            for rank, pattern in enumerate(("viewership", "watch", "hour", "trend", "rtr", "alert")):
                if pattern in low:
                    score += 100 - rank
        scored.append((-score, idx, tool))
    return [tool for _score, _idx, tool in sorted(scored)[:limit]]


def build_compact_tool_execution_system_prompt(plan: ToolChoicePlan, *, retry: bool = False) -> str:
    prefix = "STRICT RETRY: " if retry else ""
    return (
        f"{prefix}Use bound tools when data is required. Do not describe tool schemas, "
        "do not provide pseudo-code, and do not output JSON tool-call suggestions. "
        f"Methodology={plan.methodology}. Selected tool families={list(plan.candidate_toolsets)}. "
        f"First tool target={plan.first_tool}. Required inputs={json.dumps(plan.required_inputs, sort_keys=True)}. "
        "If no suitable bound tool is available, say that explicitly and do not infer a root cause."
    )


def binding_audit(
    *,
    role: str,
    model_name: str,
    tools: Sequence[Any],
    plan: ToolChoicePlan,
    bound_model: Any | None = None,
    response: Any | None = None,
) -> dict[str, Any]:
    names = [_tool_name(tool) for tool in tools]
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    per_tool = []
    for tool in tools[:20]:
        name = _tool_name(tool)
        per_tool.append(
            {
                "name": name,
                "has_args_schema": getattr(tool, "args_schema", None) is not None,
                "sanitized_name": bool(re.match(r"^[A-Za-z_][A-Za-z0-9_\-]*$", name)),
            }
        )
    return {
        "selected_methodology": plan.methodology,
        "selected_toolsets": list(plan.candidate_toolsets),
        "required_toolsets": list(plan.required_toolsets),
        "bound_tool_count": len(tools),
        "first_20_bound_tool_names": names[:20],
        "first_20_tool_schema_status": per_tool,
        "duplicate_tool_names": duplicates,
        "model_role": role,
        "model_name": model_name,
        "tool_capable_flag": role == "tool_worker",
        "tool_profile_signature": plan.tool_profile_signature,
        "inventory_signature": plan.inventory_signature,
        "eligible_extra_tools": list(plan.eligible_extra_tools),
        "pending_authorization_extra_tools": list(plan.pending_authorization_extra_tools),
        "bind_tools_returned_bound_model": bound_model is not None,
        "raw_response_contains_tool_calls": bool(getattr(response, "tool_calls", None)) if response is not None else None,
        "plan": plan.to_dict(),
    }


def count_prior_tool_calls(messages: Iterable[Any], relevant_names: Iterable[str] | None = None) -> int:
    relevant = set(relevant_names or [])
    count = 0
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            if not relevant or name in relevant:
                count += 1
    return count


def append_tool_execution_failure(
    *,
    session_id: str | None,
    prompt: str,
    plan: ToolChoicePlan,
    response_text: str,
    reason: str,
    audit: dict[str, Any] | None = None,
) -> Path:
    root = Path(os.environ.get("AGENT_MODE_WORKDIR", "/tmp/home_agent"))
    sid = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id or "unknown_session")
    run_dir = root / sid / "_tool_execution"
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = run_dir / "EVIDENCE_LEDGER.jsonl"
    row = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entry_type": "FAILED_TOOL_EXECUTION",
        "reason": reason,
        "methodology": plan.methodology,
        "selected_toolsets": list(plan.candidate_toolsets),
        "bound_tool_count": len(plan.candidate_tools),
        "first_tool": plan.first_tool,
        "prompt_excerpt": (prompt or "")[:500],
        "response_excerpt": (response_text or "")[:500],
        "audit": audit or {},
    }
    with ledger.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return ledger


def should_force_tool_retry(*, prompt: str, plan: ToolChoicePlan, tools: Sequence[Any], messages: Sequence[Any], response: Any) -> bool:
    if not tools or not plan.data_investigation:
        return False
    if getattr(response, "tool_calls", None):
        return False
    prior_calls = count_prior_tool_calls(messages, _tool_names(tools))
    if prior_calls > 0:
        return False
    return True


def build_tool_execution_failed_message(plan: ToolChoicePlan, ledger_path: Path | None = None) -> str:
    ledger_note = f" Evidence ledger: {ledger_path}." if ledger_path else ""
    return (
        "Tool execution failed; no root cause can be confirmed.\n\n"
        f"Methodology: {plan.methodology}. Required toolsets: {list(plan.required_toolsets)}. "
        f"Attempted first-tool target: {plan.first_tool}.{ledger_note}\n\n"
        "Next fix: verify LangChain/Ollama structured tool-call binding with the minimal echo_probe test, "
        "then rerun this investigation with the same preserved date/time window."
    )


def tool_execution_policy_selftest() -> dict[str, Any]:
    prompts = {
        "receiver_reboot_dvr_playback": "Receiver R1911746693 reboots during DVR playback around 9pm. Software U820. Check RTR alerts and pull S3 logs.",
        "popup_signal_loss_investigation": "Joey R2200001234 signal lost popup during live TV around 2:30pm yesterday; Hopper R1100005678. Pull S3 logs.",
        "qos_ota_switchback_investigation": "Investigate QoS OTA switchback throughput stall followed by ABR session switch around 11:45pm last Tuesday.",
        "viewership_rtr_investigation": "Content partner lost viewership between July 1-7, 2026; query top services by watch hours and RTR anomalies.",
    }
    observed = {name: build_tool_choice_plan(prompt).methodology for name, prompt in prompts.items()}
    scoped = {name: list(selected_toolsets_for_prompt(prompt)) for name, prompt in prompts.items()}
    expected_scoped = {
        "receiver_reboot_dvr_playback": ["s3_stb_logs", "rtr_alerts_mcp"],
        "popup_signal_loss_investigation": ["s3_stb_logs", "rtr_alerts_mcp"],
        "qos_ota_switchback_investigation": ["qos_mcp", "rtr_alerts_mcp"],
        "viewership_rtr_investigation": ["viewership", "rtr_alerts_mcp"],
    }
    status = "pass" if observed == {k: k for k in prompts} and scoped == expected_scoped else "fail"
    return {"status": status, "observed_methodologies": observed, "scoped_toolsets": scoped}
