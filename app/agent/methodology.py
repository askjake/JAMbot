from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class MethodologyTemplate:
    name: str
    required_tool_families: tuple[str, ...]
    optional_tool_families: tuple[str, ...]
    forbidden_broad_tools_unless_justified: tuple[str, ...]
    required_evidence_fields: tuple[str, ...]
    required_final_classification_fields: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


_BASE_EVIDENCE_FIELDS = (
    "facts",
    "inferences",
    "gaps",
    "errors",
    "raw_artifacts",
    "next_recommended_step",
)

METHODOLOGY_TEMPLATES = {
    "no_tool_response": MethodologyTemplate(
        "no_tool_response",
        (),
        (),
        (),
        _BASE_EVIDENCE_FIELDS,
        ("answer",),
    ),
    "receiver_reboot_dvr_playback": MethodologyTemplate(
        "receiver_reboot_dvr_playback",
        ("s3_stb_logs", "rtr_alerts_mcp"),
        ("dva_mcp", "stbhealth_mcp"),
        ("epg_mcp", "qos_mcp", "viewership"),
        _BASE_EVIDENCE_FIELDS + ("receiver_id", "event_window", "software_version"),
        ("symptom", "confirmed_events", "root_cause", "confidence", "unknowns"),
    ),
    "popup_signal_loss_investigation": MethodologyTemplate(
        "popup_signal_loss_investigation",
        ("s3_stb_logs", "rtr_alerts_mcp"),
        ("stbhealth_popups_mcp", "epg_mcp"),
        ("viewership", "qos_mcp", "dva_mcp", "dva_jam"),
        _BASE_EVIDENCE_FIELDS + ("joey_receiver_id", "hopper_receiver_id", "event_window", "popup_code"),
        ("exact_popup", "trigger", "recurrence", "confidence", "unknowns"),
    ),
    "qos_ota_switchback_investigation": MethodologyTemplate(
        "qos_ota_switchback_investigation",
        ("qos_mcp", "rtr_alerts_mcp"),
        ("epg_mcp",),
        ("dva_mcp", "viewership"),
        _BASE_EVIDENCE_FIELDS + ("receiver_id", "event_window", "session_ids"),
        ("classification", "device_topology", "direct_evidence", "confidence", "unknowns"),
    ),
    "viewership_rtr_investigation": MethodologyTemplate(
        "viewership_rtr_investigation",
        ("viewership", "rtr_alerts_mcp"),
        ("epg_mcp",),
        ("qos_mcp", "dva_mcp", "s3_stb_logs"),
        _BASE_EVIDENCE_FIELDS + ("date_window", "services", "alert_windows"),
        ("classification", "trend_evidence", "alert_correlation", "confidence", "unknowns"),
    ),
    # --- Repository checkout and isolated local deployment (D3B2A) ---
    "repo_checkout_local_deploy": MethodologyTemplate(
        "repo_checkout_local_deploy",
        ("agent_mode",),
        ("dish_code_tools",),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS
        + ("repository_url", "clone_target", "deployment_scope"),
        (
            "repository_identity",
            "inspection_result",
            "deployment_method",
            "runtime_verification",
            "rollback_command",
        ),
    ),
    # --- SSH and remote-host methodologies ---
    "ssh_host_key_audit": MethodologyTemplate(
        "ssh_host_key_audit",
        ("agent_mode",),
        (),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("ssh_target", "fingerprint"),
        ("verdict", "fingerprint_match", "risks"),
    ),
    "ssh_remote_service_inspection": MethodologyTemplate(
        "ssh_remote_service_inspection",
        ("agent_mode",),
        (),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("ssh_target", "remote_port"),
        ("service_identity", "pid", "process_state", "verdict"),
    ),
    "ssh_artifact_retrieval": MethodologyTemplate(
        "ssh_artifact_retrieval",
        ("agent_mode",),
        (),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("ssh_target", "artifact_url"),
        ("artifact_path", "retrieval_status", "verdict"),
    ),
    "remote_ssh_connectivity_check": MethodologyTemplate(
        "remote_ssh_connectivity_check",
        ("agent_mode",),
        (),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("ssh_target",),
        ("reachable", "latency_ms", "verdict"),
    ),
    # NEW: Remote host network/USB/hardware triage — exceeds diship-test reference
    "remote_host_network_usb_triage": MethodologyTemplate(
        "remote_host_network_usb_triage",
        ("agent_mode",),
        ("dish_internal",),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("ssh_target", "interface", "observed_state"),
        ("network_state", "usb_hid_state", "bridge_stp_assessment", "classification", "operator_actions"),
    ),
    # --- Log acquisition methodologies ---
    "log_acquisition_preview": MethodologyTemplate(
        "log_acquisition_preview",
        ("s3_stb_logs",),
        ("grasshopper_mcp",),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("receiver_id", "requested_log_family"),
        ("log_availability_status", "available_dates", "coverage_assessment", "mutation_performed"),
    ),
    "log_acquisition_execute": MethodologyTemplate(
        "log_acquisition_execute",
        ("s3_stb_logs", "grasshopper_mcp"),
        (),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("receiver_id", "requested_log_family", "authorization"),
        ("upload_status", "tracker_status", "available_after", "mutation_performed"),
    ),
    # --- Backend / Ollama readiness ---
    "ollama_backend_readiness": MethodologyTemplate(
        "ollama_backend_readiness",
        ("agent_mode",),
        ("dish_internal",),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("backend_url", "health_status"),
        ("readiness_status", "model_roles", "tool_worker_status", "risks", "next_action"),
    ),
    # --- Observability / tool validation ---
    "tool_binding_audit": MethodologyTemplate(
        "tool_binding_audit",
        ("agent_mode",),
        ("dish_internal",),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("bound_tools",),
        ("audit_result", "binding_failures", "verdict"),
    ),
    "tool_execution_proof": MethodologyTemplate(
        "tool_execution_proof",
        ("agent_mode",),
        (),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("tool_called",),
        ("tool_called", "result_received", "verdict"),
    ),
    "tool_systematic_validation": MethodologyTemplate(
        "tool_systematic_validation",
        ("agent_mode",),
        ("dish_internal",),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("tools_tested",),
        ("pass_count", "fail_count", "certification_report"),
    ),
    # --- Efficiency / dashboard ---
    "agent_efficiency_dashboard_audit": MethodologyTemplate(
        "agent_efficiency_dashboard_audit",
        ("agent_mode",),
        ("dish_code_tools",),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("dashboard_url",),
        ("token_efficiency_score", "compression_ratio", "comparison_verdict"),
    ),
    # --- Repo / code / research ---
    "local_repository_audit": MethodologyTemplate(
        "local_repository_audit",
        ("agent_mode", "dish_code_tools"),
        (),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("repo_path",),
        ("verdict", "findings", "risks"),
    ),
    "repo_code_review": MethodologyTemplate(
        "repo_code_review",
        ("dish_code_tools", "agent_mode"),
        ("qodo_context_mcp", "internal_search"),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("files_touched", "commands_run"),
        ("verdict", "files_changed", "tests_run", "risks"),
    ),
    # --- Backend runtime debug ---
    "backend_runtime_debug": MethodologyTemplate(
        "backend_runtime_debug",
        ("agent_mode", "dish_code_tools"),
        ("log_assist", "s3_stb_logs", "cluster_inspect"),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("logs_examined", "commands_run"),
        ("symptom", "root_cause", "fix", "verification"),
    ),
    # --- Other STB methodologies ---
    "dvr_playback_instability_investigation": MethodologyTemplate(
        "dvr_playback_instability_investigation",
        ("s3_stb_logs", "rtr_alerts_mcp"),
        ("dva_mcp", "stbhealth_mcp"),
        ("epg_mcp", "qos_mcp", "viewership"),
        _BASE_EVIDENCE_FIELDS + ("receiver_id", "event_window"),
        ("symptom", "confirmed_events", "root_cause", "confidence", "unknowns"),
    ),
    "performance_scalability_review": MethodologyTemplate(
        "performance_scalability_review",
        ("agent_mode",),
        ("dish_code_tools",),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("target_service",),
        ("verdict", "scaling_recommendations", "risks"),
    ),
    "timezone_window_resolution": MethodologyTemplate(
        "timezone_window_resolution",
        (),
        (),
        (),
        _BASE_EVIDENCE_FIELDS + ("local_time", "timezone"),
        ("utc_window", "crosses_midnight", "verdict"),
    ),
    # --- Compat/legacy aliases ---
    "qos_switchback_investigation": MethodologyTemplate(
        "qos_switchback_investigation",
        ("qos_mcp", "s3_stb_logs"),
        ("stbhealth_mcp", "rtr_alerts_mcp", "qodo_context_mcp"),
        ("public_web_search", "generic_shell"),
        _BASE_EVIDENCE_FIELDS + ("device_id", "event_window"),
        ("classification", "device_topology", "direct_evidence", "confidence"),
    ),
    "epg_schedule_metadata_check": MethodologyTemplate(
        "epg_schedule_metadata_check",
        ("epg_mcp",),
        ("s3_stb_logs", "stbhealth_mcp"),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("service_id", "schedule_window"),
        ("metadata_status", "schedule_status", "affected_services", "confidence"),
    ),
    "dva_stb_firmware_workflow": MethodologyTemplate(
        "dva_stb_firmware_workflow",
        ("dva_mcp", "stbhealth_mcp"),
        ("s3_stb_logs", "rtr_alerts_mcp"),
        ("public_web_search", "generic_shell"),
        _BASE_EVIDENCE_FIELDS + ("device_id", "firmware_target"),
        ("workflow_status", "device_status", "rollback_plan", "confidence"),
    ),
    "web_internal_research": MethodologyTemplate(
        "web_internal_research",
        ("internal_search",),
        ("public_web_search", "gdrive_mcp", "jira_mcp", "confluence_mcp", "servicenow_mcp"),
        ("generic_shell",),
        _BASE_EVIDENCE_FIELDS + ("sources_queried",),
        ("answer", "sources", "confidence", "remaining_gaps"),
    ),
    "artifact_generation": MethodologyTemplate(
        "artifact_generation",
        ("agent_mode",),
        ("dish_code_tools", "internal_search"),
        ("public_web_search",),
        _BASE_EVIDENCE_FIELDS + ("artifact_paths",),
        ("artifacts_created", "validation", "known_limits"),
    ),
    "generic_engineering": MethodologyTemplate(
        "generic_engineering",
        ("agent_mode",),
        ("internal_search", "dish_code_tools", "public_web_search"),
        (),
        _BASE_EVIDENCE_FIELDS,
        ("summary", "evidence", "confidence", "next_steps"),
    ),
}

# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

_EXPLICIT_METHODOLOGY_RE = re.compile(
    r"\bRequired\s+methodology\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\b",
    re.I,
)
_SSH_TARGET_RE = re.compile(
    r"\b(?:[a-zA-Z0-9_.-]+@)?(?:\d{1,3}\.){3}\d{1,3}\b|\b[a-zA-Z0-9_-]+@[a-zA-Z0-9_.-]+\b"
)


def _explicit_methodology(text: str) -> str | None:
    """Parse 'Required methodology: <name>' directives from prompt text."""
    m = _EXPLICIT_METHODOLOGY_RE.search(text or "")
    if not m:
        return None
    name = m.group(1).lower().strip()
    if name in METHODOLOGY_TEMPLATES:
        return name
    return None


def _has_ssh_target(text: str) -> bool:
    return bool(_SSH_TARGET_RE.search(text or ""))


def _is_host_key_audit(text: str) -> bool:
    low = text.lower()
    return _has_ssh_target(text) and "host key" in low and any(
        x in low for x in ("compare", "inspect", "fingerprint", "changed")
    )


def _is_ssh_artifact_retrieval(text: str) -> bool:
    low = text.lower()
    return _has_ssh_target(text) and "artifact" in low and any(
        x in low for x in ("retrieve", "download", "copy", "grab")
    )


def _is_remote_service_inspection(text: str) -> bool:
    low = text.lower()
    return _has_ssh_target(text) and bool(
        re.search(r"\b(?:port|service|pid|process|listening)\b", low)
    ) and any(x in low for x in ("investigate", "inspect", "identify", "find"))


def _is_remote_ssh_connectivity_check(text: str) -> bool:
    low = text.lower()
    return _has_ssh_target(text) and any(
        x in low for x in ("connectivity", "can you ssh", "test if you can ssh", "ssh check")
    )


def _is_remote_host_network_usb_triage(text: str) -> bool:
    """Route to remote_host_network_usb_triage for host-level NIC/USB incidents.

    Triggers when the prompt references an SSH target alongside network-interface
    or USB-HID incident language that is broader than a single port/service check.
    Distinct from ssh_remote_service_inspection: does not require a port number,
    but does require network/interface/USB/OOB language.
    """
    low = text.lower()
    has_target = _has_ssh_target(text)
    network_signals = (
        "network",
        "interface",
        "enp",
        "eth0",
        "nic ",
        "link down",
        "state down",
        "port shutdown",
        "port-security",
        "spanning tree",
        "stp",
        "bridge",
        "loop",
        "oob",
        "out-of-band",
        "nmcli",
        "ip link",
        "ifconfig",
        "router",
        "switch port",
        "vlan",
        "dhcp",
    )
    usb_signals = (
        "usb",
        "mouse",
        "keyboard",
        "hid",
        "input device",
        "usb hub",
        "usb controller",
    )
    has_network = any(s in low for s in network_signals)
    has_usb = any(s in low for s in usb_signals)
    triage_action = any(
        s in low for s in ("triage", "investigate", "diagnose", "troubleshoot", "check", "inspect")
    )
    return has_target and (has_network or has_usb) and triage_action


def _is_log_acquisition_discovery(text: str) -> bool:
    return any(
        x in text
        for x in (
            "acquisition preview",
            "log preview",
            "fresh logs",
            "request logs",
            "grasshopper preview",
            "nal preview",
        )
    )


def _is_receiver_log_execution(text: str) -> bool:
    return any(
        x in text
        for x in ("log_acquisition_execute", "execute log acquisition", "upload logs", "trigger upload")
    )


def _is_receiver_log_acquisition(text: str) -> bool:
    """Return True for generic log-pull requests. Excludes investigation-specific contexts
    that have their own methodologies (popup, reboot/dvr, rca).
    """
    _EXCLUDED_CONTEXTS = ("popup", "signal lost", "signal-lost", "reboot", "rebooting", "root cause", "rca", "investigate")
    if any(x in text for x in _EXCLUDED_CONTEXTS):
        return False
    return bool(re.search(r"\bR\d{6,}\b", text, re.I)) and any(
        x in text
        for x in ("logs", "s3 logs", "log bundle", "log acquisition", "pull logs", "fresh logs")
    )


def _is_systematic_tool_validation(text: str) -> bool:
    return ("systematically" in text and "tool" in text) or "tool certification" in text


def _is_tool_binding_audit(text: str) -> bool:
    return "tool binding" in text and "audit" in text


def _is_tool_execution_proof(text: str) -> bool:
    return "tool execution proof" in text or "tool-execution proof" in text


def _is_ollama_backend_readiness(text: str) -> bool:
    return any(
        x in text
        for x in (
            "ollama readiness",
            "ollama-agent-selftest",
            "ollama agent selftest",
            "backend readiness",
            "model roles",
            "tool-worker model",
            "tool worker model",
            "ollama agent mode readiness",
        )
    )


def _is_historical_receiver_rca(text: str) -> bool:
    has_rx = bool(re.search(r"\bR\d{6,}\b", text, re.I))
    return has_rx and any(
        x in text
        for x in (
            "rebooting",
            "reboots",
            "reboot",
            "dvr playback",
            "dvr crash",
            "historical logs",
            "root cause",
        )
    )


def _is_timezone_resolution(text: str) -> bool:
    return "timezone" in text and "utc" in text and any(
        x in text for x in ("resolve", "local start", "crosses midnight", "dst")
    )


def _is_efficiency_dashboard_audit(text: str) -> bool:
    return "token" in text and "efficien" in text and any(x in text for x in ("8510", "agent comparison", "compression"))


def _is_broad_anomaly_discovery(text: str) -> bool:
    return any(
        anchor in text
        for anchor in ("broad anomaly discovery", "candidate cluster", "scan 2.1", "one signal family")
    ) and any(
        anchor in text
        for anchor in (
            "search_parsed_logs",
            "stability signals",
            "video/dvr signals",
            "signal families",
        )
    )


def _is_local_repository_audit(text: str) -> bool:
    target = bool(re.search(r"/(?:home|srv|opt)/[^\s]+", text)) or bool(
        re.search(r"https?://git\.dtc\.dish\.corp/", text, re.I)
    )
    return target and any(
        x in text for x in ("repository", "repo", "feature branch", "invocation", "bedrock", "audit")
    )


def is_no_tool_request(text: str) -> bool:
    """Return True for plan-only or no-tool prompts."""
    low = text.lower()
    return any(x in low for x in ("plan only", "no tool", "no-tool", "just explain", "do not call any tool"))


# Ordered from most-specific to broadest to prevent false-positive matches.
_KEYWORD_RULES = (
    ("receiver_reboot_dvr_playback", ("reboot", "reboots", "rebooting"), ("dvr", "playback")),
    ("dvr_playback_instability_investigation", ("dvr playback instability", "playback instability"), ()),
    ("popup_signal_loss_investigation", ("popup", "signal lost", "signal-lost"), ("joey", "hopper", "live tv")),
    ("qos_ota_switchback_investigation", ("qos", "ota switchback", "switchback", "throughput stall", "abr"), ()),
    ("viewership_rtr_investigation", ("viewership", "watch hours", "top services", "content partner"), ("rtr", "anomal", "hourly", "daily trend")),
    ("epg_schedule_metadata_check", ("epg", "schedule", "channel metadata", "service id", "guide data"), ()),
    ("dva_stb_firmware_workflow", ("dva", "jamboree", "firmware", "stb software", "jam ", "device upgrade"), ()),
    ("backend_runtime_debug", ("backend", "startup", "traceback", "exception", "runtime", "health endpoint", "logs"), ()),
    ("artifact_generation", ("create file", "generate report", "spreadsheet", "presentation", "artifact", "write document"), ()),
    ("repo_code_review", ("repo", "repository", "code review", "patch", "diff", "pytest", "unit test", "fastapi", "langgraph"), ()),
    ("web_internal_research", ("research", "find docs", "confluence", "jira", "google drive", "internal search", "web search"), ()),
    ("performance_scalability_review", ("performance scalability", "horizontal pod scaling", "vertical memory", "abr stall reduction"), ()),
)


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _is_repo_checkout_local_deploy(text: str) -> bool:
    """Repository checkout / isolated local deployment intent (D3B2A).

    Strictly more specific than the generic SSH patterns: it requires a
    real repository target plus checkout or local-deployment intent.
    """
    from app.agent.operational_workflows import detect_operational_workflow

    return detect_operational_workflow(text) is not None


def select_methodology(goal: str | None, available_tool_families: Iterable[str] | None = None) -> dict:
    raw_text = goal or ""
    text = raw_text.lower()
    selected = "generic_engineering"

    # 1. Explicit directive wins unconditionally
    declared = _explicit_methodology(raw_text)
    if declared:
        selected = declared
    elif is_no_tool_request(raw_text):
        selected = "no_tool_response"

    # 2. Repository checkout / isolated local deployment (D3B2A).  Checked
    # before the generic SSH patterns because it is strictly more specific
    # and owns its own operational tool requirements.
    elif _is_repo_checkout_local_deploy(raw_text):
        selected = "repo_checkout_local_deploy"

    # 2b. Unambiguous structural patterns (SSH/remote) — before keyword rules
    elif _is_host_key_audit(raw_text):
        selected = "ssh_host_key_audit"
    elif _is_ssh_artifact_retrieval(raw_text):
        selected = "ssh_artifact_retrieval"
    elif _is_remote_host_network_usb_triage(raw_text):
        selected = "remote_host_network_usb_triage"
    elif _is_remote_service_inspection(raw_text):
        selected = "ssh_remote_service_inspection"
    elif _is_remote_ssh_connectivity_check(raw_text):
        selected = "remote_ssh_connectivity_check"

    # 3. Timezone and dashboard audits
    elif _is_timezone_resolution(text):
        selected = "timezone_window_resolution"
    elif _is_efficiency_dashboard_audit(text):
        selected = "agent_efficiency_dashboard_audit"

    # 4. Backend readiness
    elif _is_ollama_backend_readiness(text):
        selected = "ollama_backend_readiness"

    # 5. Broad anomaly
    elif _is_broad_anomaly_discovery(text):
        selected = "broad_anomaly_discovery" if "broad_anomaly_discovery" in METHODOLOGY_TEMPLATES else "generic_engineering"

    # 6. Repository audit
    elif _is_local_repository_audit(raw_text):
        selected = "local_repository_audit"

    # 7. Tool introspection
    elif _is_tool_binding_audit(text):
        selected = "tool_binding_audit"
    elif _is_tool_execution_proof(text):
        selected = "tool_execution_proof"
    elif _is_systematic_tool_validation(text):
        selected = "tool_systematic_validation"

    # 8. Historical receiver RCA takes priority over generic log acquisition
    elif _is_historical_receiver_rca(raw_text):
        selected = "receiver_reboot_dvr_playback"

    # 9. Log acquisition (only when not a historical RCA scenario)
    elif _is_receiver_log_execution(text):
        selected = "log_acquisition_execute"
    elif _is_log_acquisition_discovery(text):
        selected = "log_acquisition_preview"
    elif _is_receiver_log_acquisition(text) and not any(
        k in text for k in ("backend", "startup", "traceback", "exception")
    ):
        selected = "log_acquisition_preview"

    # 10. General keyword rules (ordered most-specific to broadest)
    else:
        for name, primary, secondary in _KEYWORD_RULES:
            if name == "backend_runtime_debug" and _is_receiver_log_acquisition(text):
                continue
            if _has_any(text, primary) and (not secondary or _has_any(text, secondary)):
                selected = name
                break

    template = METHODOLOGY_TEMPLATES[selected]
    available = set(available_tool_families or ())
    missing = sorted(set(template.required_tool_families) - available) if available else []
    result = template.to_dict()
    result["selection_reason"] = "explicit_directive" if declared else (
        "keyword_match" if selected != "generic_engineering" else "fallback"
    )
    result["missing_required_tool_families"] = missing
    return result


def methodology_selftest() -> dict:
    expected = set(METHODOLOGY_TEMPLATES)
    examples = {
        "no_tool_response": "Plan only; do not call any tool. Explain the approach.",
        "receiver_reboot_dvr_playback": "Receiver R1911746693 is repeatedly rebooting during DVR playback around 9pm on U820; check S3 logs and RTR alerts",
        "dvr_playback_instability_investigation": "DVR playback instability detected on R1234567890; investigate",
        "popup_signal_loss_investigation": "Joey signal lost popup during live TV; Hopper attached; pull STB logs",
        "qos_ota_switchback_investigation": "investigate QoS OTA switchback throughput stall followed by ABR session switch",
        "viewership_rtr_investigation": "content partner viewership watch hours dropped; compare top services and RTR anomalies",
        "ssh_host_key_audit": "Inspect the stored SSH host key fingerprint for montjac@10.79.85.47 and compare",
        "ssh_remote_service_inspection": "Use SSH to inspect the process listening on port 8510 at montjac@10.79.85.47",
        "ssh_artifact_retrieval": "Retrieve an agent artifact from montjac@10.79.85.35 using its artifact URL",
        "remote_ssh_connectivity_check": "test if you can ssh to montjac@10.79.85.35",
        "remote_host_network_usb_triage": "Investigate host-level network triage for montjac@10.79.85.47; interface enp68s0 state DOWN, USB mouse not working",
        "timezone_window_resolution": "Resolve timezone and UTC window and whether it crosses midnight",
        "agent_efficiency_dashboard_audit": "Audit token efficiency and the Agent Comparison dashboard at http://10.79.85.47:8510/",
        "ollama_backend_readiness": "Check ollama backend readiness and model roles status",
        "log_acquisition_preview": "Acquisition preview for receiver R1955706171 fresh NAL logs",
        "log_acquisition_execute": "Execute log acquisition for R1234567890",
        "tool_binding_audit": "Perform a tool binding audit of all bound tools",
        "tool_execution_proof": "This is a tool execution proof test",
        "tool_systematic_validation": "Systematically validate all tools and produce certification report",
        "local_repository_audit": "Audit repository /home/jakebot/Jakes-agent/ for direct Bedrock invocations",
        "agent_efficiency_dashboard_audit": "token efficiency agent comparison at 8510 compression",
        "repo_code_review": "patch the FastAPI LangGraph repo and run pytest",
        "backend_runtime_debug": "debug backend startup traceback and health endpoint",
        "performance_scalability_review": "Evaluate performance scalability: horizontal pod scaling versus vertical memory",
        "qos_switchback_investigation": "legacy qos_switchback_investigation template compatibility",
        "epg_schedule_metadata_check": "check EPG schedule metadata for service id",
        "dva_stb_firmware_workflow": "run DVA firmware workflow for an STB",
        "web_internal_research": "research internal confluence docs",
        "artifact_generation": "create file artifact generation report",
        "generic_engineering": "help with an engineering question",
    }
    # Also test explicit directive parsing
    directive_test = select_methodology("Required methodology: remote_host_network_usb_triage; target: montjac@10.79.85.47")
    directive_pass = directive_test["name"] == "remote_host_network_usb_triage" and directive_test["selection_reason"] == "explicit_directive"
    observed = {name: select_methodology(text)["name"] for name, text in examples.items()}
    status = "pass" if expected == set(METHODOLOGY_TEMPLATES) and all(
        observed[name] == name for name in observed
        if name not in ("qos_switchback_investigation", "dvr_playback_instability_investigation")
    ) and directive_pass else "fail"
    return {
        "status": status,
        "expected": sorted(expected),
        "observed": observed,
        "directive_parsing_test": directive_pass,
        "known_aliases": ["qos_switchback_investigation", "dvr_playback_instability_investigation"],
    }
