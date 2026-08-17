"""Verified tool registry and routing from TOOL_REFERENCE.md v5.3."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolContract:
    name: str
    server: str
    write_capable: bool = False
    operator_required: bool = False
    two_step_action: str | None = None


_TOOL_ROWS = [
    ("get_tool_info", "s3_stb_logs", False, False, None),
    ("get_heavy_auth_status", "s3_stb_logs", False, False, None),
    ("list_dates", "s3_stb_logs", False, False, None),
    ("list_parsed_dates", "s3_stb_logs", False, False, None),
    ("search_parsed_logs", "s3_stb_logs", False, False, None),
    ("list_files", "s3_stb_logs", False, False, None),
    ("read_log", "s3_stb_logs", False, False, None),
    ("count_alerts", "rtr_alerts_mcp", False, False, None),
    ("list_anomalies", "rtr_alerts_mcp", False, False, None),
    ("search_investigation_cases", "s3_stb_logs", False, False, None),
    ("record_investigation_case", "s3_stb_logs", True, False, None),
    ("record_case_outcome", "s3_stb_logs", True, False, None),
    ("get_issue_profile_catalog", "s3_stb_logs", False, False, None),
    ("register_issue_profile", "s3_stb_logs", True, False, None),
    ("grasshopper_plan_profile_upload", "grasshopper_mcp", False, False, None),
    ("grasshopper_upload_profile_logs", "grasshopper_mcp", True, False, None),
    ("verify_profile_upload_receipt", "s3_stb_logs", False, False, None),
    ("plan_profile_investigation", "s3_stb_logs", False, False, None),
    ("verify_receiver_log_coverage", "s3_stb_logs", False, False, None),
    ("human_review_dashboard_status", "s3_stb_logs", False, True, None),
    ("human_review_materialize_engineer_contexts", "s3_stb_logs", True, True, None),
    ("human_review_fix_lineage_refresh", "s3_stb_logs", True, True, None),
    ("human_review_fix_lineage", "s3_stb_logs", False, True, None),
    ("create_human_review_queue", "s3_stb_logs", True, False, "create"),
    ("list_human_review_queue_items", "s3_stb_logs", False, False, None),
    ("build_human_evidence_bundle", "s3_stb_logs", True, False, "build"),
    ("export_human_adjudication_packet", "s3_stb_logs", True, False, "export"),
    ("audit_human_adjudication_packet", "s3_stb_logs", False, False, None),
    ("validate_human_adjudication_packet_readiness", "s3_stb_logs", False, False, None),
    ("audit_human_review_queue_integrity", "s3_stb_logs", False, False, None),
    ("audit_case_adjudication_staleness", "s3_stb_logs", False, False, None),
    ("get_learning_system_status", "s3_stb_logs", False, False, None),
    ("list_packet_integrity_failures", "s3_stb_logs", False, False, None),
    ("list_upload_trackers", "s3_stb_logs", False, False, None),
    ("record_upload_tracker", "s3_stb_logs", True, False, None),
    ("update_upload_tracker_from_s3", "s3_stb_logs", True, False, None),
]

TOOL_CONTRACTS = {
    row[0]: ToolContract(*row) for row in _TOOL_ROWS
}

BANNED_TOOLS = frozenset({
    "search_jira", "get_jira_issue", "list_jira_projects",
    "get_jira_issue_comments", "get_jira_issue_changelog",
    "get_jira_issue_attachments", "get_dashboard_status",
    "get_dashboard_metrics", "list_s3_stb_log_dates",
    "list_human_review_queue", "list_evidence_bundles",
    "list_response_contracts", "list_grasshopper_uploads",
    "list_issue_profiles", "search_rtr_alerts",
    # Legacy app-only tools that are not in the attached verified reference.
    "get_stb_issue_profile_catalog", "alert_time_series",
    "export_receivers_csv", "investigate_top_offenders_from_report_text",
    "build_log_capsule", "tool_trigger_analysis",
    "tool_get_analysis_status", "tool_get_analysis_result",
})


def contract_for(tool_name: str) -> ToolContract:
    if tool_name in BANNED_TOOLS:
        raise ValueError(f"BANNED_TOOL:{tool_name}")
    try:
        return TOOL_CONTRACTS[tool_name]
    except KeyError as exc:
        raise ValueError(f"UNVERIFIED_TOOL:{tool_name}") from exc
