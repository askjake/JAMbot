"""Human-readable final report generation from the persisted run state."""
from __future__ import annotations

from typing import Any

from .state import RunState


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["_No rows._"]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |")
    return out


def render_report(state: RunState) -> str:
    data = state.data
    final = data.get("final", {})
    lines = [
        "# Nightly RCA v6 — Implementation Run Report",
        "",
        f"- Run ID: `{state.run_id}`",
        f"- Mode: `{state.mode}`",
        f"- Status: `{state.status}`",
        f"- Started: `{state.started_at}`",
        f"- Completed: `{state.completed_at or 'in progress'}`",
        f"- Verdict: **{final.get('executive_verdict', 'not finalized')}**",
        "",
        "## Environment and source inventory",
        "",
        f"Active date window: `{', '.join(state.active_date_window) or 'UNKNOWN'}`",
        "",
    ]
    env = data.get("environment", {}).get("environment", {})
    if isinstance(env, dict):
        lines.extend(_table(["Field", "Observed"], [[k, v] for k, v in sorted(env.items()) if k not in {"allowed_tools"}]))

    effective = data.get("effective_configuration", {})
    if isinstance(effective, dict) and effective:
        lines.extend(["", "### Effective configuration provenance", ""])
        safe_fields = [
            "effective_mode", "effective_mode_source", "role",
            "write_authorization_present", "notify_enabled", "webhook_configured",
            "output_dir", "max_upload_files", "pending_batch_size",
            "t2i_atlas_enabled", "t2i_atlas_profile",
            "t2i_atlas_font_identity_pinned",
            "configured_server_families", "env_file_path", "env_file_exists",
            "cron_timezone", "cron_timezone_evidence", "cron_timezone_confidence",
        ]
        lines.extend(_table(
            ["Field", "Effective value / evidence"],
            [[field, effective.get(field, "")] for field in safe_fields],
        ))
        sources = effective.get("configuration_sources", {})
        if isinstance(sources, dict):
            lines.append("")
            lines.extend(_table(
                ["Setting", "Source"],
                [[key, value] for key, value in sorted(sources.items())],
            ))

    atlas = data.get("t2i_atlas")
    if isinstance(atlas, dict):
        lines.extend(["", "## T2I Log Atlas", ""])
        lines.extend(_table(
            ["Field", "Observed"],
            [
                ["Status", atlas.get("status", "UNKNOWN")],
                ["Codec", atlas.get("codec", "")],
                ["Profile", atlas.get("profile", "")],
                ["Source SHA256", atlas.get("source_sha256", "")],
                ["Source events", atlas.get("source_records", "")],
                ["Templates", atlas.get("template_count", "")],
                ["Pages", atlas.get("page_count", "")],
                ["Manifest", atlas.get("manifest_path", "")],
                ["Compact records", atlas.get("compact_records_path", "")],
            ],
        ))
        if atlas.get("status") == "GENERATION_ERROR":
            lines.append("")
            lines.append(
                f"Generation error: `{atlas.get('error_class', 'UnknownError')}: "
                f"{atlas.get('error', '')}`"
            )

    lines.extend(["", "## Unresolved case resolution", ""])
    unresolved = data.get("unresolved_resolution", {}).get("rows", [])
    lines.extend(_table(
        ["Case", "Profile", "Receiver", "Date", "Classification", "Action"],
        [[r.get("case_id"), r.get("profile"), r.get("receiver"), r.get("date"), r.get("new_classification"), r.get("action_taken")] for r in unresolved],
    ))

    lines.extend(["", "## Candidate clusters", ""])
    lines.extend(_table(
        ["Candidate", "Signal", "Families", "Count", "Anomaly", "Recurrence", "Priority"],
        [[c.get("candidate_id"), c.get("suspected_profile"), ", ".join(c.get("signal_families", [])), c.get("alert_count"), c.get("anomaly_score"), c.get("recurrence_nights"), c.get("priority_score")] for c in data.get("candidate_clusters", [])],
    ))

    lines.extend(["", "## Profile analysis and validation", ""])
    lines.extend(_table(
        ["Profile", "Source", "Validation", "Receiver", "Logs present", "Reason"],
        [[v.get("profile_id"), v.get("profile_source"), v.get("classification"), v.get("best_receiver"), v.get("required_logs_present"), v.get("reason")] for v in data.get("profile_validations", [])],
    ))

    lines.extend(["", "## Data collection, cases, bundles, and packets", ""])
    lines.extend(_table(
        ["Profile", "Receiver", "Upload", "Receipt"],
        [[u.get("profile_id"), u.get("receiver_id"), u.get("status"), u.get("receipt_verified", "n/a")] for u in data.get("uploads", [])],
    ))
    lines.append("")
    lines.extend(_table(
        ["Case", "Logical key", "Status", "Bundle", "Packet"],
        [[c.get("case_id"), c.get("logical_case_key"), c.get("status"), _lookup(data.get("bundles", []), c.get("case_id"), "status"), _lookup(data.get("packets", []), c.get("case_id"), "packet_status")] for c in data.get("cases", [])],
    ))

    lines.extend(["", "## Dashboard and canaries", ""])
    lines.append(f"Dashboard status: `{data.get('dashboard', {}).get('status', 'UNKNOWN')}`")
    lines.append("")
    lines.extend(_table(
        ["Canary", "Pass", "Evidence status"],
        [[c.get("canary"), c.get("pass"), _evidence_status(c.get("evidence"))] for c in data.get("canaries", {}).get("canaries", [])],
    ))

    lines.extend(["", "## Pending investigation backlog", ""])
    pending_metrics = data.get("pending_selector_metrics", {})
    if pending_metrics:
        lines.append(f"Automatic receipt reconciliation:")
        lines.append(f"- {pending_metrics.get('pending_selected_logical', '?')} logical investigations selected.")
        _deferred_l = pending_metrics.get("pending_deferred_batch_logical", 0)
        if _deferred_l:
            lines.append(f"- {_deferred_l} eligible logical investigations were deferred by the batch limit.")
        _backoff_p = pending_metrics.get("pending_backoff_physical", 0)
        if _backoff_p:
            lines.append(f"- {_backoff_p} physical tracker rows are in backoff.")
        _dup_p = pending_metrics.get("pending_duplicate_suppressed_physical", 0)
        if _dup_p:
            lines.append(f"- {_dup_p} additional physical tracker rows represent duplicate logical investigations.")
        lines.append("")
        lines.extend(_table(
            ["Metric", "Value"],
            [[k, v] for k, v in sorted(pending_metrics.items())],
        ))
    else:
        lines.append("_No pending selector metrics available._")

    pending_rows = data.get("pending_investigations", [])
    if isinstance(pending_rows, list) and pending_rows:
        lines.extend(["", "### Typed pending tracker identities", ""])
        lines.extend(_table(
            [
                "Currentness", "Profile", "Receiver", "Tracker ID",
                "Local correlation", "Grasshopper request", "Identifier provenance",
                "Workflow / receipt",
            ],
            [
                [
                    row.get("currentness", "UNKNOWN"),
                    row.get("profile_id") or row.get("issue_profile") or "",
                    row.get("receiver_id", ""),
                    row.get("tracker_id") or "—",
                    row.get("local_correlation_id") or "—",
                    row.get("grasshopper_request_id") or "—",
                    row.get("identifier_provenance", "UNKNOWN"),
                    row.get("workflow_status") or row.get("receipt_status") or row.get("status") or "",
                ]
                for row in pending_rows[:50]
                if isinstance(row, dict)
            ],
        ))
        if len(pending_rows) > 50:
            lines.append("")
            lines.append(
                f"_{len(pending_rows) - 50} additional physical tracker rows omitted from this report._"
            )

    lines.extend(["", "## Final numbers", ""])
    nums = final.get("final_numbers", {})
    lines.extend(_table(["Metric", "Value"], [[k, v] for k, v in nums.items()]))

    lines.extend(["", "## Remaining blockers", ""])
    for blocker in final.get("remaining_blockers", []):
        lines.append(f"- {blocker}")

    lines.extend(["", "## Bounded learning recommendations", ""])
    recs = final.get("learning_recommendations", [])
    if not recs:
        lines.append("_No adaptive changes recommended._")
    for rec in recs:
        lines.append(f"- **{rec.get('type')}**: {rec.get('reason')} Next: {rec.get('action')}")

    lines.extend([
        "", "## Safety model", "",
        "The pipeline never auto-promotes a root cause from alert volume alone. New profiles remain review-gated, write operations require explicit commit authorization, and evidence gaps remain visible rather than being converted into facts.",
        "",
    ])
    return "\n".join(lines)


def _lookup(rows: list[dict[str, Any]], case_id: str, field: str) -> Any:
    for row in rows:
        if row.get("case_id") == case_id:
            return row.get(field, "")
    return ""


def _evidence_status(value: Any) -> str:
    if not isinstance(value, dict):
        return "unstructured"
    return str(value.get("status") or value.get("integrity_status") or value.get("code") or "observed")
