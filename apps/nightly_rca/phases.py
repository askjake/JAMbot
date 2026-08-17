"""Implementation of Blocks 1A-4 as a serial, state-carrying workflow."""
from __future__ import annotations

import logging
from collections import Counter
from datetime import date
import re
from typing import Any

from .config import Settings
from .executor import SerialExecutor
from .history import learning_recommendations, load_history_signals
from .logic import (
    as_list,
    blocking_codes,
    case_date,
    case_id,
    case_profile,
    case_receiver,
    classify_fix_lineage,
    coverage_summary,
    discover_clusters,
    draft_profile,
    extract_candidate_receivers,
    extract_hash,
    extract_identifier,
    json_text,
    logical_case_key,
    match_clusters_to_profiles,
    normalize_cases,
    normalize_catalog,
    parse_dates,
    rows_from,
    select_validation_targets,
)
from .pending_selector import (
    select_pending_investigations,
    selector_metrics,
    update_metadata_after_attempt,
)
from .state import (
    RunState,
    RunStore,
    normalize_pending_identifier_provenance,
    evaluate_duplicate_preflight,
    stable_hash,
    utc_now,
)
from .transport import ToolTransportError

from .grasshopper_contract import (
    AcquisitionState,
    GrasshopperPlanResult,
    GrasshopperUploadResult,
    IdentifierSet,
    ProfileMetadata,
    build_grasshopper_arguments,
    build_identifier_set,
    profile_metadata_from_catalog,
    determine_acquisition_state,
    parse_plan_response,
    parse_upload_response,
    ACQUISITION_STATE_META,
)

log = logging.getLogger("nightly_rca.phases")


ALERT_FAMILIES: list[tuple[str, str]] = [
    ("stability", "PMGR_UNEXPECTED_EXIT,RTRD_REBOOT,RTRD_PRAM_SEGV,DATA_WATCH_DOG_REPEATED_FAILURES,COLD_BOOT_DETECTED"),
    ("process_service", (
        "QTUI_TYPE_ERROR,QTUI_REFERENCE_ERROR,QTUI_ERROR_INITIALIZATION,"
        "SGS_RESPONSE_ERR_UNKNOWN,SGSPROXY_REQUEST_EXPIRED,LATENT_RESPONSE"
    )),
    ("video_dvr", (
        "VAR_PLAYERHEALTH_VIDEO_FREEZE_DETECTED,VAR_PLAYERHEALTH_SUSTAINED_FREEZE_DETECTED,"
        "PLC_ERROR_DVRSRC_STUCK,PLC_ERROR_MESSAGE_ABANDONED,PLC_EVENT_TRANSITION_FAILURE,"
        "PLC_ERROR_SEGMENT_LLOTT"
    )),
    ("guide_epg_popup", (
        "1031,LLOTT_SCHEDULE_DOWNLOAD_ERROR,LLOTT_SCHEDULE_MISSING_EVENT_INFO,"
        "PLC_BUFFERING_LLOTT,LLOTT_SET_ENVIRONMENT_DOWNLOAD_ERROR,"
        "FAVORITE_LIST_DATA_LOAD_ERROR_FROM_HOPPER,LOCKED_LIST_DATA_LOAD_ERROR_FROM_HOPPER"
    )),
    ("networking_topology", "LOST_LINK_TO_HOPPER,NETRA_MULTIPLE_SUBNETS_DETECTED,NETRA_DHCP_IP_CHANGED,IPLL_FALLBACK_DUPLICATE_SUID_FOUND,IPLL_FALLBACK_SIGNAL_LOSS_TUNE"),
    ("drm_playback", "PLC_ERROR_DRM_NO_DECRYPTION_KEY,PLC_ERROR_DRM_SERVER_TIMEOUT_PLAY_TRY_1,PLC_ERROR_DRM_SERVER_TIMEOUT_PLAY_TRY_2,PLC_ERROR_DRM_FAILED_REKEY,PLC_ERROR_DRM_SERVER_NOT_REACHABLE,2004,LOADER_FAILURE"),
    ("update_firmware", (
        "IM_FW_DL_FAILURE,IM_BACKUP_STB_FAILURE,CONFIG_FETCHING_ISSUES,"
        "EXCEEDED_DETACHED_DEFAULT_TUNE_THRESHOLD,EXCEEDED_LAUNCHER_START_DEFAULT_TUNE_THRESHOLD"
    )),
    ("app_service", "YOUTUBE_SHELF_HTTP_REQUEST_FAILURE,YOUTUBE_SHELF_LOAD_ERROR,APP_LAUNCH_FAILURE,APP_CRASH_DETECTED"),
]

IN_SCOPE_CASE_STATUSES = {"unreviewed", "confirmed_issue", "open", "active", "pending_review"}
OUT_OF_SCOPE_CASE_STATUSES = {"not_this_profile", "confirmed_no_issue", "not_a_real_issue", "dismissed", "closed"}


def _resp(result: dict[str, Any]) -> Any:
    return result.get("response") if result.get("status") == "OK" else None


def _code_available(result: dict[str, Any]) -> bool:
    """Return True if a code_tools call succeeded (not TOOL_NOT_FOUND or transport error)."""
    return result.get("status") == "OK"


# Alert names that map to a known source symbol for code-tool enrichment
_ALERT_TO_SYMBOL: dict[str, tuple[str, str]] = {
    "PLC_ERROR_SEGMENT_LLOTT": ("stbctrl", "PLC_ERROR_SEGMENT_LLOTT"),
    "LOADER_FAILURE": ("stbctrl", "LOADER_FAILURE"),
    "YOUTUBE_SHELF_HTTP_REQUEST_FAILURE": ("ATV_Qt_UI", "YOUTUBE_SHELF_HTTP_REQUEST_FAILURE"),
    "FAVORITE_LIST_DATA_LOAD_ERROR_FROM_HOPPER": ("stbctrl", "FAVORITE_LIST_DATA_LOAD_ERROR_FROM_HOPPER"),
    "LOCKED_LIST_DATA_LOAD_ERROR_FROM_HOPPER": ("stbctrl", "LOCKED_LIST_DATA_LOAD_ERROR_FROM_HOPPER"),
    "QTUI_ERROR_INITIALIZATION": ("ATV_Qt_UI", "QTUI_ERROR_INITIALIZATION"),
    "EXCEEDED_LAUNCHER_START_DEFAULT_TUNE_THRESHOLD": ("stbctrl", "EXCEEDED_LAUNCHER_START_DEFAULT_TUNE_THRESHOLD"),
}


def _status_text(row: dict[str, Any]) -> str:
    return str(row.get("outcome_status") or row.get("status") or "").lower()


def _existing_jira_ids(case: dict[str, Any]) -> set[str]:
    values: list[Any] = []
    for key in ("jira", "jira_id", "jira_ids", "jira_issue_ids", "jira_fix_ids"):
        values.extend(as_list(case.get(key)))
    out: set[str] = set()
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            if item.strip():
                out.add(item.strip())
    return out


def _first_date(value: Any, fallback: str) -> str:
    if isinstance(value, dict):
        for key in ("date", "event_date", "selected_date", "newest"):
            if value.get(key):
                return str(value[key])[:10]
        candidates = rows_from(value, "candidates", "receivers", "items", "results")
        for candidate in candidates:
            for key in ("date", "event_date"):
                if candidate.get(key):
                    return str(candidate[key])[:10]
    return fallback


def _file_count(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    for key in ("file_count", "files_planned", "planned_file_count", "count"):
        raw = value.get(key)
        if isinstance(raw, int):
            return raw
    files = value.get("files")
    return len(files) if isinstance(files, list) else None


def _write_performed(value: Any) -> bool:
    return isinstance(value, dict) and value.get("write_performed") is True


def _call_pass(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("ok") is False:
        return False
    for key in ("pass", "passed", "integrity_ok", "reviewer_ready", "current", "active", "enabled", "token_active"):
        if key in value:
            return bool(value[key])
    status = str(value.get("status") or value.get("integrity_status") or "").upper()
    if any(bad in status for bad in (
        "FAIL", "BROKEN", "STALE", "ERROR", "BLOCK", "PARTIAL",
        "UNAVAILABLE", "INACTIVE", "DISABLED", "EXPIRED", "UNKNOWN",
    )):
        return False
    if any(good in status for good in (
        "PASS", "READY", "ACTIVE", "HEALTHY", "COMPLETE", "CURRENT",
        "OK", "VERIFIED", "SUCCESS",
    )):
        return True
    return value.get("ok") is True


def _result_code(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return str(value.get("result") or value.get("status") or value.get("code") or "").upper()


# ── Expected-negative canary classifier ──────────────────────────────────────
# Narrow matcher: only a recognized "no logs" condition passes.
# Authentication, authorization, timeout, tool-not-found, schema errors all FAIL.

_NO_LOGS_PATTERNS: list[str] = [
    "no logs found for receiver",
    "no data found for receiver",
    "filenotfounderror",
    "no s3 data",
    "no_logs",
    "no_data",
    "receiver not found in s3",
]

_REJECT_PATTERNS: list[str] = [
    "401", "403", "accessdenied", "access denied",
    "unauthorized", "authentication", "forbidden",
    "timeout", "timed out", "connection reset",
    "tool_not_found", "unknown tool", "tool not found",
    "invalid arguments", "schema validation", "schema_validation",
    "invalid_params", "missing required",
]


def _classify_expected_no_logs(result: dict[str, Any], expected_receiver: str) -> dict[str, Any]:
    """Classify a list_dates call against the synthetic test receiver.

    Returns a structured evidence dict with pass=True only when the response
    or error clearly indicates the configured receiver has no logs.
    """
    status = result.get("status", "")
    error = str(result.get("error") or "")
    response = result.get("response")
    error_lower = error.lower()
    response_str = str(response).lower() if response is not None else ""

    evidence: dict[str, Any] = {
        "receiver_id": expected_receiver,
        "tool": "list_dates",
        "observed_status": status,
        "observed_error_class": status,
    }

    # If the call succeeded with OK status but returned data, the synthetic
    # receiver unexpectedly HAS logs — this is a canary failure.
    if status == "OK" and response is not None:
        # Check if response indicates no data via structured fields
        if isinstance(response, dict):
            dates = response.get("dates") or response.get("items") or []
            if not dates:
                evidence.update({"status": "EXPECTED_NO_LOGS_CONFIRMED", "expected_condition": "no_logs", "pass": True})
                return evidence
        evidence.update({
            "status": "UNEXPECTED_LOGS_FOR_SYNTHETIC_RECEIVER",
            "expected_condition": "no_logs",
            "pass": False,
            "detail": "The synthetic test receiver unexpectedly returned log data.",
        })
        return evidence

    # Check for rejection patterns first (these MUST fail)
    for pattern in _REJECT_PATTERNS:
        if pattern in error_lower or pattern in response_str:
            evidence.update({
                "status": "CANARY_FAILED_INFRASTRUCTURE",
                "expected_condition": "no_logs",
                "pass": False,
                "detail": f"Rejected: matched infrastructure failure pattern '{pattern}'",
                "matched_reject_pattern": pattern,
            })
            return evidence

    # Check for recognized no-logs patterns (these pass)
    for pattern in _NO_LOGS_PATTERNS:
        if pattern in error_lower or pattern in response_str:
            # Additional check: the receiver must be mentioned in the error/response
            # unless the pattern itself is generic enough (e.g., from a FileNotFoundError)
            receiver_mentioned = expected_receiver.lower() in error_lower or expected_receiver.lower() in response_str
            generic_no_logs = pattern in ("filenotfounderror", "no_logs", "no_data", "no s3 data")
            if receiver_mentioned or generic_no_logs:
                evidence.update({
                    "status": "EXPECTED_NO_LOGS_CONFIRMED",
                    "expected_condition": "no_logs",
                    "pass": True,
                    "matched_pattern": pattern,
                })
                return evidence

    # No recognized pattern matched — treat as infrastructure/transport failure
    evidence.update({
        "status": "CANARY_FAILED_UNRECOGNIZED_ERROR",
        "expected_condition": "no_logs",
        "pass": False,
        "detail": f"Error did not match any recognized no-logs pattern: {error[:200]}",
    })
    return evidence


def _persisted_or_idempotent(value: Any, *identity_keys: str) -> bool:
    """Treat immutable same-hash ALREADY_EXISTS responses as successful persistence."""
    if not isinstance(value, dict) or value.get("ok") is False:
        return False
    if _write_performed(value):
        return True
    code = _result_code(value)
    if "ALREADY_EXISTS" in code:
        return True
    return any(bool(extract_identifier(value, key)) for key in identity_keys)


def _receipt_ready(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if as_list(value.get("landed_log_types")):
        return True
    statuses = {
        str(value.get("receipt_status") or "").lower(),
        str(value.get("workflow_status") or "").lower(),
        str(value.get("status") or "").lower(),
    }
    return bool(statuses & {
        "receipt_complete", "logs_landed", "ready_for_analysis",
        "ready", "complete", "completed",
    })


def _receipt_check_error(value: Any) -> bool:
    """Return True if tracker poll signals a transient receipt-check error (fast-fail)."""
    if not isinstance(value, dict):
        return False
    code = str(value.get("error_code") or value.get("receipt_status") or "").lower()
    return code == "receipt_check_error"


def _coverage_ready_receiver(value: Any, preferred: list[str] | None = None) -> str:
    summary = coverage_summary(value)
    rows = summary.get("rows", [])
    ready: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("coverage_status") or row.get("status") or "").lower()
        explicit = row.get("required_logs_present")
        logs = as_list(row.get("available_log_types") or row.get("log_types"))
        is_ready = explicit is True or (
            explicit is not False
            and status in {"ready_for_analysis", "ready", "complete", "covered", "ok"}
            and (bool(logs) or status in {"ready_for_analysis", "complete", "covered", "ok"})
        )
        if is_ready:
            rid = str(row.get("receiver_id") or row.get("rxid") or row.get("device_id") or "")
            if rid:
                ready.append(rid)
    for rid in preferred or []:
        if rid in ready:
            return rid
    return ready[0] if ready else ""


def _tracker_rows(value: Any) -> list[dict[str, Any]]:
    return rows_from(value, "trackers", "items", "results")


class PhaseRunner:
    def __init__(self, settings: Settings, state: RunState, store: RunStore, executor: SerialExecutor):
        self.settings = settings
        self.state = state
        self.store = store
        self.executor = executor

    def checkpoint(self, phase: str, payload: Any) -> Any:
        self.store.save_phase(phase, payload)
        manifest = self.state.data.setdefault("phase_manifest", {})
        manifest[phase] = {
            "status": payload.get("status") if isinstance(payload, dict) else "RECORDED",
            "payload_hash": stable_hash(payload),
            "file": f"phase_{phase.replace('.', '_').replace('/', '_')}.json",
        }
        self.store.save(self.state)
        return payload

    def effective_queue_id(self) -> str:
        """Return a run-stable immutable queue id rather than reusing `main`."""
        existing = str(self.state.data.get("effective_queue_id") or "")
        if existing:
            return existing
        base = re.sub(r"[^a-zA-Z0-9_-]", "-", self.settings.queue_id).strip("-") or "main"
        run = re.sub(r"[^a-zA-Z0-9_-]", "-", self.state.run_id).strip("-")
        queue_id = f"{base}-{run}"[:120]
        self.state.data["effective_queue_id"] = queue_id
        return queue_id

    def _grasshopper_profile_metadata(self, profile_id: str) -> ProfileMetadata:
        return profile_metadata_from_catalog(
            profile_id,
            self.state.data.get("profile_catalog") or self.state.data.get("profile_catalog_initial") or {},
        )

    async def phase_0_environment(self) -> dict[str, Any]:
        env = await self.executor.call(phase="0", step="0.1", tool="get_tool_info", arguments={}, required=False)
        auth = await self.executor.call(phase="0", step="0.2", tool="get_heavy_auth_status", arguments={}, required=False)
        payload = {
            "status": "PHASE_0_IDENTITY_PASS" if _resp(env) or _resp(auth) else "PHASE_0_ENV_UNAVAILABLE",
            "environment": _resp(env) or {},
            "heavy_auth": _resp(auth) or {},
            "operator_role": self.settings.role,
            "commit_mode": self.settings.commit,
            "effective_configuration": self.state.data.get("effective_configuration", {}),
        }
        self.state.data["environment"] = payload
        return self.checkpoint("0", payload)

    async def phase_1_source_inventory(self) -> dict[str, Any]:
        calls: list[tuple[str, str, dict[str, Any]]] = [
            ("1.1", "get_issue_profile_catalog", {}),
            ("1.2", "list_parsed_dates", {"receiver_id": "", "limit": 30}),
            ("1.3", "search_investigation_cases", {"outcome_status": "", "limit": 10}),
            ("1.4", "list_human_review_queue_items", {"queue_id": self.settings.queue_id, "limit": 10}),
            ("1.5", "human_review_dashboard_status", {"scan_limit": 20}),
            ("1.6", "list_packet_integrity_failures", {"limit": 20}),
            ("1.7", "list_upload_trackers", {"limit": 200, "max_inline_trackers": 200, "pending_only": True, "persist_full_report": False}),
            ("1.8", "get_learning_system_status", {"scan_limit": 50}),
        ]
        results: dict[str, Any] = {}
        for step, tool, args in calls:
            result = await self.executor.call(phase="1", step=step, tool=tool, arguments=args)
            results[tool] = _resp(result)

        lineage_tool = "human_review_fix_lineage_refresh" if self.settings.commit else "human_review_fix_lineage"
        lineage = await self.executor.call(
            phase="1", step="1.9", tool=lineage_tool,
            arguments={"logical_case_key": "GLOBAL", "scan_limit": 100},
        )
        results[lineage_tool] = _resp(lineage)

        active_dates = parse_dates(results.get("list_parsed_dates"))[:7]
        self.state.active_date_window = active_dates
        cases = normalize_cases(results.get("search_investigation_cases"))
        catalog = normalize_catalog(results.get("get_issue_profile_catalog"))
        self.state.data["case_inventory_all"] = cases
        self.state.data["profile_catalog_initial"] = catalog

        # Rehydrate unfinished log-acquisition work from both the MCP tracker
        # store and the local cross-run ledger. MCP is authoritative when rows
        # share a tracker id; local state is a durable fallback for tracker-write
        # or transport failures.
        pending_by_key: dict[str, dict[str, Any]] = {}
        local_pending_rows = self.store.load_pending_investigations()
        local_pending_available = self.store.pending_inventory_available
        source_rows = local_pending_rows + _tracker_rows(
            results.get("list_upload_trackers")
        )
        for tracker in source_rows:
            if not isinstance(tracker, dict):
                continue
            normalized = normalize_pending_identifier_provenance(
                tracker, current_run_id=self.state.run_id
            )
            workflow = str(normalized.get("workflow_status") or "").upper()
            receipt = str(normalized.get("receipt_status") or normalized.get("status") or "").lower()
            if workflow.startswith("READY_FOR_ANALYSIS") or workflow in {"COMPLETE", "CLOSED"} or receipt in {"receipt_complete", "logs_landed", "ready_for_analysis", "complete"}:
                continue
            profile_id = str(normalized.get("profile_id") or normalized.get("issue_profile") or "")
            receiver_id = str(normalized.get("receiver_id") or "")
            physical_identity = str(normalized.get("physical_identity") or "")
            if not (profile_id and receiver_id and physical_identity):
                continue
            row = {
                **normalized,
                "profile_id": profile_id,
                "issue_profile": profile_id,
                "receiver_id": receiver_id,
            }
            row.setdefault("workflow_status", "WAITING_FOR_LOGS")
            pending_by_key[physical_identity] = row
        if local_pending_available:
            self.store.save_pending_investigations(list(pending_by_key.values()))
            pending = self.store.load_pending_investigations()
        else:
            # Preserve the unreadable ledger for operator recovery. Remote rows
            # may still be reported in-memory, but duplicate-sensitive writes
            # remain blocked by the source-availability contract.
            pending = list(pending_by_key.values())
        self.state.data["pending_investigations"] = pending
        payload = {
            "status": "OK" if active_dates else "PARTIAL",
            "active_date_window": active_dates,
            "sources": results,
            "case_count": len(cases),
            "profile_count": len(catalog),
            "pending_investigation_count": len(pending),
        }
        self.state.data["source_inventory"] = {
            "status": payload["status"], "active_date_window": active_dates,
            "case_count": len(cases), "profile_count": len(catalog),
            "pending_investigation_count": len(pending),
            "source_availability": {
                **{name: value is not None for name, value in results.items()},
                "local_pending_ledger": bool(local_pending_available),
            },
            "source_error_classes": {
                "local_pending_ledger": self.store.pending_load_error_class,
            },
        }
        return self.checkpoint("1", payload)

    async def phase_d1_unresolved(self) -> dict[str, Any]:
        search = await self.executor.call(
            phase="D1", step="D1.1", tool="search_investigation_cases",
            arguments={"outcome_status": "unreviewed", "limit": 200}, required=False,
        )
        cases = normalize_cases(_resp(search))
        in_scope = [c for c in cases if _status_text(c) in IN_SCOPE_CASE_STATUSES or not _status_text(c)]
        out_scope = [c for c in cases if _status_text(c) in OUT_OF_SCOPE_CASE_STATUSES]
        rows: list[dict[str, Any]] = []
        refresh_tool = "human_review_fix_lineage_refresh" if self.settings.commit else "human_review_fix_lineage"

        for idx, case in enumerate(in_scope[: self.settings.max_unresolved_cases], start=1):
            profile = case_profile(case)
            receiver = case_receiver(case)
            event_date = case_date(case)
            key = logical_case_key(profile, receiver, event_date)
            if not (profile and receiver and event_date):
                rows.append({
                    "case_id": case_id(case), "profile": profile, "receiver": receiver,
                    "date": event_date, "previous_status": _status_text(case),
                    "logical_case_key": key, "new_classification": "CANNOT_DETERMINE",
                    "jira_ids": [], "action_taken": "SKIPPED_INVALID_CASE_KEY",
                    "lineage": None, "details": None,
                })
                continue
            lineage = await self.executor.call(
                phase="D1", step=f"D1.2.{idx}", tool=refresh_tool,
                arguments={"logical_case_key": key, "scan_limit": 100},
            )
            lineage_response = _resp(lineage)
            classification = classify_fix_lineage(lineage_response)
            if classification == "FIX_LINEAGE_UPDATED" and isinstance(lineage_response, dict):
                observed_issues = {str(x) for x in as_list(lineage_response.get("jira_issue_ids")) if x}
                if observed_issues and observed_issues.issubset(_existing_jira_ids(case)):
                    classification = "STILL_OPEN"
            details = None
            if isinstance(lineage_response, dict) and as_list(lineage_response.get("jira_fix_ids")):
                details_call = await self.executor.call(
                    phase="D1", step=f"D1.3.{idx}", tool="human_review_fix_lineage",
                    arguments={"logical_case_key": key, "scan_limit": 100},
                )
                details = _resp(details_call)

            action = "NONE"
            if classification in {"NEWLY_RESOLVED", "FIX_LINEAGE_UPDATED"} and self.settings.commit:
                jira_ids = as_list((lineage_response or {}).get("jira_fix_ids")) or as_list((lineage_response or {}).get("jira_issue_ids"))
                newly_resolved = classification == "NEWLY_RESOLVED"
                update = await self.executor.call(
                    phase="D1", step=f"D1.5.{idx}", tool="record_case_outcome",
                    arguments={
                        "case_id": case_id(case),
                        "outcome_status": "confirmed_issue" if newly_resolved else (_status_text(case) or "unreviewed"),
                        "human_confirmed": newly_resolved,
                        "jira": str(jira_ids[0]) if jira_ids else "",
                        "resolution": (
                            "Confirmed fix lineage linked by nightly workflow."
                            if newly_resolved else
                            "JIRA issue lineage attached; resolution remains unconfirmed."
                        ),
                    },
                )
                action = "UPDATED" if _resp(update) is not None else "UPDATE_FAILED"
            elif classification in {"NEWLY_RESOLVED", "FIX_LINEAGE_UPDATED"}:
                action = "WOULD_UPDATE"

            rows.append({
                "case_id": case_id(case), "profile": profile, "receiver": receiver,
                "date": event_date, "previous_status": _status_text(case),
                "logical_case_key": key, "new_classification": classification,
                "jira_ids": as_list((lineage_response or {}).get("jira_fix_ids")) + as_list((lineage_response or {}).get("jira_issue_ids")),
                "action_taken": action, "lineage": lineage_response, "details": details,
            })
        counts = Counter(row["new_classification"] for row in rows)
        payload = {
            "status": "OK" if search.get("status") == "OK" else "CANNOT_DETERMINE",
            "cases_returned": len(cases), "in_scope": len(in_scope), "out_of_scope": len(out_scope),
            "rows": rows, "classification_counts": dict(counts),
        }
        self.state.data["unresolved_resolution"] = {
            **{k: v for k, v in payload.items() if k != "rows"},
            "rows": [
                {k: row.get(k) for k in (
                    "case_id", "profile", "receiver", "date", "previous_status",
                    "logical_case_key", "new_classification", "jira_ids", "action_taken",
                )}
                for row in rows
            ],
        }
        return self.checkpoint("D1", payload)

    async def phase_2_discovery(self) -> dict[str, Any]:
        family_results: dict[str, Any] = {}
        for idx, (family, alert_names) in enumerate(ALERT_FAMILIES, start=1):
            result = await self.executor.call(
                phase="2", step=f"2.{idx}", tool="count_alerts",
                arguments={"alert_name": alert_names, "group_by": "alert_name"},
            )
            family_results[family] = _resp(result)
        anomalies = await self.executor.call(
            phase="2", step="2.8", tool="list_anomalies",
            arguments={"min_score": 90, "max_results": 20},
        )
        history_signals = load_history_signals(
            self.settings.output_dir, self.state.run_id, self.settings.history_runs
        )
        clusters = discover_clusters(
            family_results, _resp(anomalies), self.state.active_date_window,
            history_counts=history_signals["recurrence"],
            history_validation_passes=history_signals["validation_passes"],
            history_validation_failures=history_signals["validation_failures"],
            limit=self.settings.max_candidate_clusters,
        )
        payload = {
            "status": "OK" if clusters else "OK_QUIET",
            "family_results": family_results,
            "anomalies": _resp(anomalies),
            "history_signals": history_signals,
            "candidate_clusters": clusters,
        }
        self.state.data["candidate_clusters"] = clusters
        return self.checkpoint("2", payload)

    async def phase_2b_code_context(self) -> dict[str, Any]:
        """Enrich candidate_clusters with code-tool lookups (source symbol, git blame, repo clone).

        Entirely optional — degrades to no-op when code_tools_mcp is not configured.
        A TOOL_NOT_FOUND or transport error from any step is caught and noted; it never
        blocks the pipeline.
        """
        clusters: list[dict[str, Any]] = self.state.data.get("candidate_clusters", [])
        if not clusters:
            return self.checkpoint("2b", {"status": "SKIPPED_NO_CLUSTERS", "enriched": 0})

        # Check whether code_tools_mcp is even configured.
        code_url = self.settings.server_urls.get("code_tools_mcp", "")
        if not code_url:
            return self.checkpoint("2b", {
                "status": "SKIPPED_NOT_CONFIGURED",
                "note": "Set NIGHTLY_RCA_CODE_MCP_URL to enable source-code enrichment",
                "enriched": 0,
            })

        enriched = 0
        clone_attempts: list[dict[str, Any]] = []

        for idx, cluster in enumerate(clusters):
            patterns: list[str] = cluster.get("key_patterns", [])
            code_hits: list[dict[str, Any]] = []

            for pattern in patterns[:3]:  # cap per cluster
                symbol_hint = _ALERT_TO_SYMBOL.get(pattern.upper())
                if symbol_hint:
                    repo, symbol = symbol_hint
                    # Try find_symbol first
                    res = await self.executor.call(
                        phase="2b", step=f"2b.{idx+1}.sym",
                        tool="code_find_symbol",
                        arguments={"repository": repo, "symbol_name": symbol},
                        required=False,
                    )
                    if _code_available(res) and res.get("response"):
                        code_hits.append({
                            "type": "symbol_definition",
                            "repository": repo,
                            "symbol": symbol,
                            "result_summary": str(res["response"])[:400],
                        })
                    else:
                        # Fall back to regex search
                        res2 = await self.executor.call(
                            phase="2b", step=f"2b.{idx+1}.grep",
                            tool="code_search_regex",
                            arguments={"repository": repo, "pattern": pattern, "max_results": 5},
                            required=False,
                        )
                        if _code_available(res2) and res2.get("response"):
                            code_hits.append({
                                "type": "regex_search",
                                "repository": repo,
                                "pattern": pattern,
                                "result_summary": str(res2["response"])[:400],
                            })
                else:
                    # Unknown symbol — attempt a cross-repo regex search
                    # Try the most likely repos based on signal_families
                    families = cluster.get("signal_families", [])
                    search_repo = "ATV_Qt_UI" if "app_service" in families else "stbctrl"
                    res3 = await self.executor.call(
                        phase="2b", step=f"2b.{idx+1}.xgrep",
                        tool="code_search_regex",
                        arguments={"repository": search_repo, "pattern": pattern, "max_results": 5},
                        required=False,
                    )
                    if _code_available(res3) and res3.get("response"):
                        code_hits.append({
                            "type": "regex_search",
                            "repository": search_repo,
                            "pattern": pattern,
                            "result_summary": str(res3["response"])[:400],
                        })
                    elif res3.get("status") == "TOOL_NOT_FOUND":
                        # Repository not indexed — attempt git clone
                        clone_res = await self.executor.call(
                            phase="2b", step=f"2b.{idx+1}.clone",
                            tool="git_clone_repo",
                            arguments={"repository": search_repo},
                            required=False,
                        )
                        clone_attempts.append({
                            "repository": search_repo,
                            "status": clone_res.get("status"),
                            "error": clone_res.get("error"),
                        })

            if code_hits:
                cluster["code_context"] = code_hits
                enriched += 1

        payload = {
            "status": "OK" if enriched > 0 else "OK_QUIET",
            "enriched": enriched,
            "total_clusters": len(clusters),
            "clone_attempts": clone_attempts,
        }
        return self.checkpoint("2b", payload)

    async def phase_3_profile_matching(self) -> dict[str, Any]:
        catalog_call = await self.executor.call(
            phase="3", step="3.1", tool="get_issue_profile_catalog",
            arguments={"include_patterns": True},
        )
        catalog = normalize_catalog(_resp(catalog_call))
        clusters = self.state.data.get("candidate_clusters", [])
        matches = match_clusters_to_profiles(
            clusters, catalog, self.state.data.get("case_inventory_all", [])
        )
        match_by_id = {m["candidate_id"]: m for m in matches}
        for cluster in clusters:
            match = match_by_id.get(cluster["candidate_id"], {})
            cluster["known_profile_match"] = match.get("best_profile_match") or None
            cluster["new_profile_needed"] = bool(match.get("new_profile_required"))
        self.state.data["profile_catalog"] = catalog
        self.state.data["profile_matches"] = matches
        payload = {"status": "OK", "profile_count": len(catalog), "matches": matches}
        return self.checkpoint("3", payload)

    async def phase_4_profile_design(self) -> dict[str, Any]:
        clusters = {c["candidate_id"]: c for c in self.state.data.get("candidate_clusters", [])}
        proposed: list[dict[str, Any]] = []
        for match in self.state.data.get("profile_matches", []):
            if match.get("match_quality") == "NEW_PROFILE_NEEDED":
                proposed.append(draft_profile(clusters[match["candidate_id"]]))
        self.state.data["proposed_profiles"] = proposed
        payload = {"status": "OK", "proposed_profiles": proposed}
        return self.checkpoint("4", payload)

    async def phase_5_profile_validation(self) -> dict[str, Any]:
        """Validate profiles and turn external log waits into durable resumable work.

        Execution order (receipt reconciliation is fully decoupled from validation):
        1. Load all pending physical tracker rows.
        2. select_pending_investigations(batch_size=pending_batch_size)
        3. For each selected physical tracker:
             call _reconcile_upload_tracker exactly once
             store the result by physical tracker identity
             update attempted-row metadata
        4. Convert log-ready reconciliation results into validation targets.
        5. Combine those targets with fresh validation targets.
        6. Apply select_validation_targets(max_total=max_profiles_per_run)
        7. Perform at most max_profiles_per_run downstream validations.
        8. Atomically save the pending ledger.
        """
        clusters = {c["candidate_id"]: c for c in self.state.data.get("candidate_clusters", [])}
        proposed_by_candidate: dict[str, dict[str, Any]] = {}
        proposed = self.state.data.get("proposed_profiles", [])
        unmatched_ids = [m["candidate_id"] for m in self.state.data.get("profile_matches", []) if m.get("new_profile_required")]
        for cid, profile in zip(unmatched_ids, proposed):
            proposed_by_candidate[cid] = profile

        # ── Step 1: Load all pending physical tracker rows ────────────────────────
        _now_utc = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        _all_pending = self.state.data.get("pending_investigations", [])

        # ── Step 2: select_pending_investigations(batch_size=pending_batch_size) ──
        _selected_pending = select_pending_investigations(
            _all_pending,
            batch_size=self.settings.pending_batch_size,
            now=_now_utc,
        )
        _selected_tracker_ids: frozenset[str] = frozenset(
            str(row.get("tracker_id") or row.get("request_id") or "")
            for row in _selected_pending if row.get("tracker_id") or row.get("request_id")
        )
        _pending_selector_metrics = selector_metrics(
            _all_pending, _selected_pending, now=_now_utc,
            batch_size=self.settings.pending_batch_size,
        )
        log.info("pending_selector: %s", _pending_selector_metrics)
        self.state.data["pending_selector_metrics"] = _pending_selector_metrics

        # ── Step 3: Reconcile each selected physical tracker exactly once ─────────
        # Results cached by physical tracker ID; no reconciliation happens elsewhere.
        outstanding: dict[str, dict[str, Any]] = {
            str(row.get("tracker_id") or row.get("request_id")): dict(row)
            for row in _all_pending
            if row.get("tracker_id") or row.get("request_id")
        }
        _reconciliation_cache: dict[str, dict[str, Any]] = {}
        receipt_ready_targets: list[dict[str, Any]] = []
        receipt_validations: list[dict[str, Any]] = []

        for ridx, pending_row in enumerate(_selected_pending, start=1):
            tracker_id = str(pending_row.get("tracker_id") or pending_row.get("request_id") or "")
            primary_rx = str(pending_row.get("receiver_id") or "")
            pid = str(pending_row.get("profile_id") or pending_row.get("issue_profile") or "")
            cid = str(pending_row.get("candidate_id") or f"resume_{stable_hash([pid, primary_rx])[:10]}")

            receipt = await self._reconcile_upload_tracker(
                ridx, pending_row, step_prefix=f"5.P{ridx:02d}.resume_receipt",
            )
            receipt_ok = _receipt_ready(receipt)

            # Cache the reconciliation result by physical tracker identity
            if tracker_id:
                _reconciliation_cache[tracker_id] = receipt

            # Update metadata for attempted row
            was_error = not receipt_ok and not (receipt or {}).get("status") in (None, "WAITING_FOR_LOGS", "NOT_READY")
            update_metadata_after_attempt(
                pending_row,
                now=_now_utc,
                was_error=was_error,
                selector_generation=int(_pending_selector_metrics.get("pending_selected", 0)),
            )
            # Update in outstanding map
            if tracker_id:
                outstanding[tracker_id] = dict(pending_row)

            # ── Step 4 (partial): Convert log-ready results into validation targets
            if receipt_ok:
                coverage_call = await self.executor.call(
                    phase="5", step=f"5.P{ridx:02d}.resume_coverage", tool="verify_receiver_log_coverage",
                    arguments={"receiver_ids": primary_rx, "profile": pid, "max_dates_per_receiver": 3},
                )
                coverage_response = _resp(coverage_call)
                summary = coverage_summary(coverage_response)
                ready_rx = _coverage_ready_receiver(coverage_response, [primary_rx])

                if summary["required_logs_present"] and ready_rx:
                    # Fully ready — becomes a validation target (already validated)
                    outstanding.pop(tracker_id, None)
                    target = {
                        "candidate_id": cid,
                        "profile_id": pid,
                        "profile_source": "resumed_tracker",
                        "draft": None,
                        "resumed_tracker": pending_row,
                        "cluster": {
                            "candidate_id": cid,
                            "suspected_profile": str(pending_row.get("suspected_profile") or pid.removeprefix("auto_")),
                            "alert_name": str(pending_row.get("alert_name") or ""),
                            "key_patterns": [str(pending_row.get("alert_name") or "")],
                            "receivers_seen": [primary_rx],
                            "dates": [str(pending_row.get("event_date") or "")],
                        },
                    }
                    row = self._validation_row(
                        target, [primary_rx], ready_rx, None, coverage_response, summary,
                        "PROFILE_VALIDATION_PASS",
                        f"Persisted upload tracker {tracker_id} is ready; S3 coverage was revalidated.",
                        tracker_id, resumed=True,
                    )
                    receipt_validations.append(row)
                    receipt_ready_targets.append(target)
                else:
                    # Receipt ready but coverage not confirmed
                    pending_row.update({"workflow_status": "RECEIPT_OK_COVERAGE_PENDING", "last_checked_at": utc_now(), "profile_id": pid, "receiver_id": primary_rx})
                    if tracker_id:
                        outstanding[tracker_id] = dict(pending_row)
            else:
                # Not ready yet
                pending_row.update({"workflow_status": "WAITING_FOR_LOGS", "last_checked_at": utc_now()})
                if tracker_id:
                    outstanding[tracker_id] = dict(pending_row)

        # ── Step 5: Build fresh validation targets from anomaly matches ───────────
        fresh_validation_targets: list[dict[str, Any]] = []
        target_keys: set[tuple[str, str]] = set()
        target_index_by_key: dict[tuple[str, str], int] = {}
        for match in self.state.data.get("profile_matches", []):
            quality = match.get("match_quality")
            if quality not in {"MATCHES_EXISTING_PROFILE", "PARTIAL_MATCH", "NEW_PROFILE_NEEDED"}:
                continue
            cid = match["candidate_id"]
            draft = proposed_by_candidate.get(cid)
            pid = match.get("best_profile_match") or (draft or {}).get("profile_id")
            if not pid:
                continue
            target = {
                "candidate_id": cid,
                "profile_id": pid,
                "profile_source": "proposed" if draft else "catalog",
                "draft": draft,
                "cluster": clusters.get(cid, {}),
            }
            target_index = len(fresh_validation_targets)
            fresh_validation_targets.append(target)
            for rx in target["cluster"].get("receivers_seen", []):
                key = (str(pid), str(rx))
                target_keys.add(key)
                target_index_by_key.setdefault(key, target_index)

        # Merge remaining pending trackers (not yet reconciled this cycle)
        # into fresh targets if they overlap with anomaly targets.
        pending_rows = list(outstanding.values())
        self.store.save_pending_investigations(pending_rows)
        self.state.data["pending_investigations"] = pending_rows

        for tracker in pending_rows:
            pid = str(tracker.get("profile_id") or tracker.get("issue_profile") or "")
            rx = str(tracker.get("receiver_id") or "")
            if not (pid and rx):
                continue
            tracker_id_val = str(tracker.get("tracker_id") or tracker.get("request_id") or "")
            # Skip if already reconciled this cycle
            if tracker_id_val in _selected_tracker_ids:
                continue
            key = (pid, rx)
            existing_index = target_index_by_key.get(key)
            if existing_index is not None and not fresh_validation_targets[existing_index].get("resumed_tracker"):
                fresh_validation_targets[existing_index]["resumed_tracker"] = tracker
                fresh_validation_targets[existing_index]["profile_source"] = "catalog_or_proposed+resumed_tracker"
                continue
            cid = str(tracker.get("candidate_id") or f"resume_{stable_hash([pid, rx])[:10]}")
            fresh_validation_targets.append({
                "candidate_id": cid,
                "profile_id": pid,
                "profile_source": "resumed_tracker",
                "draft": None,
                "resumed_tracker": tracker,
                "cluster": {
                    "candidate_id": cid,
                    "suspected_profile": str(tracker.get("suspected_profile") or pid.removeprefix("auto_")),
                    "alert_name": str(tracker.get("alert_name") or ""),
                    "key_patterns": [str(tracker.get("alert_name") or "")],
                    "receivers_seen": [rx],
                    "dates": [str(tracker.get("event_date") or "")],
                },
            })
            target_keys.add(key)
            target_index_by_key.setdefault(key, len(fresh_validation_targets) - 1)

        # ── Step 6: Combine receipt-ready targets with fresh, then cap ────────────
        combined_targets = receipt_ready_targets + fresh_validation_targets
        validation_targets = select_validation_targets(
            combined_targets,
            max_total=self.settings.max_profiles_per_run,
        )

        # ── Step 7: Perform at most max_profiles_per_run downstream validations ──
        # Receipt-ready items are already fully validated; the downstream loop
        # only processes targets that need fresh investigation or cohort checks.
        # The reconciliation cache is consumed (never re-called).
        _receipt_ready_ids: frozenset[int] = frozenset(
            id(t) for t in receipt_ready_targets
        )
        validations: list[dict[str, Any]] = list(receipt_validations)
        actionable: list[dict[str, Any]] = [r for r in receipt_validations if r.get("classification") == "PROFILE_VALIDATION_PASS"]

        for idx, target in enumerate(validation_targets, start=1):
            # Skip targets already fully validated during receipt reconciliation
            if id(target) in _receipt_ready_ids:
                continue

            cluster = target["cluster"]
            cluster_receivers = [str(x) for x in cluster.get("receivers_seen", []) if x]
            alert_name = str(cluster.get("alert_name") or (cluster.get("key_patterns") or [""])[0] or "")
            resumed = target.get("resumed_tracker") or {}
            tracker_id = str(resumed.get("tracker_id") or "")
            plan_response: Any = None
            coverage_response: Any = None
            candidate_receivers: list[str] = list(dict.fromkeys(cluster_receivers))
            cohort_receiver = ""
            acquisition_request: dict[str, Any] = {}

            # If this target has a resumed tracker that was reconciled in Step 3,
            # consume the cached result — never call _reconcile_upload_tracker again.
            _tracker_id_val = str(resumed.get("tracker_id") or resumed.get("request_id") or "")
            _cached_receipt = _reconciliation_cache.get(_tracker_id_val) if _tracker_id_val else None

            if resumed and (_cached_receipt is not None or _tracker_id_val in _selected_tracker_ids):
                # Already reconciled in Step 3; use cached receipt result.
                # Items with receipt_ok+coverage pass are already in receipt_ready_targets
                # and skipped above. Items here had receipt_ok but coverage NOT confirmed,
                # or receipt was not ready. Use the cached result.
                primary_rx = str(resumed.get("receiver_id") or (candidate_receivers[0] if candidate_receivers else ""))
                cached = _cached_receipt or {}
                receipt_ok = _receipt_ready(cached)
                if receipt_ok:
                    # Receipt was ready but coverage failed in Step 4 — try cohort
                    coverage_call = await self.executor.call(
                        phase="5", step=f"5.V{idx:02d}.resume_coverage", tool="verify_receiver_log_coverage",
                        arguments={"receiver_ids": primary_rx, "profile": target["profile_id"], "max_dates_per_receiver": 3},
                    )
                    coverage_response = _resp(coverage_call)
                    summary = coverage_summary(coverage_response)
                    cohort_receiver = await self._find_cohort_receiver(
                        idx, primary_rx, alert_name, target["profile_id"], candidate_receivers,
                    )
                    if cohort_receiver:
                        cohort_cov = await self.executor.call(
                            phase="5", step=f"5.V{idx:02d}.resume_cohort", tool="verify_receiver_log_coverage",
                            arguments={"receiver_ids": cohort_receiver, "profile": target["profile_id"], "max_dates_per_receiver": 3},
                        )
                        coverage_response = _resp(cohort_cov)
                        summary = coverage_summary(coverage_response)
                        classification = "PROFILE_VALIDATION_PASS"
                        reason = f"Primary receiver {primary_rx} is still pending; cohort peer {cohort_receiver} has verified logs. Tracker {tracker_id} remains durable."
                        candidate_receivers = [cohort_receiver] + [x for x in candidate_receivers if x != cohort_receiver]
                    else:
                        classification = "AWAITING_LOGS"
                        reason = f"Persisted tracker {tracker_id} for receiver {primary_rx} is still waiting for logs; it will be reconciled on the next cron run."
                else:
                    # Receipt not ready
                    primary_rx = str(resumed.get("receiver_id") or (candidate_receivers[0] if candidate_receivers else ""))
                    summary = {"required_logs_present": False, "covered_receivers": 0}
                    classification = "AWAITING_LOGS"
                    reason = f"Persisted tracker {tracker_id} for receiver {primary_rx} is still waiting for logs; it will be reconciled on the next cron run."
                row = self._validation_row(target, candidate_receivers, cohort_receiver or primary_rx, plan_response, coverage_response, summary, classification, reason, tracker_id, resumed=True)
                validations.append(row)
                if classification == "PROFILE_VALIDATION_PASS":
                    actionable.append(row)
                continue

            # Alert-only anomaly rows may initially lack receiver samples.
            if not candidate_receivers and alert_name:
                candidate_receivers = await self._query_alert_receivers(idx, alert_name, exclude=set())
            if not candidate_receivers:
                row = self._validation_row(
                    target, [], "", None, None,
                    {"required_logs_present": False, "covered_receivers": 0},
                    "PROFILE_NEEDS_MORE_DATA",
                    "No receiver identity was available from anomaly samples or RTR receiver cohorts; the candidate remains queued for a later run.",
                    "",
                )
                validations.append(row)
                continue

            plan = await self.executor.call(
                phase="5", step=f"5.V{idx:02d}.a", tool="plan_profile_investigation",
                arguments={
                    "profile": target["profile_id"], "receiver_ids": ",".join(candidate_receivers),
                    "prefer": "balanced", "max_dates_per_receiver": 3, "max_candidates": 10,
                },
            )
            plan_response = _resp(plan)
            candidate_receivers = extract_candidate_receivers(plan_response) or candidate_receivers
            coverage = await self.executor.call(
                phase="5", step=f"5.V{idx:02d}.b", tool="verify_receiver_log_coverage",
                arguments={"receiver_ids": ",".join(candidate_receivers[:10]), "profile": target["profile_id"], "max_dates_per_receiver": 3},
            )
            coverage_response = _resp(coverage)
            summary = coverage_summary(coverage_response)
            ready_rx = _coverage_ready_receiver(coverage_response, candidate_receivers)

            if plan.get("status") != "OK":
                classification = "PROFILE_VALIDATION_FAIL"
                reason = plan.get("error") or "Investigation planning failed."
                best_receiver = candidate_receivers[0]
            elif summary["required_logs_present"] and ready_rx:
                classification = "PROFILE_VALIDATION_PASS"
                reason = "A candidate receiver with all required S3 log coverage was observed."
                best_receiver = ready_rx
            else:
                primary_rx = candidate_receivers[0]
                request = await self._request_grasshopper_upload(
                    idx, primary_rx, target["profile_id"], alert_name, target.get("candidate_id", ""),
                )
                acquisition_request = request
                tracker_id = str(request.get("tracker_id") or "")
                logs_arrived = False
                if request.get("submitted") and request.get("tracker_write_succeeded") is not False:
                    logs_arrived = await self._wait_for_log_arrival(idx, primary_rx, tracker_id, target["profile_id"])
                if logs_arrived:
                    recheck = await self.executor.call(
                        phase="5", step=f"5.V{idx:02d}.recheck", tool="verify_receiver_log_coverage",
                        arguments={"receiver_ids": primary_rx, "profile": target["profile_id"], "max_dates_per_receiver": 3},
                    )
                    coverage_response = _resp(recheck)
                    summary = coverage_summary(coverage_response)
                    ready_rx = _coverage_ready_receiver(coverage_response, [primary_rx])
                if logs_arrived and summary["required_logs_present"] and ready_rx:
                    classification = "PROFILE_VALIDATION_PASS"
                    reason = f"Receiver {primary_rx} logs arrived and coverage was verified during immediate reconciliation."
                    best_receiver = ready_rx
                else:
                    cohort_receiver = await self._find_cohort_receiver(
                        idx, primary_rx, alert_name, target["profile_id"], candidate_receivers,
                    )
                    if cohort_receiver:
                        cohort_cov = await self.executor.call(
                            phase="5", step=f"5.V{idx:02d}.c", tool="verify_receiver_log_coverage",
                            arguments={"receiver_ids": cohort_receiver, "profile": target["profile_id"], "max_dates_per_receiver": 3},
                        )
                        coverage_response = _resp(cohort_cov)
                        summary = coverage_summary(coverage_response)
                        candidate_receivers = [cohort_receiver] + [r for r in candidate_receivers if r != cohort_receiver]
                        classification = "PROFILE_VALIDATION_PASS"
                        reason = f"Primary receiver {primary_rx} lacks logs; cohort peer {cohort_receiver} has verified coverage."
                        if request.get("submitted"):
                            reason += f" Primary upload remains tracked as {tracker_id}."
                        best_receiver = cohort_receiver
                    elif request.get("submitted"):
                        classification = "AWAITING_LOGS"
                        if request.get("tracker_write_succeeded") is False:
                            reason = (
                                f"Receiver {primary_rx} upload was accepted, but the external tracker write failed. "
                                "A local reconciliation record was preserved; no automatic resubmission is allowed."
                            )
                        else:
                            reason = f"Receiver {primary_rx} upload was accepted and durably queued as {tracker_id}; the next cron run will reconcile receipt and retry cohorts."
                        best_receiver = primary_rx
                        pending = dict(request.get("pending") or {})
                        pending.setdefault("tracker_id", tracker_id)
                        pending.setdefault("request_id", request.get("request_id"))
                        pending.update({
                            "profile_id": target["profile_id"], "issue_profile": target["profile_id"],
                            "receiver_id": primary_rx, "candidate_id": target.get("candidate_id", ""),
                            "alert_name": alert_name, "workflow_status": "WAITING_FOR_LOGS",
                            "origin_run_id": self.state.run_id, "last_checked_at": utc_now(),
                            # Preserve the exact non-secret plan identity locally
                            # even when the deployed tracker schema cannot retain
                            # every modern field yet. Never synthesize an external ID.
                            "grasshopper_profile": request.get("grasshopper_profile", ""),
                            "selected_file_ids": list(request.get("selected_file_ids") or []),
                            "selected_file_count": request.get("file_count"),
                            "plan_fingerprint": request.get("plan_fingerprint", ""),
                            "local_correlation_id": request.get("local_correlation_id", ""),
                            "grasshopper_request_id": request.get("grasshopper_request_id", ""),
                            "identifier_provenance": request.get("identifier_provenance", ""),
                        })
                        pending_key = tracker_id or str(request.get("local_correlation_id") or request.get("request_id") or "")
                        if pending_key:
                            outstanding[pending_key] = pending
                    elif request.get("status") == "UPLOAD_PLAN_ONLY":
                        classification = "UPLOAD_PLAN_ONLY"
                        reason = f"Receiver {primary_rx} lacks logs. Dry-run verified the upload plan; no upload or tracker write was claimed."
                        best_receiver = primary_rx
                    else:
                        classification = "COVERAGE_UNOBTAINABLE"
                        reason = f"Receiver {primary_rx} lacks logs, the upload request was not accepted, and no cohort peer with verified coverage was found."
                        best_receiver = primary_rx

            row = self._validation_row(target, candidate_receivers, best_receiver, plan_response, coverage_response, summary, classification, reason, tracker_id)
            if acquisition_request:
                row["log_acquisition"] = acquisition_request
                row["log_acquisition_status"] = acquisition_request.get("status")
            validations.append(row)
            if classification == "PROFILE_VALIDATION_PASS":
                actionable.append(row)

        # ── Step 8: Atomically save the pending ledger ────────────────────────────
        # Re-save in case the validation loop added new pending entries.
        if outstanding:
            final_pending_rows = list(outstanding.values())
            self.store.save_pending_investigations(final_pending_rows)
            self.state.data["pending_investigations"] = final_pending_rows

        compact_validations: list[dict[str, Any]] = []
        for row in validations:
            compact = {k: row.get(k) for k in (
                "candidate_id", "profile_id", "profile_source", "candidate_receivers",
                "best_receiver", "required_logs_present", "covered_receivers",
                "classification", "reason", "confidence", "tracker_id", "resumed",
                "log_acquisition_status", "local_correlation_id",
                "grasshopper_request_id", "identifier_provenance",
                "origin_run_id", "historical_carry_forward", "currentness",
                "submission_attempted", "submission_accepted",
            )}
            compact["suspected_profile"] = row.get("cluster", {}).get("suspected_profile")
            if row.get("draft"):
                compact["draft"] = row.get("draft")
            compact_validations.append(compact)
        compact_actionable = [row for row in compact_validations if row.get("classification") == "PROFILE_VALIDATION_PASS"]
        self.state.data["profile_validations"] = compact_validations
        self.state.data["actionable_profiles"] = compact_actionable
        payload = {
            "status": "OK", "validations": validations,
            "actionable_profiles": compact_actionable,
            "pending_investigations": self.state.data.get("pending_investigations", []),
        }
        return self.checkpoint("5", payload)


    @staticmethod
    def _validation_row(
        target: dict[str, Any], candidate_receivers: list[str], best_receiver: str,
        plan: Any, coverage: Any, summary: dict[str, Any], classification: str,
        reason: str, tracker_id: str, resumed: bool = False,
    ) -> dict[str, Any]:
        return {
            **target,
            "candidate_receivers": candidate_receivers,
            "best_receiver": best_receiver,
            "plan": plan,
            "coverage": coverage,
            "required_logs_present": bool(summary.get("required_logs_present")),
            "covered_receivers": int(summary.get("covered_receivers") or 0),
            "classification": classification,
            "reason": reason,
            "tracker_id": tracker_id,
            "resumed": resumed,
            "confidence": "high" if classification == "PROFILE_VALIDATION_PASS" else "medium",
        }

    async def _request_grasshopper_upload(
        self, idx: int, receiver_id: str, profile_id: str, alert_name: str, candidate_id: str,
    ) -> dict[str, Any]:
        """Plan, safety-check, submit, and durably track one bounded upload request.

        Uses the typed grasshopper_contract adapter for nested response parsing,
        correct file-count extraction from plan.selected_file_count, and
        proven identifier provenance separation.
        """
        safe_cid = re.sub(r"[^a-zA-Z0-9_-]", "-", candidate_id)[:32] or "candidate"
        safe_run = re.sub(r"[^a-zA-Z0-9_-]", "-", self.state.run_id)[:40]
        local_correlation_id = f"nightly-rca-phase5-{safe_cid}-{receiver_id}-{safe_run}"
        profile_metadata = self._grasshopper_profile_metadata(profile_id)
        source_availability = (
            self.state.data.get("source_inventory", {})
            .get("source_availability", {})
        )
        availability = bool(source_availability.get("list_upload_trackers")) and bool(
            source_availability.get("local_pending_ledger")
        )
        duplicate = evaluate_duplicate_preflight(
            self.state.data.get("pending_investigations", []),
            inventory_available=availability,
            receiver_id=receiver_id,
            requested_log_types=profile_metadata.requested_log_types,
            grasshopper_profile=profile_metadata.grasshopper_profile,
        )
        if duplicate["status"] != "PASS":
            duplicate_state = (
                AcquisitionState.UPLOAD_BLOCKED_DUPLICATE_REQUEST
                if duplicate["status"] == "BLOCK"
                else AcquisitionState.UPLOAD_BLOCKED_DUPLICATE_PREFLIGHT_UNAVAILABLE
            )
            return {
                "request_id": local_correlation_id,
                "local_correlation_id": local_correlation_id,
                "receiver_id": receiver_id,
                "profile_id": profile_id,
                "grasshopper_profile": profile_metadata.grasshopper_profile,
                "status": duplicate_state.value,
                "submitted": False,
                "submission_attempted": False,
                "submission_accepted": False,
                "duplicate_preflight": duplicate,
                "reason": ACQUISITION_STATE_META[duplicate_state]["operator_message"],
                "origin_run_id": self.state.run_id,
            }
        plan_arguments = build_grasshopper_arguments(
            profile_metadata,
            receiver_id,
            max_upload_files=self.settings.max_upload_files,
        )

        # ── Plan ──────────────────────────────────────────────────────────────
        raw_plan = await self.executor.call(
            phase="5", step=f"5.{idx}.upload_plan", tool="grasshopper_plan_profile_upload",
            arguments=plan_arguments,
        )
        plan = parse_plan_response(raw_plan, profile_metadata=profile_metadata)
        state = determine_acquisition_state(plan, upload=None, dry_run=True,
                                            max_upload_files=self.settings.max_upload_files)

        base = {
            "request_id": local_correlation_id,
            "receiver_id": receiver_id,
            "profile_id": profile_id,
            "grasshopper_profile": profile_metadata.grasshopper_profile,
            "profile_metadata": profile_metadata.to_public_dict(),
            "plan": plan.to_public_dict(),
            "file_count": plan.selected_file_count,
            "selected_file_ids": list(plan.selected_file_ids),
            "requested_profile": plan.requested_profile,
            "resolved_profile": plan.resolved_profile,
            "profile_alias_used": plan.profile_alias_used,
            "profile_identity_matches": plan.profile_identity_matches,
            "plan_fingerprint": "sha256:" + stable_hash({
                "receiver_id": receiver_id,
                "resolved_profile": plan.resolved_profile or profile_metadata.grasshopper_profile,
                "selected_file_ids": list(plan.selected_file_ids),
                "selected_file_count": plan.selected_file_count,
                "upload_mode": profile_metadata.upload_mode,
                "max_files_per_type": plan_arguments.get("max_files_per_type"),
                "max_total_files": plan_arguments.get("max_total_files"),
            }),
            "local_correlation_id": local_correlation_id,
            "origin_run_id": self.state.run_id,
            "historical_carry_forward": False,
            "currentness": "CURRENT_RUN",
            "submission_attempted": False,
            "submission_accepted": False,
        }

        if state == AcquisitionState.UPLOAD_PLAN_FAILED:
            mismatch_reason = (
                "Grasshopper resolved a profile that does not match the requested catalog profile"
                if plan.profile_identity_matches is False else "upload plan failed"
            )
            return {**base, "status": state.value, "submitted": False,
                    "reason": plan.executor_error or mismatch_reason}

        if state == AcquisitionState.UPLOAD_BLOCKED_UNKNOWN_BATCH_SIZE:
            return {**base, "status": state.value, "submitted": False,
                    "reason": "upload plan did not expose a file count; the safety cap cannot be verified"}

        if state == AcquisitionState.UPLOAD_BLOCKED_BATCH_LIMIT:
            return {**base, "status": state.value, "submitted": False,
                    "reason": f"planned {plan.selected_file_count} files exceeds cap {self.settings.max_upload_files}"}

        if state == AcquisitionState.UPLOAD_SKIPPED_NO_FILES:
            return {**base, "status": state.value, "submitted": False,
                    "reason": "Grasshopper found no source files for this receiver/profile"}

        # ── Upload ────────────────────────────────────────────────────────────
        base["submission_attempted"] = bool(self.settings.commit)
        raw_upload = await self.executor.call(
            phase="5", step=f"5.{idx}.upload", tool="grasshopper_upload_profile_logs",
            arguments=build_grasshopper_arguments(
                profile_metadata,
                receiver_id,
                max_upload_files=self.settings.max_upload_files,
                dry_run=not self.settings.commit,
            ),
        )
        upload = parse_upload_response(raw_upload)

        if not self.settings.commit:
            return {**base, "status": AcquisitionState.UPLOAD_PLAN_ONLY.value,
                    "submitted": False, "upload": upload.to_public_dict()}

        # ── Determine final acquisition state ─────────────────────────────────
        final_state = determine_acquisition_state(
            plan, upload=upload, dry_run=False,
            max_upload_files=self.settings.max_upload_files,
        )
        ids = build_identifier_set(
            local_correlation_id=local_correlation_id,
            upload_result=upload,
            origin_run_id=self.state.run_id,
        )

        if final_state in (
            AcquisitionState.UPLOAD_REJECTED,
            AcquisitionState.UPLOAD_PROTOCOL_ERROR,
            AcquisitionState.UPLOAD_ACCEPTANCE_UNKNOWN,
        ):
            return {
                **base,
                "status": final_state.value,
                "submitted": False,
                "upload": upload.to_public_dict(),
                "grasshopper_request_id": ids.grasshopper_request_id,
                "submission_accepted": False,
                "request_created": upload.request_created,
                "http_status": upload.http_status,
                "upstream_code": upload.upstream_code,
                "automatic_retry": upload.automatic_retry,
                "retry_policy": upload.retry_policy,
                "reason": ACQUISITION_STATE_META[final_state]["operator_message"],
            }

        # ── Tracker registration ──────────────────────────────────────────────
        tracker = await self.executor.call(
            phase="5", step=f"5.{idx}.tracker", tool="record_upload_tracker",
            arguments={
                "receiver_id": receiver_id,
                "request_id": local_correlation_id,
                "grasshopper_request_id": ids.grasshopper_request_id,
                "issue_profile": profile_id,
                "grasshopper_profile": profile_metadata.grasshopper_profile,
                "requested_log_types": ",".join(profile_metadata.requested_log_types),
                "selected_file_count": plan.selected_file_count,
                "status": "upload_requested_pending_receipt",
                "notes": f"Nightly RCA phase 5 auto-request: alert={alert_name} cluster={safe_cid} coverage=no_s3_data",
                "origin": "nightly_rca_v6",
                "origin_run_id": self.state.run_id,
                "candidate_id": candidate_id,
                "alert_name": alert_name,
                "workflow_status": "WAITING_FOR_LOGS",
            },
        )
        tracker_response = _resp(tracker)
        tracker_write_succeeded = bool(
            tracker.get("status") == "OK"
            and isinstance(tracker_response, dict)
            and (
                _write_performed(tracker_response)
                or bool(extract_identifier(tracker_response, "tracker_id"))
            )
        )
        tracker_id = extract_identifier(tracker_response, "tracker_id") if tracker_write_succeeded else ""
        ids = build_identifier_set(
            local_correlation_id=local_correlation_id,
            upload_result=upload,
            tracker_id=tracker_id,
            origin_run_id=self.state.run_id,
        )

        if tracker_write_succeeded:
            pending = dict(tracker_response)
            status = AcquisitionState.UPLOAD_SUBMITTED_TRACKED.value
        else:
            pending = {
                "tracker_id": "",
                "request_id": local_correlation_id,
                "grasshopper_request_id": ids.grasshopper_request_id,
                "identifier_provenance": ids.identifier_provenance,
                "workflow_status": "TRACKER_WRITE_FAILED_RECONCILIATION_REQUIRED",
                "receipt_status": "UNKNOWN",
                "origin_run_id": self.state.run_id,
                "historical_carry_forward": False,
                "currentness": "CURRENT_RUN",
            }
            status = AcquisitionState.UPLOAD_ACCEPTED_TRACKER_WRITE_FAILED.value
        return {
            **base, "status": status, "submitted": True,
            "submission_attempted": True,
            "submission_accepted": True,
            "tracker_id": ids.tracker_id,
            "grasshopper_request_id": ids.grasshopper_request_id,
            "local_correlation_id": ids.local_correlation_id,
            "identifier_provenance": ids.identifier_provenance,
            "tracker_write_succeeded": tracker_write_succeeded,
            "upload": upload.to_public_dict(), "tracker": {
                "tracker_id": ids.tracker_id,
                "write_performed": bool(tracker_write_succeeded),
            },
            "pending": pending,
        }

    async def _reconcile_upload_tracker(self, idx: int, tracker: dict[str, Any], *, step_prefix: str = "") -> dict[str, Any]:
        tracker_id = str(tracker.get("tracker_id") or "")
        args: dict[str, Any] = {"tracker_id": tracker_id} if tracker_id else {
            "receiver_id": str(tracker.get("receiver_id") or ""),
            "request_id": str(tracker.get("request_id") or ""),
            "issue_profile": str(tracker.get("profile_id") or tracker.get("issue_profile") or ""),
        }
        args.update({"checked_by": "nightly_rca_v6", "dry_run": not self.settings.commit})
        step_id = step_prefix or f"5.{idx}.resume_receipt"
        result = await self.executor.call(
            phase="5", step=step_id, tool="update_upload_tracker_from_s3", arguments=args,
        )
        return _resp(result) or {}

    async def _wait_for_log_arrival(
        self, idx: int, receiver_id: str, tracker_id: str, profile_id: str,
    ) -> bool:
        """Perform bounded receipt checks; first check is immediate, later waits are opt-in."""
        import asyncio

        max_polls = max(1, int(self.settings.log_poll_max_attempts))
        interval = max(0, int(self.settings.log_poll_interval_seconds))
        for poll_num in range(1, max_polls + 1):
            if poll_num > 1 and interval:
                await asyncio.sleep(interval)
            result = await self.executor.call(
                phase="5", step=f"5.{idx}.poll_{poll_num}", tool="update_upload_tracker_from_s3",
                arguments={"tracker_id": tracker_id, "receiver_id": receiver_id,
                           "checked_by": "nightly_rca_v6", "dry_run": False},
            )
            resp = _resp(result)
            if _receipt_check_error(resp):
                return False
            if _receipt_ready(resp):
                return True
        return False

    async def _query_alert_receivers(self, idx: int, alert_name: str, exclude: set[str]) -> list[str]:
        if not alert_name:
            return []
        result = await self.executor.call(
            phase="5", step=f"5.{idx}.cohort_query", tool="count_alerts",
            arguments={"alert_name": alert_name, "group_by": "receiver", "as_csv": False},
        )
        response = _resp(result)
        rows = rows_from(response, "buckets", "items", "results", "receivers")
        if not rows and isinstance(response, dict) and isinstance(response.get("counts"), dict):
            rows = [{"key": key, "count": count} for key, count in response["counts"].items()]
        peers: list[str] = []
        for row in rows:
            rid = str(row.get("key") or row.get("receiver_id") or row.get("rxid") or "")
            if rid and rid not in exclude and rid not in peers:
                peers.append(rid)
            if len(peers) >= self.settings.log_cohort_max_peers:
                break
        return peers

    async def _find_cohort_receiver(
        self, idx: int, primary_rx: str, alert_name: str, profile_id: str,
        cluster_receivers: list[str],
    ) -> str:
        """Use cluster alternates first, then RTR peers, and verify each bounded batch."""
        peers = [rx for rx in cluster_receivers if rx and rx != primary_rx]
        discovered = await self._query_alert_receivers(idx, alert_name, exclude={primary_rx})
        peers.extend(rx for rx in discovered if rx not in peers)
        peers = peers[: self.settings.log_cohort_max_peers]
        batch_size = max(1, self.settings.log_cohort_batch_size)
        for batch_start in range(0, len(peers), batch_size):
            batch = peers[batch_start: batch_start + batch_size]
            coverage = await self.executor.call(
                phase="5", step=f"5.{idx}.cohort_cov.{batch_start}", tool="verify_receiver_log_coverage",
                arguments={"receiver_ids": ",".join(batch), "profile": profile_id, "max_dates_per_receiver": 3},
            )
            ready = _coverage_ready_receiver(_resp(coverage), batch)
            if ready:
                return ready
        return ""

    async def phase_6_data_collection(self) -> dict[str, Any]:
        uploads: list[dict[str, Any]] = []
        fallback_date = self.state.active_date_window[0] if self.state.active_date_window else date.today().isoformat()
        for idx, profile in enumerate(self.state.data.get("actionable_profiles", []), start=1):
            receiver = profile.get("best_receiver", "")
            if not receiver:
                uploads.append({"profile_id": profile["profile_id"], "status": "BLOCKED_NO_RECEIVER"})
                continue
            # Phase 5 already proved these logs exist in S3. Do not submit a
            # duplicate Grasshopper upload merely because the profile advanced.
            if profile.get("required_logs_present") is True:
                uploads.append({
                    "profile_id": profile["profile_id"], "receiver_id": receiver,
                    "date": fallback_date, "file_count": 0,
                    "status": "ALREADY_AVAILABLE", "receipt_verified": True,
                })
                continue
            profile_metadata = self._grasshopper_profile_metadata(profile["profile_id"])
            source_availability = (
                self.state.data.get("source_inventory", {})
                .get("source_availability", {})
            )
            duplicate = evaluate_duplicate_preflight(
                self.state.data.get("pending_investigations", []),
                inventory_available=(
                    bool(source_availability.get("list_upload_trackers"))
                    and bool(source_availability.get("local_pending_ledger"))
                ),
                receiver_id=receiver,
                requested_log_types=profile_metadata.requested_log_types,
                grasshopper_profile=profile_metadata.grasshopper_profile,
            )
            if duplicate["status"] != "PASS":
                blocked_state = (
                    AcquisitionState.UPLOAD_BLOCKED_DUPLICATE_REQUEST
                    if duplicate["status"] == "BLOCK"
                    else AcquisitionState.UPLOAD_BLOCKED_DUPLICATE_PREFLIGHT_UNAVAILABLE
                )
                uploads.append({
                    "profile_id": profile["profile_id"],
                    "receiver_id": receiver,
                    "date": fallback_date,
                    "file_count": None,
                    "status": blocked_state.value,
                    "reason": ACQUISITION_STATE_META[blocked_state]["operator_message"],
                    "duplicate_preflight": duplicate,
                    "submission_attempted": False,
                    "submission_accepted": False,
                    "current_s3_coverage": "UNKNOWN",
                })
                continue
            plan = await self.executor.call(
                phase="6", step=f"6.{idx}.1", tool="grasshopper_plan_profile_upload",
                arguments=build_grasshopper_arguments(
                    profile_metadata,
                    receiver,
                    max_upload_files=self.settings.max_upload_files,
                ),
            )
            parsed_plan = parse_plan_response(plan, profile_metadata=profile_metadata)
            plan_response = parsed_plan.to_public_dict()
            count = parsed_plan.selected_file_count
            event_date = _first_date(plan_response, fallback_date)
            row: dict[str, Any] = {
                "profile_id": profile["profile_id"], "receiver_id": receiver,
                "date": event_date, "plan": plan_response, "file_count": count,
                "grasshopper_profile": profile_metadata.grasshopper_profile,
                "profile_metadata": profile_metadata.to_public_dict(),
                "submission_attempted": False,
                "submission_accepted": False,
                "origin_run_id": self.state.run_id,
                "historical_carry_forward": False,
                "currentness": "CURRENT_RUN",
                "current_s3_coverage": "UNKNOWN",
                "status": "PLANNED",
            }
            plan_state = determine_acquisition_state(
                parsed_plan,
                upload=None,
                dry_run=True,
                max_upload_files=self.settings.max_upload_files,
            )
            if plan_state == AcquisitionState.UPLOAD_PLAN_FAILED:
                row["status"] = plan_state.value
                row["reason"] = parsed_plan.executor_error or "Grasshopper plan did not return success"
                uploads.append(row)
                continue
            if plan_state == AcquisitionState.UPLOAD_BLOCKED_UNKNOWN_BATCH_SIZE:
                row["status"] = plan_state.value
                row["reason"] = "upload plan did not expose a file count; the ≤50-file safety cap cannot be verified"
                uploads.append(row)
                continue
            if plan_state == AcquisitionState.UPLOAD_BLOCKED_BATCH_LIMIT:
                row["status"] = plan_state.value
                row["reason"] = f"planned {count} files exceeds cap {self.settings.max_upload_files}"
                uploads.append(row)
                continue
            if plan_state == AcquisitionState.UPLOAD_SKIPPED_NO_FILES:
                row["status"] = plan_state.value
                row["reason"] = "Grasshopper selected zero matching files; no submission occurred and current S3 coverage remains unknown"
                uploads.append(row)
                continue
            upload = await self.executor.call(
                phase="6", step=f"6.{idx}.2", tool="grasshopper_upload_profile_logs",
                arguments=build_grasshopper_arguments(
                    profile_metadata,
                    receiver,
                    max_upload_files=self.settings.max_upload_files,
                    dry_run=not self.settings.commit,
                ),
            )
            parsed_upload = parse_upload_response(upload)
            upload_response = parsed_upload.to_public_dict()
            row["upload"] = upload_response
            if not self.settings.commit:
                row["status"] = AcquisitionState.UPLOAD_PLAN_ONLY.value
                uploads.append(row)
                continue
            row["submission_attempted"] = True
            final_state = determine_acquisition_state(
                parsed_plan,
                parsed_upload,
                dry_run=False,
                max_upload_files=self.settings.max_upload_files,
            )
            row["status"] = final_state.value
            row["grasshopper_request_id"] = parsed_upload.inner_request_id
            if final_state != AcquisitionState.UPLOAD_ACCEPTED:
                row["reason"] = ACQUISITION_STATE_META[final_state]["operator_message"]
                uploads.append(row)
                continue
            row["submission_accepted"] = True
            receipt = await self.executor.call(
                phase="6", step=f"6.{idx}.3", tool="verify_profile_upload_receipt",
                arguments={"receiver_id": receiver, "profile": profile_metadata.grasshopper_profile, "date": event_date},
            )
            row["receipt"] = _resp(receipt)
            row["receipt_verified"] = _call_pass(_resp(receipt))
            row["status"] = "UPLOADED" if row["receipt_verified"] else "RECEIPT_UNVERIFIED"
            uploads.append(row)
        compact_uploads = [
            {k: row.get(k) for k in (
                "profile_id", "receiver_id", "date", "file_count", "status",
                "reason", "receipt_verified", "grasshopper_profile",
                "grasshopper_request_id", "submission_attempted", "submission_accepted",
                "origin_run_id", "historical_carry_forward", "currentness",
                "current_s3_coverage",
            )}
            for row in uploads
        ]
        self.state.data["uploads"] = compact_uploads
        return self.checkpoint("6", {"status": "OK", "uploads": uploads})

    async def phase_7_registration(self) -> dict[str, Any]:
        catalog_call = await self.executor.call(phase="7", step="7.1", tool="get_issue_profile_catalog", arguments={})
        current_catalog = normalize_catalog(_resp(catalog_call)) or self.state.data.get("profile_catalog", {})
        uploads_by_profile = {row.get("profile_id"): row for row in self.state.data.get("uploads", [])}
        rows: list[dict[str, Any]] = []
        for idx, validation in enumerate(self.state.data.get("actionable_profiles", []), start=1):
            pid = validation["profile_id"]
            draft = validation.get("draft")
            if pid in current_catalog:
                rows.append({"profile_id": pid, "status": "ALREADY_REGISTERED"})
                continue
            if not draft:
                rows.append({"profile_id": pid, "status": "CATALOG_PROFILE_MISSING_FROM_REFRESH"})
                continue
            upload_row = uploads_by_profile.get(pid, {})
            data_ready = upload_row.get("receipt_verified") is True or upload_row.get("status") == "ALREADY_AVAILABLE"
            if self.settings.commit and not data_ready:
                rows.append({
                    "profile_id": pid, "status": "REGISTRATION_BLOCKED_NO_VERIFIED_UPLOAD",
                    "upload_status": upload_row.get("status", "MISSING"),
                })
                continue
            if not self.settings.commit:
                rows.append({
                    "profile_id": pid,
                    "status": "WOULD_REGISTER" if upload_row.get("status") == "DRY_RUN_PLANNED" else "WOULD_NOT_REGISTER_DATA_BLOCKED",
                    "draft": draft, "upload_status": upload_row.get("status", "MISSING"),
                })
                continue
            registration = await self.executor.call(
                phase="7", step=f"7.2.{idx}", tool="register_issue_profile",
                arguments={
                    "issue_profile": pid,
                    "description": draft["short_description"],
                    "grasshopper_profile": pid,
                    "upload_mode": "balanced",
                    "core_log_types": draft["required_log_types"]["core"],
                },
            )
            registration_response = _resp(registration)
            registered = registration.get("status") == "OK" and (
                _write_performed(registration_response)
                or extract_identifier(registration_response, "profile_id", "issue_profile") == pid
            )
            rows.append({
                "profile_id": pid,
                "status": "REGISTERED" if registered else "REGISTRATION_FAILED",
                "response": registration_response, "error": registration.get("error"),
            })
        self.state.data["registrations"] = rows
        return self.checkpoint("7", {"status": "OK", "registrations": rows})

    async def phase_8_cases(self) -> dict[str, Any]:
        search = await self.executor.call(
            phase="8", step="8.1", tool="search_investigation_cases",
            arguments={"outcome_status": "unreviewed", "limit": 50},
        )
        existing_cases = normalize_cases(_resp(search))
        existing_by_key = {
            logical_case_key(case_profile(c), case_receiver(c), case_date(c)): c
            for c in existing_cases if case_profile(c) and case_receiver(c) and case_date(c)
        }
        uploads_by_profile = {u.get("profile_id"): u for u in self.state.data.get("uploads", [])}
        registrations_by_profile = {r.get("profile_id"): r for r in self.state.data.get("registrations", [])}
        rows: list[dict[str, Any]] = []
        fallback_date = self.state.active_date_window[0] if self.state.active_date_window else date.today().isoformat()
        for idx, profile in enumerate(self.state.data.get("actionable_profiles", [])[: self.settings.max_cases_per_run], start=1):
            pid = profile["profile_id"]
            receiver = profile.get("best_receiver", "")
            upload_row = uploads_by_profile.get(pid, {})
            event_date = upload_row.get("date") or fallback_date
            key = logical_case_key(pid, receiver, event_date)
            data_ready = upload_row.get("receipt_verified") is True or upload_row.get("status") == "ALREADY_AVAILABLE"
            if self.settings.commit and not data_ready:
                rows.append({
                    "case_id": "", "logical_case_key": key, "profile_id": pid,
                    "receiver_id": receiver, "date": event_date,
                    "status": "CASE_BLOCKED_DATA_COLLECTION", "real_case": False,
                    "upload_status": upload_row.get("status", "MISSING"),
                })
                continue
            registration_status = registrations_by_profile.get(pid, {}).get("status")
            if self.settings.commit and profile.get("draft") and registration_status not in {"REGISTERED", "ALREADY_REGISTERED"}:
                rows.append({
                    "case_id": "", "logical_case_key": key, "profile_id": pid,
                    "receiver_id": receiver, "date": event_date,
                    "status": "CASE_BLOCKED_PROFILE_REGISTRATION", "real_case": False,
                    "registration_status": registration_status or "MISSING",
                })
                continue
            if key in existing_by_key:
                existing = existing_by_key[key]
                rows.append({
                    "case_id": case_id(existing), "logical_case_key": key,
                    "profile_id": pid, "receiver_id": receiver, "date": event_date,
                    "status": "EXISTING_CASE_REUSED", "real_case": True,
                })
                continue
            evidence = {
                "logical_case_key": key,
                "profile_id": pid,
                "receiver_id": receiver,
                "event_date": event_date,
                "candidate_id": profile.get("candidate_id"),
                "validation": {
                    "classification": profile.get("classification"),
                    "required_logs_present": profile.get("required_logs_present"),
                    "coverage": profile.get("coverage"),
                },
                "upload": uploads_by_profile.get(pid),
                "evidence_classification": "RUNTIME_FACT",
                "promotion": "PROVISIONAL_RCA",
                "promotion_reason": "Human adjudication and all RCA gates are not yet complete.",
            }
            if not self.settings.commit:
                synthetic = f"dryrun-{stable_hash(key)[:12]}"
                rows.append({
                    "case_id": synthetic, "logical_case_key": key,
                    "profile_id": pid, "receiver_id": receiver, "date": event_date,
                    "status": "WOULD_CREATE", "real_case": False, "investigation": evidence,
                })
                continue
            create = await self.executor.call(
                phase="8", step=f"8.2.{idx}", tool="record_investigation_case",
                arguments={
                    "investigation_json": json_text(evidence),
                    "case_type": pid, "outcome_status": "unreviewed", "source": "nightly_rca_v6",
                },
            )
            response = _resp(create)
            cid = extract_identifier(response, "case_id", "id", "investigation_id")
            rows.append({
                "case_id": cid, "logical_case_key": key,
                "profile_id": pid, "receiver_id": receiver, "date": event_date,
                "status": "CREATED" if cid else "CREATE_FAILED", "real_case": bool(cid),
                "response": response,
            })
        compact_cases = [
            {k: row.get(k) for k in (
                "case_id", "logical_case_key", "profile_id", "receiver_id",
                "date", "status", "real_case", "upload_status", "registration_status",
            )}
            for row in rows
        ]
        self.state.data["cases"] = compact_cases
        return self.checkpoint("8", {"status": "OK", "cases": rows})

    async def phase_9_queue(self) -> dict[str, Any]:
        queue_id = self.effective_queue_id()
        eligible_cases = [c for c in self.state.data.get("cases", []) if c.get("real_case") and c.get("case_id")]
        case_ids = [str(c["case_id"]) for c in eligible_cases]
        if not case_ids:
            payload = {
                "status": "QUEUE_SKIPPED_NO_ELIGIBLE_CASES", "queue_id": queue_id,
                "queue_hash": "", "item_count": 0, "persisted": False,
                "reason": "No real case created or reused in this run.",
            }
            self.state.data["queue"] = dict(payload)
            return self.checkpoint("9", payload)

        preview = await self.executor.call(
            phase="9", step="9.1", tool="create_human_review_queue",
            arguments={
                "queue_id": queue_id, "case_ids": ",".join(case_ids),
                "max_items": min(len(case_ids), self.settings.max_cases_per_run), "dry_run": True,
            },
        )
        response = _resp(preview)
        queue_hash = extract_hash(response, "queue_hash", "expected_queue_hash", "confirmation_hash")
        item_count = int(response.get("item_count") or 0) if isinstance(response, dict) else 0
        if response is None:
            queue_status = "QUEUE_PREVIEW_FAILED"
        elif item_count == 0:
            queue_status = "QUEUE_SKIPPED_NO_ELIGIBLE_ITEMS"
        elif not queue_hash:
            queue_status = "QUEUE_HASH_MISSING"
        else:
            queue_status = "QUEUE_PREVIEW_OK"
        payload: dict[str, Any] = {
            "status": queue_status, "queue_id": queue_id, "queue_hash": queue_hash,
            "item_count": item_count, "case_ids": case_ids, "preview": response,
            "persisted": False, "idempotent": False,
        }
        if self.settings.commit and queue_status in {"QUEUE_PREVIEW_FAILED", "QUEUE_HASH_MISSING"}:
            self.state.metrics["persist_failures"] += 1
        if self.settings.commit and queue_status == "QUEUE_PREVIEW_OK":
            persist = await self.executor.call(
                phase="9", step="9.2", tool="create_human_review_queue",
                arguments={
                    "queue_id": queue_id, "case_ids": ",".join(case_ids),
                    "max_items": min(len(case_ids), self.settings.max_cases_per_run),
                    "dry_run": False,
                    "confirm_create": "CREATE_HUMAN_REVIEW_QUEUE",
                    "expected_queue_hash": queue_hash,
                },
            )
            persisted_response = _resp(persist)
            payload["persist_response"] = persisted_response
            payload["persisted"] = _persisted_or_idempotent(persisted_response, "queue_id")
            payload["idempotent"] = "ALREADY_EXISTS" in _result_code(persisted_response)
            payload["status"] = "QUEUE_PERSISTED" if payload["persisted"] else "QUEUE_PERSIST_FAILED"
            if not payload["persisted"]:
                self.state.metrics["persist_failures"] += 1
        if payload["persisted"]:
            verify = await self.executor.call(
                phase="9", step="9.3", tool="list_human_review_queue_items",
                arguments={"queue_id": queue_id, "limit": min(500, max(10, item_count))},
            )
            verification = _resp(verify)
            observed_count = None
            if isinstance(verification, dict):
                observed_count = verification.get("item_count")
                if observed_count is None:
                    observed_count = len(as_list(verification.get("items")))
            payload["verification"] = verification
            payload["verification_ok"] = verification is not None and (
                observed_count is None or int(observed_count) == item_count
            )
            if not payload["verification_ok"]:
                payload["status"] = "QUEUE_PERSISTED_VERIFICATION_MISMATCH"
        self.state.data["queue"] = {k: payload.get(k) for k in (
            "status", "queue_id", "queue_hash", "item_count", "persisted",
            "idempotent", "verification_ok", "case_ids",
        )}
        return self.checkpoint("9", payload)

    async def phase_10_bundles(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        queue = self.state.data.get("queue", {})
        queue_id = str(queue.get("queue_id") or self.effective_queue_id())
        for idx, case in enumerate(self.state.data.get("cases", []), start=1):
            if not case.get("real_case"):
                rows.append({**case, "status": "BUNDLE_SKIPPED_NO_REAL_CASE", "persisted": False})
                continue
            if not queue.get("persisted"):
                status = "BUNDLE_SKIPPED_QUEUE_PREVIEW_ONLY" if not self.settings.commit else "BUNDLE_BLOCKED_QUEUE_NOT_PERSISTED"
                rows.append({**case, "status": status, "persisted": False, "blocking_codes": ["QUEUE_NOT_PERSISTED"]})
                continue
            preview = await self.executor.call(
                phase="10", step=f"10.{idx}.a", tool="build_human_evidence_bundle",
                arguments={
                    "queue_id": queue_id, "case_id": case["case_id"],
                    "logical_case_key": case["logical_case_key"], "dry_run": True,
                },
            )
            response = _resp(preview)
            bundle_hash = extract_hash(response, "bundle_hash", "expected_bundle_hash", "confirmation_hash")
            blockers = blocking_codes(response) if response is not None else [preview.get("error") or "PREVIEW_FAILED"]
            row = {**case, "preview": response, "bundle_hash": bundle_hash, "blocking_codes": blockers, "persisted": False}
            if blockers:
                row["status"] = "BUNDLE_BLOCKED"
            elif not bundle_hash:
                row["status"] = "BUNDLE_HASH_MISSING"
            else:
                persist = await self.executor.call(
                    phase="10", step=f"10.{idx}.b", tool="build_human_evidence_bundle",
                    arguments={
                        "queue_id": queue_id, "case_id": case["case_id"],
                        "logical_case_key": case["logical_case_key"], "dry_run": False,
                        "confirm_build": "BUILD_HUMAN_EVIDENCE_BUNDLE",
                        "expected_bundle_hash": bundle_hash,
                    },
                )
                persisted_response = _resp(persist)
                row["persist_response"] = persisted_response
                row["bundle_id"] = extract_identifier(persisted_response, "bundle_id", "evidence_bundle_id")
                row["persisted"] = _persisted_or_idempotent(persisted_response, "bundle_id", "evidence_bundle_id")
                row["idempotent"] = "ALREADY_EXISTS" in _result_code(persisted_response)
                row["status"] = "BUNDLE_PERSISTED" if row["persisted"] else "BUNDLE_PERSIST_FAILED"
                if not row["persisted"]:
                    self.state.metrics["persist_failures"] += 1
            rows.append(row)
        compact_bundles = [
            {k: row.get(k) for k in (
                "case_id", "logical_case_key", "profile_id", "receiver_id", "date",
                "status", "bundle_hash", "bundle_id", "persisted", "idempotent", "blocking_codes",
            )}
            for row in rows
        ]
        self.state.data["bundles"] = compact_bundles
        return self.checkpoint("10", {"status": "OK", "queue_id": queue_id, "bundles": rows})

    async def phase_11_packets(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        queue = self.state.data.get("queue", {})
        queue_id = str(queue.get("queue_id") or self.effective_queue_id())
        persisted_bundles = [b for b in self.state.data.get("bundles", []) if b.get("persisted")]
        if persisted_bundles and not queue.get("persisted"):
            rows = [
                {**bundle, "packet_persisted": False, "packet_status": "PACKET_BLOCKED_QUEUE_NOT_PERSISTED"}
                for bundle in persisted_bundles
            ]
            self.state.data["packets"] = rows
            return self.checkpoint("11", {"status": "BLOCKED", "packets": rows})
        for idx, bundle in enumerate(persisted_bundles, start=1):
            preview = await self.executor.call(
                phase="11", step=f"11.{idx}.a", tool="export_human_adjudication_packet",
                arguments={
                    "queue_id": queue_id, "case_id": bundle["case_id"],
                    "logical_case_key": bundle["logical_case_key"], "dry_run": True,
                },
            )
            response = _resp(preview)
            packet_hash = extract_hash(response, "packet_hash", "expected_packet_hash", "expected_batch_hash", "confirmation_hash")
            blockers = blocking_codes(response) if response is not None else [preview.get("error") or "PREVIEW_FAILED"]
            row = {**bundle, "packet_preview": response, "packet_hash": packet_hash, "packet_persisted": False}
            if blockers:
                row["packet_status"] = "PACKET_BLOCKED"
                row["packet_blocking_codes"] = blockers
            elif not packet_hash:
                row["packet_status"] = "PACKET_HASH_MISSING"
            else:
                persist = await self.executor.call(
                    phase="11", step=f"11.{idx}.b", tool="export_human_adjudication_packet",
                    arguments={
                        "queue_id": queue_id, "case_id": bundle["case_id"],
                        "logical_case_key": bundle["logical_case_key"], "dry_run": False,
                        "confirm_export": "EXPORT_HUMAN_ADJUDICATION_PACKET",
                        "expected_packet_hash": packet_hash,
                    },
                )
                persisted_response = _resp(persist)
                packet_id = extract_identifier(persisted_response, "packet_id", "batch_id")
                row["packet_id"] = packet_id
                row["packet_persisted"] = _persisted_or_idempotent(persisted_response, "packet_id", "batch_id")
                row["idempotent"] = "ALREADY_EXISTS" in _result_code(persisted_response)
                row["packet_status"] = "PACKET_PERSISTED" if row["packet_persisted"] else "PACKET_PERSIST_FAILED"
                if row["packet_persisted"] and packet_id:
                    audit = await self.executor.call(
                        phase="11", step=f"11.{idx}.c", tool="audit_human_adjudication_packet",
                        arguments={"packet_id": packet_id},
                    )
                    row["audit"] = _resp(audit)
                elif not row["packet_persisted"]:
                    self.state.metrics["persist_failures"] += 1
            rows.append(row)
        compact_packets = []
        for row in rows:
            compact = {k: row.get(k) for k in (
                "case_id", "logical_case_key", "profile_id", "receiver_id", "date",
                "packet_hash", "packet_id", "packet_persisted", "packet_status",
                "idempotent", "packet_blocking_codes",
            )}
            if row.get("audit") is not None:
                compact["audit_ok"] = _call_pass(row.get("audit"))
            compact_packets.append(compact)
        self.state.data["packets"] = compact_packets
        return self.checkpoint("11", {"status": "OK", "queue_id": queue_id, "packets": rows})

    async def phase_12_fix_lineage(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for idx, packet in enumerate(self.state.data.get("packets", []), start=1):
            if not packet.get("packet_persisted"):
                continue
            if not self.settings.commit:
                rows.append({"logical_case_key": packet["logical_case_key"], "status": "WOULD_REFRESH"})
                continue
            result = await self.executor.call(
                phase="12", step=f"12.{idx}", tool="human_review_fix_lineage_refresh",
                arguments={"logical_case_key": packet["logical_case_key"], "scan_limit": 100},
            )
            rows.append({
                "logical_case_key": packet["logical_case_key"],
                "status": result.get("status"), "response": _resp(result),
                "classification": classify_fix_lineage(_resp(result)),
            })
        self.state.data["fix_lineage"] = rows
        return self.checkpoint("12", {"status": "OK", "fix_lineage": rows})

    async def phase_13_dashboard(self) -> dict[str, Any]:
        materialize = await self.executor.call(
            phase="13", step="13.1", tool="human_review_materialize_engineer_contexts",
            arguments={
                "scan_limit": 50, "persist": self.settings.commit,
                "max_cases_per_run": 10, "max_runtime_seconds": self.settings.max_runtime_seconds,
            },
        )
        status = await self.executor.call(
            phase="13", step="13.2", tool="human_review_dashboard_status",
            arguments={"scan_limit": 50},
        )
        materialization_response = _resp(materialize)
        dashboard_response = _resp(status)
        materialization_ok = _call_pass(materialization_response)
        dashboard_ok = _call_pass(dashboard_response)
        payload = {
            "status": "OK" if materialization_ok and dashboard_ok else "PARTIAL",
            "materialization": materialization_response, "dashboard_status": dashboard_response,
            "materialization_ok": materialization_ok, "dashboard_ready": dashboard_ok,
            "persist_requested": self.settings.commit,
        }
        self.state.data["dashboard"] = payload
        return self.checkpoint("13", payload)

    async def phase_14_canaries(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        auth = await self.executor.call(phase="14", step="14.A", tool="get_heavy_auth_status", arguments={})
        rows.append({"canary": "A_heavy_auth", "pass": _call_pass(_resp(auth)), "skipped": False, "evidence": _resp(auth)})

        packet_ids = [p.get("packet_id") for p in self.state.data.get("packets", []) if p.get("packet_id")]
        if packet_ids:
            for idx, packet_id in enumerate(packet_ids, start=1):
                ready = await self.executor.call(
                    phase="14", step=f"14.B.{idx}", tool="validate_human_adjudication_packet_readiness",
                    arguments={"packet_id": packet_id},
                )
                rows.append({"canary": f"B_packet_{packet_id}", "pass": _call_pass(_resp(ready)), "skipped": False, "evidence": _resp(ready)})
        else:
            rows.append({"canary": "B_packet_readiness", "pass": None, "skipped": True, "evidence": {"status": "SKIPPED_NO_PERSISTED_PACKET"}})

        if self.state.data.get("queue", {}).get("persisted"):
            queue = await self.executor.call(
                phase="14", step="14.C", tool="audit_human_review_queue_integrity",
                arguments={"queue_id": self.state.data.get("queue", {}).get("queue_id") or self.effective_queue_id(), "limit": 10},
            )
            rows.append({"canary": "C_queue_integrity", "pass": _call_pass(_resp(queue)), "skipped": False, "evidence": _resp(queue)})
        else:
            rows.append({"canary": "C_queue_integrity", "pass": None, "skipped": True, "evidence": {"status": "SKIPPED_QUEUE_NOT_PERSISTED"}})

        real_cases = [c for c in self.state.data.get("cases", []) if c.get("real_case") and c.get("case_id")]
        if real_cases:
            for idx, case in enumerate(real_cases, start=1):
                stale = await self.executor.call(
                    phase="14", step=f"14.D.{idx}", tool="audit_case_adjudication_staleness",
                    arguments={"case_id": case["case_id"]},
                )
                rows.append({"canary": f"D_case_{case['case_id']}", "pass": _call_pass(_resp(stale)), "skipped": False, "evidence": _resp(stale)})
        else:
            rows.append({"canary": "D_case_staleness", "pass": None, "skipped": True, "evidence": {"status": "SKIPPED_NO_PERSISTED_CASE"}})

        # E — Expected-negative canary: verify the synthetic test receiver returns
        # a recognized no-logs result (not auth failure, timeout, or schema error).
        e_result = await self.executor.call(
            phase="14", step="14.E", tool="list_dates",
            arguments={"receiver_id": self.settings.test_receiver_id, "limit": 5},
            tolerate_error=True,
        )
        e_evidence = _classify_expected_no_logs(e_result, self.settings.test_receiver_id)
        rows.append({
            "canary": "E_expected_no_logs",
            "pass": e_evidence["pass"],
            "skipped": False,
            "evidence": e_evidence,
        })
        if e_evidence["pass"]:
            self.state.metrics["expected_negative_canary_pass"] += 1

        executed = [row for row in rows if not row.get("skipped")]
        failures = [row for row in executed if row.get("pass") is not True]
        skipped = [row for row in rows if row.get("skipped")]
        if failures:
            overall = "FAIL"
        elif skipped:
            overall = "PASS_WITH_SKIPS"
        else:
            overall = "PASS"
        payload = {
            "status": overall,
            "executed_count": len(executed),
            "failure_count": len(failures),
            "skipped_count": len(skipped),
            "canaries": rows,
        }
        self.state.data["canaries"] = payload
        return self.checkpoint("14", payload)

    def phase_15_finalize_local(self) -> dict[str, Any]:
        self.state.data["learning_recommendations"] = learning_recommendations(self.state.data)
        hard_blockers = self._remaining_blockers()
        pending_notes = self._pending_investigation_notes()
        canaries = self.state.data.get("canaries", {})
        numbers = {
            "queues_persisted": int(bool(self.state.data.get("queue", {}).get("persisted"))),
            "bundles_built": sum(1 for x in self.state.data.get("bundles", []) if x.get("persisted")),
            "packets_persisted": sum(1 for x in self.state.data.get("packets", []) if x.get("packet_persisted")),
            "fix_lineage_records": len(self.state.data.get("fix_lineage", [])),
            "canaries_failed": int(canaries.get("failure_count", 0)),
            "canaries_skipped": int(canaries.get("skipped_count", 0)),
            "dashboard_ready": int(bool(self.state.data.get("dashboard", {}).get("dashboard_ready"))),
            "workflow_blockers": len(hard_blockers),
            "pending_investigations": len(self.state.data.get("pending_investigations", [])),
            "TOOL_NOT_FOUND_count": self.state.metrics["tool_not_found"],
            "STEP_FAILED_count": self.state.metrics["step_failed"],
            "ROLE_ERRORS": self.state.metrics["role_errors"],
            "PERSIST_FAILURES": self.state.metrics["persist_failures"],
            "WRITE_BLOCKED": self.state.metrics.get("write_blocked", 0),
            "expected_negative_canary_pass": self.state.metrics.get("expected_negative_canary_pass", 0),
        }
        all_notes = hard_blockers + pending_notes
        payload = {
            "status": "COMPLETE",
            "executive_verdict": self._verdict(numbers),
            "final_numbers": numbers,
            "remaining_blockers": all_notes or ["None observed in this run."],
            "hard_blockers": hard_blockers,
            "pending_investigation_notes": pending_notes,
            "learning_recommendations": self.state.data["learning_recommendations"],
        }
        self.state.data["final"] = payload
        return self.checkpoint("15", payload)

    def _verdict(self, numbers: dict[str, int]) -> str:
        if numbers["ROLE_ERRORS"] or numbers["PERSIST_FAILURES"] or numbers["WRITE_BLOCKED"]:
            return "PARTIAL — persistence or authorization defects require operator attention."
        if numbers["STEP_FAILED_count"] or numbers["TOOL_NOT_FOUND_count"]:
            return "PARTIAL — workflow completed with tool/data gaps."
        if numbers["canaries_failed"]:
            return "PARTIAL — one or more acceptance canaries failed."
        if self.settings.commit and not numbers["dashboard_ready"]:
            return "PARTIAL — persisted workflow completed, but dashboard readiness was not confirmed."
        if numbers["workflow_blockers"]:
            return "PARTIAL — workflow completed with evidence or promotion blockers."
        if self.settings.commit and numbers["pending_investigations"]:
            return "DEFERRED — pending investigations were persisted and will be reconciled automatically on the next cron run."
        if self.settings.commit and numbers["packets_persisted"]:
            return "PASS — serial investigation, persistence, review packet, lineage, dashboard, and canaries completed."
        if self.settings.commit:
            return "PASS_WITH_NO_PACKETS — workflow completed cleanly, but no case was eligible for a persisted packet."
        return "DRY_RUN_PASS — read/preview workflow completed without authorized writes; persistence canaries were explicitly skipped."

    def _pending_investigation_notes(self) -> list[str]:
        notes: list[str] = []
        pending_classes = {"AWAITING_LOGS", "PROFILE_NEEDS_MORE_DATA", "UPLOAD_PLAN_ONLY"}
        for validation in self.state.data.get("profile_validations", []):
            if validation.get("classification") in pending_classes:
                notes.append(
                    f"{validation.get('profile_id')}: {validation.get('classification')} — {validation.get('reason')}"
                )
        return notes

    def _remaining_blockers(self) -> list[str]:
        blockers: list[str] = []
        queue = self.state.data.get("queue", {})
        acceptable_queue_statuses = {
            "QUEUE_PERSISTED", "QUEUE_PREVIEW_OK",
            "QUEUE_SKIPPED_NO_ELIGIBLE_CASES", "QUEUE_SKIPPED_NO_ELIGIBLE_ITEMS",
        }
        if queue and queue.get("status") not in acceptable_queue_statuses:
            blockers.append(f"queue: {queue.get('status')}")
        pending_classes = {"AWAITING_LOGS", "PROFILE_NEEDS_MORE_DATA", "UPLOAD_PLAN_ONLY"}
        for validation in self.state.data.get("profile_validations", []):
            classification = validation.get("classification")
            if classification not in {"PROFILE_VALIDATION_PASS", *pending_classes}:
                blockers.append(f"{validation.get('profile_id')}: {classification} — {validation.get('reason')}")
        for case in self.state.data.get("cases", []):
            if case.get("status") == "CASE_BLOCKED_DATA_COLLECTION":
                blockers.append(f"{case.get('logical_case_key')}: CASE_BLOCKED_DATA_COLLECTION ({case.get('upload_status')})")
            elif case.get("status") == "CASE_BLOCKED_PROFILE_REGISTRATION":
                blockers.append(f"{case.get('logical_case_key')}: CASE_BLOCKED_PROFILE_REGISTRATION ({case.get('registration_status')})")
        for bundle in self.state.data.get("bundles", []):
            if bundle.get("status") in {
                "BUNDLE_BLOCKED", "BUNDLE_PERSIST_FAILED", "BUNDLE_HASH_MISSING",
                "BUNDLE_BLOCKED_QUEUE_NOT_PERSISTED",
            }:
                blockers.append(f"{bundle.get('case_id')}: {bundle.get('status')} {bundle.get('blocking_codes', [])}")
        for packet in self.state.data.get("packets", []):
            if packet.get("packet_status") not in {"PACKET_PERSISTED", "PACKET_PREVIEW_OK"}:
                blockers.append(f"{packet.get('case_id')}: {packet.get('packet_status')}")
            elif packet.get("audit_ok") is False:
                blockers.append(f"{packet.get('case_id')}: PACKET_AUDIT_FAILED")
        return blockers

