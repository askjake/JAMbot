"""Optional Google Chat notification. No webhook or credential is embedded."""
from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .state import RunState, RunStore

log = logging.getLogger("nightly_rca.notify")

_SEP = "━" * 42
_EXPECTED_PHASES = (
    "0", "1", "D1", "2", "2b", "3", "4", "5", "6",
    "7", "8", "9", "10", "11", "12", "13", "14", "15",
)


def send(webhook_url: str, text: str, timeout: float = 8.0) -> dict[str, Any] | None:
    if not webhook_url:
        log.info("notification skipped: NIGHTLY_RCA_WEBHOOK is not configured")
        return None
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        log.warning("notification failed (non-fatal): %s", exc)
        return None


def _verdict_emoji(verdict: str) -> str:
    value = verdict.upper()
    if value.startswith("PASS"):
        return "✅"
    if value.startswith(("DRY_RUN", "DEFERRED")):
        return "🔵"
    if value.startswith("PARTIAL"):
        return "🟡"
    if value.startswith("ABORTED"):
        return "🔴"
    return "⚪"


def _rows(value: Any, *keys: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if not isinstance(value, dict):
        return []
    for key in keys:
        rows = value.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _count(row: dict[str, Any], *keys: str) -> int:
    for key in keys:
        raw = row.get(key)
        try:
            if raw is not None and raw != "":
                return int(float(raw))
        except (TypeError, ValueError):
            continue
    return 0


def _confidence(anomaly_records: list, validations: list, cases: list) -> str:
    confirmed = [
        case for case in cases
        if str(case.get("status") or case.get("outcome_status") or "").upper()
        in {"CONFIRMED", "CONFIRMED_ISSUE", "ROOT_CAUSE_CONFIRMED", "RESOLVED"}
    ]
    if confirmed:
        return "🟢 High — confirmed root cause(s) this cycle"
    if cases:
        return "🟡 Moderate — evidence-backed investigation case(s) persisted for review"
    if any(v.get("classification") == "PROFILE_VALIDATION_PASS" for v in validations):
        return "🟡 Moderate — log-validated profiles are ready for case promotion"
    if anomaly_records:
        return "🟡 Low-moderate — anomalies detected, no log-confirmed triage yet"
    return "⚪ No high-confidence findings this cycle"


def _load_phase(store: "RunStore", phase: str) -> dict[str, Any]:
    safe = phase.replace(".", "_").replace("/", "_")
    path: Path = store.run_dir / f"phase_{safe}.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            log.debug("could not load phase file %s: %s", path, exc)
    return {}


def summarize(state: "RunState", store: "RunStore") -> str:  # type: ignore[override]
    """Build a concise, truthful Google Chat report from persisted run state."""
    final = state.data.get("final", {})
    verdict = str(final.get("executive_verdict", "UNKNOWN"))
    nums = final.get("final_numbers", {})
    metrics = state.metrics
    date_window = state.active_date_window
    blockers = [
        str(item) for item in final.get("remaining_blockers", [])
        if item and "none" not in str(item).lower()
    ]
    recs = _rows(final.get("learning_recommendations", []))

    candidates = _rows(state.data.get("candidate_clusters", []))
    validations = _rows(state.data.get("profile_validations", []))
    all_cases = _rows(state.data.get("cases", []))
    real_cases = [case for case in all_cases if case.get("real_case") and case.get("case_id")]
    bundles = _rows(state.data.get("bundles", []))
    unresolved = _rows(state.data.get("unresolved_resolution", {}), "rows")
    canaries = _rows(state.data.get("canaries", {}), "canaries")
    pending = _rows(state.data.get("pending_investigations", []))

    phase2 = _load_phase(store, "2")
    anomaly_payload = phase2.get("anomalies", {})
    anomaly_records = _rows(anomaly_payload, "records", "anomalies", "items", "results")
    anomaly_summary = _rows(anomaly_payload, "summary") or anomaly_records
    phase2_candidates = _rows(phase2.get("candidate_clusters", [])) or candidates

    completed_at = (state.completed_at or "").replace("+00:00", "Z")
    date_text = (
        ", ".join(date_window[:3]) + ("…" if len(date_window) > 3 else "")
        if date_window else "UNKNOWN"
    )

    lines: list[str] = [
        f"📋 *Nightly RCA — STB Intelligence Report*  `{completed_at}`",
        "",
        _SEP,
        f"{_verdict_emoji(verdict)} *Verdict:* {verdict}",
        f"🔑 *Run:* `{state.run_id}`  |  *Phases:* {len(state.completed_phases)}/{len(_EXPECTED_PHASES)}",
        f"📅 *Date window:* {date_text}",
    ]
    effective = state.data.get("effective_configuration", {})
    if isinstance(effective, dict) and effective:
        lines.append(
            "⚙️ *Effective mode:* "
            f"{effective.get('effective_mode', state.mode)} "
            f"({effective.get('effective_mode_source', 'UNKNOWN')}); "
            f"cron TZ={effective.get('cron_timezone', 'UNKNOWN')} "
            f"[{effective.get('cron_timezone_evidence', 'NOT_DIRECTLY_VERIFIED')}]"
        )
    lines.extend([
        _SEP,
    ])

    lines.append("*1. 🔭 Fleet Anomalies (Phase 2)*")
    if anomaly_summary:
        total = sum(_count(row, "max_actual", "actual", "actual_count", "count", "event_count") for row in anomaly_summary)
        lines.append(f"  {len(anomaly_summary)} alert type(s) above threshold  |  {total:,} total events")
        ordered = sorted(
            anomaly_summary,
            key=lambda row: _count(row, "max_actual", "actual", "actual_count", "count", "event_count"),
            reverse=True,
        )
        for anomaly in ordered[:6]:
            name = str(anomaly.get("alert_name") or anomaly.get("name") or "?")
            actual = _count(anomaly, "max_actual", "actual", "actual_count", "count", "event_count")
            typical = _count(anomaly, "max_typical", "typical", "baseline_count")
            pct = ""
            for record in anomaly_records:
                if str(record.get("alert_name") or record.get("name") or "") == name:
                    try:
                        pct = f"  (+{float(record.get('percentage_change', 0)):.0f}% vs typical)"
                    except (TypeError, ValueError):
                        pct = ""
                    break
            lines.append(f"  • `{name}` — {actual:,} actual  (typical {typical:,}){pct}")
    else:
        lines.append("  _(no anomalies above threshold this cycle)_")
    lines.append("")

    lines.append("*2. 🆕 New STBs Queued for Analysis (Phase 2)*")
    display_candidates = phase2_candidates or candidates
    if display_candidates:
        for candidate in display_candidates[:8]:
            samples = _rows(candidate.get("receiver_samples", []))
            sample = samples[0] if samples else {}
            receivers = [str(value) for value in candidate.get("receivers_seen", []) if value]
            receiver = str(
                sample.get("t_receiver_id") or sample.get("receiver_id")
                or (receivers[0] if receivers else candidate.get("candidate_id", "?"))
            )
            count = _count(sample, "n_alert_count", "alert_count", "count") or _count(
                candidate, "alert_count", "actual_count", "count"
            )
            family = (candidate.get("signal_families") or [candidate.get("alert_name") or "?"])[0]
            lines.append(
                f"  • `{receiver}` — {candidate.get('suspected_profile', '?')} | {family} "
                f"({count:,} alerts/24h)"
            )
    else:
        lines.append("  _(no new candidates — fleet is quiet or all patterns are known)_")
    lines.append("")

    lines.append("*3. 🚨 STB Problems Identified (Phases 8–11)*")
    if real_cases:
        for case in real_cases[:8]:
            lines.append(
                f"  • `{case.get('receiver_id', '?')}` | "
                f"{case.get('profile', case.get('profile_id', '?'))} — {case.get('status', '?')}"
            )
    elif pending:
        lines.append("  _(No confirmed problems — log acquisition is pending and will resume automatically)_")
    else:
        lines.append("  _(No evidence-backed cases were opened this cycle)_")
    lines.append("")

    lines.append("*4. 🔄 Investigation Status (Phases 3–6)*")
    if validations:
        lines.extend(["```", f"{'Receiver':<16} {'Profile':<28} {'Logs':<5} Status", f"{'─'*16} {'─'*28} {'─'*5} {'─'*18}"])
        for validation in validations[:10]:
            logs = "✅" if validation.get("required_logs_present") else "❌"
            lines.append(
                f"{str(validation.get('best_receiver', '?')):<16} "
                f"{str(validation.get('profile_id', '?'))[:28]:<28} "
                f"{logs:<5} {validation.get('classification', '?')}"
            )
        lines.append("```")
    elif display_candidates:
        lines.append(f"  ⏸ {len(display_candidates)} candidate(s) remain unvalidated")
    else:
        lines.append("  ✅ No active investigations this cycle")
    lines.append("")

    lines.append("*5. 📌 Resolution Status (Phase D1 + Cases)*")
    resolved = [row for row in unresolved if row.get("action_taken") not in (None, "", "NO_ACTION", "NONE")]
    if resolved:
        for row in resolved[:5]:
            lines.append(
                f"  • `{row.get('case_id', '?')}` — {row.get('new_classification', '?')} | "
                f"{row.get('action_taken', '?')}"
            )
    elif unresolved:
        lines.append(f"  _{len(unresolved)} unresolved case(s) reviewed — no new closures this cycle_")
    else:
        lines.append("  _No open cases_")
    lines.append("")

    lines.append("*6. 🎯 Confidence*")
    lines.append(f"  {_confidence(anomaly_records, validations, real_cases)}")
    for blocker in blockers[:3]:
        lines.append(f"  ⚠️  {blocker}")
    lines.append("")

    lines.append("*7. 🐥 System Health Canaries (Phase 14)*")
    if canaries:
        for canary in canaries:
            if canary.get("pass") is True:
                icon = "✅"
            elif canary.get("skipped"):
                icon = "⏭ Skipped"
            else:
                icon = "❌"
            lines.append(f"  {icon}  `{canary.get('canary', '?')}`")
    else:
        lines.append("  _(canary data unavailable)_")
    lines.append("")

    lines.append("*8. ➡️  What's Needed Next*")
    step = 1
    if anomaly_records and not display_candidates:
        lines.append(f"  {step}. Candidate clustering — anomalies were observed but no clusters formed")
        step += 1
    if display_candidates and not validations:
        lines.append(f"  {step}. Validation retry — {len(display_candidates)} candidate(s) have not reached profile validation")
        step += 1
    if pending:
        pending_metrics = state.data.get("pending_selector_metrics", {})
        _sel_logical = pending_metrics.get("pending_selected_logical", pending_metrics.get("pending_selected", "?"))
        _deferred_logical = pending_metrics.get("pending_deferred_batch_logical", 0)
        _backoff_phys = pending_metrics.get("pending_backoff_physical", 0)
        _dup_phys = pending_metrics.get("pending_duplicate_suppressed_physical", 0)
        lines.append(f"  {step}. Automatic receipt reconciliation:")
        lines.append(f"       {_sel_logical} logical investigations selected.")
        if _deferred_logical:
            lines.append(f"       {_deferred_logical} eligible logical investigations were deferred by the batch limit.")
        if _backoff_phys:
            lines.append(f"       {_backoff_phys} physical tracker rows are in backoff.")
        if _dup_phys:
            lines.append(f"       {_dup_phys} additional physical tracker rows represent duplicate logical investigations.")
        historical = sum(1 for row in pending if row.get("historical_carry_forward") is True)
        current = sum(1 for row in pending if row.get("currentness") == "CURRENT_RUN")
        unknown_origin = sum(1 for row in pending if row.get("currentness") == "ORIGIN_RUN_UNKNOWN")
        if historical or current or unknown_origin:
            lines.append(
                f"       Physical-row currentness: current={current}, "
                f"historical_carry_forward={historical}, origin_unknown={unknown_origin}."
            )
        step += 1
    passed = [row for row in validations if row.get("classification") == "PROFILE_VALIDATION_PASS"]
    if passed and not real_cases:
        lines.append(f"  {step}. Case promotion review — {len(passed)} log-validated profile(s) produced no persisted case")
        step += 1
    if real_cases and not bundles:
        lines.append(f"  {step}. Bundle/packet generation — {len(real_cases)} case(s) exist, but no bundles were built")
        step += 1
    if unresolved:
        lines.append(f"  {step}. Review {len(unresolved)} unresolved case(s) for signal changes")
        step += 1
    for recommendation in recs[:3]:
        lines.append(
            f"  {step}. [{recommendation.get('type', 'REC')}] {recommendation.get('reason', '')} "
            f"→ {recommendation.get('action', '')}"
        )
        step += 1
    if step == 1:
        lines.append("  ✅ No outstanding actions")
    lines.append("")

    step_failures = metrics.get("step_failed", nums.get("STEP_FAILED_count", 0))
    tool_calls = metrics.get("tool_calls", "?")
    write_blocked = metrics.get("write_blocked", nums.get("WRITE_BLOCKED", "?"))
    expected_neg = metrics.get("expected_negative_canary_pass", 0)
    lines.extend([
        _SEP,
        f"📊 *Metrics:* tool_calls={tool_calls}  step_failed={step_failures}  write_blocked={write_blocked}  expected_negative_canary_pass={expected_neg}",
        _SEP,
    ])
    return "\n".join(lines)
