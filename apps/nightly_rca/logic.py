"""Pure transformation logic: robust parsing, matching, scoring, and bounded learning."""
from __future__ import annotations

import json
import math
import re
from datetime import date, datetime
from typing import Any


EVIDENCE_TAGS = {
    "RUNTIME_FACT", "SOURCE_CODE_FACT", "BUILD_APPLICABILITY_FACT",
    "CUSTOMER_IMPACT_FACT", "STRONG_INFERENCE", "HYPOTHESIS",
    "CONTRADICTED", "EVIDENCE_UNAVAILABLE", "TOOL_OR_DATA_DEFECT",
}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def rows_from(response: Any, *keys: str) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return [x for x in response if isinstance(x, dict)]
    if not isinstance(response, dict):
        return []
    for key in keys:
        value = response.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []


def parse_dates(response: Any) -> list[str]:
    values: list[Any] = []
    if isinstance(response, dict):
        values.extend(as_list(response.get("dates")))
        values.extend(as_list(response.get("parsed_dates")))
        values.extend(as_list(response.get("items")))
    elif isinstance(response, list):
        values.extend(response)
    parsed: set[str] = set()
    for value in values:
        candidate = value
        if isinstance(value, dict):
            candidate = value.get("date") or value.get("event_date") or value.get("day")
        if not candidate:
            continue
        text = str(candidate)[:10]
        try:
            datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            continue
        parsed.add(text)
    return sorted(parsed, reverse=True)


def normalize_cases(response: Any) -> list[dict[str, Any]]:
    return rows_from(response, "cases", "items", "results", "investigations")


def case_id(case: dict[str, Any]) -> str:
    return str(case.get("case_id") or case.get("id") or case.get("investigation_id") or "")


def case_profile(case: dict[str, Any]) -> str:
    return str(case.get("issue_profile") or case.get("profile") or case.get("case_type") or "")


def case_receiver(case: dict[str, Any]) -> str:
    return str(case.get("receiver_id") or case.get("rxid") or case.get("device_id") or "")


def case_date(case: dict[str, Any]) -> str:
    raw = case.get("event_date") or case.get("date") or case.get("created_date") or ""
    return str(raw)[:10]


def logical_case_key(profile: str, receiver: str, event_date: str) -> str:
    return f"{profile}|{receiver}|{event_date}"


def normalize_catalog(response: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(response, dict):
        return {}
    profiles = response.get("profiles") or response.get("catalog") or response.get("items") or response
    if isinstance(profiles, list):
        out: dict[str, dict[str, Any]] = {}
        for row in profiles:
            if not isinstance(row, dict):
                continue
            pid = str(row.get("profile_id") or row.get("issue_profile") or row.get("name") or "")
            if pid:
                out[pid] = row
        return out
    if isinstance(profiles, dict):
        return {str(k): (v if isinstance(v, dict) else {"value": v}) for k, v in profiles.items()}
    return {}


def _numeric_count(row: dict[str, Any]) -> int:
    """Read count aliases emitted by RTR and anomaly response variants."""
    for key in (
        "actual_count", "count", "actual", "max_actual", "event_count",
        "n_alert_count", "doc_count", "total_alert_count",
    ):
        value = row.get(key)
        try:
            if value is not None:
                return max(0, int(float(value)))
        except (TypeError, ValueError):
            continue
    return 0


def _buckets(response: Any) -> list[dict[str, Any]]:
    if not isinstance(response, dict):
        return []
    candidates = response.get("buckets") or response.get("counts") or response.get("items") or response.get("results")
    if isinstance(candidates, dict):
        return [{"key": k, "count": v} for k, v in candidates.items()]
    if isinstance(candidates, list):
        return [x for x in candidates if isinstance(x, dict)]
    total = _numeric_count(response)
    if total:
        return [{"key": "FAMILY_TOTAL", "count": total}]
    return []


def _anomaly_rows(response: Any) -> list[dict[str, Any]]:
    """Combine record and summary variants instead of accepting only the first list."""
    if isinstance(response, list):
        return [x for x in response if isinstance(x, dict)]
    if not isinstance(response, dict):
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("records", "anomalies", "summary", "items", "results"):
        value = response.get(key)
        if not isinstance(value, list):
            continue
        for row in value:
            if not isinstance(row, dict):
                continue
            marker = json.dumps(row, sort_keys=True, default=str)
            if marker not in seen:
                seen.add(marker)
                rows.append(row)
    return rows


def discover_clusters(
    family_results: dict[str, Any],
    anomaly_response: Any,
    active_dates: list[str],
    history_counts: dict[str, int] | None = None,
    history_validation_passes: dict[str, int] | None = None,
    history_validation_failures: dict[str, int] | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    history_counts = history_counts or {}
    history_validation_passes = history_validation_passes or {}
    history_validation_failures = history_validation_failures or {}
    merged: dict[str, dict[str, Any]] = {}
    default_date = active_dates[0] if active_dates else date.today().isoformat()

    for family, response in family_results.items():
        for bucket in _buckets(response):
            name = str(bucket.get("key") or bucket.get("alert_name") or bucket.get("name") or "")
            if not name or name == "FAMILY_TOTAL":
                continue
            count = _numeric_count(bucket)
            if count <= 0:
                continue
            key = name.upper()
            row = merged.setdefault(key, {
                "candidate_id": f"cand_{len(merged)+1:03d}",
                "suspected_profile": slugify(name),
                "alert_name": name,
                "receivers_seen": [],
                "receiver_samples": [],
                "dates": [default_date],
                "signal_families": [],
                "key_patterns": [],
                "alert_count": 0,
                "anomaly_score": 0.0,
                "evidence_tags": ["RUNTIME_FACT"],
            })
            row["signal_families"].append(family)
            row["key_patterns"].append(name)
            row["alert_count"] += count

    for anomaly in _anomaly_rows(anomaly_response):
        name = str(anomaly.get("alert_name") or anomaly.get("name") or anomaly.get("pattern") or "UNKNOWN_ANOMALY")
        key = name.upper()
        row = merged.setdefault(key, {
            "candidate_id": f"cand_{len(merged)+1:03d}",
            "suspected_profile": slugify(name),
            "alert_name": name,
            "receivers_seen": [],
            "receiver_samples": [],
            "dates": [default_date],
            "signal_families": ["ml_anomaly"],
            "key_patterns": [name],
            "alert_count": _numeric_count(anomaly),
            "anomaly_score": 0.0,
            "evidence_tags": ["RUNTIME_FACT"],
        })
        row.setdefault("alert_name", name)
        row.setdefault("receiver_samples", [])
        row["signal_families"].append("ml_anomaly")
        row["alert_count"] = max(int(row.get("alert_count") or 0), _numeric_count(anomaly))
        score = anomaly.get("record_score") or anomaly.get("score") or anomaly.get("anomaly_score") or 0
        try:
            row["anomaly_score"] = max(float(score), row["anomaly_score"])
        except (TypeError, ValueError):
            pass
        receivers = (
            as_list(anomaly.get("receiver_ids")) + as_list(anomaly.get("receivers")) +
            as_list(anomaly.get("affected_receivers"))
        )
        # receiver_samples: [{t_receiver_id, n_alert_count, t_host_receiver_id}]
        for sample in as_list(anomaly.get("receiver_samples")):
            rid = sample.get("t_receiver_id") if isinstance(sample, dict) else None
            if isinstance(sample, dict) and rid:
                row["receiver_samples"].append(dict(sample))
            if rid and str(rid) not in receivers:
                receivers.append(str(rid))
        for receiver in receivers:
            rid = receiver.get("receiver_id") if isinstance(receiver, dict) else receiver
            if rid:
                row["receivers_seen"].append(str(rid))
        versions = as_list(anomaly.get("versions") or anomaly.get("affected_versions"))
        if versions:
            row["affected_versions"] = sorted({str(v) for v in versions})

    ranked: list[dict[str, Any]] = []
    for row in merged.values():
        row["signal_families"] = sorted(set(row["signal_families"]))
        row["key_patterns"] = sorted(set(row["key_patterns"]))
        row["receivers_seen"] = sorted(set(row["receivers_seen"]))[:50]
        sample_by_receiver: dict[str, dict[str, Any]] = {}
        for sample in row.get("receiver_samples", []):
            if not isinstance(sample, dict):
                continue
            rid = str(sample.get("t_receiver_id") or sample.get("receiver_id") or "")
            if not rid:
                continue
            prior = sample_by_receiver.get(rid, {})
            if _numeric_count(sample) >= _numeric_count(prior):
                sample_by_receiver[rid] = sample
        row["receiver_samples"] = sorted(
            sample_by_receiver.values(), key=lambda x: (-_numeric_count(x), str(x.get("t_receiver_id") or ""))
        )[:50]
        row.setdefault("alert_name", row["key_patterns"][0] if row["key_patterns"] else row["suspected_profile"])
        signal_key = row["suspected_profile"]
        recurrence = history_counts.get(signal_key, 0)
        prior_passes = history_validation_passes.get(signal_key, 0)
        prior_failures = history_validation_failures.get(signal_key, 0)
        row["recurrence_nights"] = recurrence
        row["prior_validation_passes"] = prior_passes
        row["prior_validation_failures"] = prior_failures
        score = (
            math.log10(max(1, row["alert_count"])) * 20
            + row["anomaly_score"]
            + min(recurrence, 10) * 5
            + min(prior_passes, 5) * 4
            - min(prior_failures, 5) * 10
        )
        row["priority_score"] = round(max(0.0, score), 2)
        if row["anomaly_score"] >= 95 or row["alert_count"] >= 10000:
            row["severity_guess"] = "high"
        elif row["anomaly_score"] >= 90 or row["alert_count"] >= 1000:
            row["severity_guess"] = "medium"
        else:
            row["severity_guess"] = "low"
        evidence_sources = len(row["signal_families"])
        if prior_failures >= 2 and prior_passes == 0:
            row["confidence"] = "low"
            row["learning_disposition"] = "REVIEW_FALSE_POSITIVE_OR_COVERAGE"
        elif prior_passes >= 2:
            row["confidence"] = "high"
            row["learning_disposition"] = "REPEATEDLY_VALIDATED"
        else:
            row["confidence"] = "high" if evidence_sources >= 2 else "medium"
            row["learning_disposition"] = "OBSERVE"
        row["known_profile_match"] = None
        row["new_profile_needed"] = None
        ranked.append(row)
    ranked.sort(key=lambda x: (-x["priority_score"], x["candidate_id"]))
    return ranked[:limit]


def slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value[:64] or "unknown_signal"


def _patterns(profile: dict[str, Any], names: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for name in names:
        value = profile.get(name)
        if isinstance(value, dict):
            for v in value.values():
                out.extend(str(x) for x in as_list(v))
        else:
            out.extend(str(x) for x in as_list(value))
    return [x for x in out if x]


def regex_match(pattern: str, text: str) -> bool:
    try:
        return bool(re.search(pattern, text, re.IGNORECASE))
    except re.error:
        return pattern.lower() in text.lower()


def match_clusters_to_profiles(
    clusters: list[dict[str, Any]], catalog: dict[str, dict[str, Any]], cases: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    cases = cases or []
    results: list[dict[str, Any]] = []
    for cluster in clusters:
        texts = [str(x) for x in cluster.get("key_patterns", []) + cluster.get("signal_families", [])]
        best: tuple[int, str, dict[str, Any], list[str]] | None = None
        false_positive = False
        for pid, profile in catalog.items():
            direct = _patterns(profile, ("direct_patterns", "strong_patterns", "patterns", "direct"))
            supporting = _patterns(profile, ("supporting_patterns", "supporting"))
            exclusions = _patterns(profile, ("exclusion_patterns", "excluded_patterns", "exclusion"))
            fp = _patterns(profile, ("false_positive_patterns", "false_positive"))
            if any(regex_match(pattern, text) for pattern in fp for text in texts):
                false_positive = True
            if any(regex_match(pattern, text) for pattern in exclusions for text in texts):
                continue
            direct_hits = [pattern for pattern in direct if any(regex_match(pattern, text) for text in texts)]
            supporting_hits = [pattern for pattern in supporting if any(regex_match(pattern, text) for text in texts)]
            score = len(direct_hits) * 10 + len(supporting_hits) * 3
            if score and (best is None or score > best[0]):
                best = (score, pid, profile, direct_hits + supporting_hits)

        if false_positive and best is None:
            quality = "FALSE_POSITIVE_PATTERN"
            best_pid = ""
            reason = "Matched a registered false-positive pattern without direct profile evidence."
        elif best and best[0] >= 10:
            best_pid = best[1]
            cluster_receivers = {str(x) for x in cluster.get("receivers_seen", []) if x}
            cluster_dates = {str(x)[:10] for x in cluster.get("dates", []) if x}
            exact_duplicate = any(
                case_profile(existing) == best_pid
                and case_receiver(existing) in cluster_receivers
                and case_date(existing) in cluster_dates
                for existing in cases
            )
            quality = "DUPLICATE_OF_KNOWN_CASE" if exact_duplicate else "MATCHES_EXISTING_PROFILE"
            reason = f"Direct pattern match: {best[3][:5]}"
            if exact_duplicate:
                reason += "; same profile/receiver/date already exists in casebook."
        elif best:
            best_pid = best[1]
            quality = "PARTIAL_MATCH"
            reason = f"Supporting-only match: {best[3][:5]}"
        elif not cluster.get("key_patterns"):
            best_pid = ""
            quality = "INSUFFICIENT_EVIDENCE"
            reason = "No usable alert/pattern evidence."
        else:
            best_pid = ""
            quality = "NEW_PROFILE_NEEDED"
            reason = "No registered direct or supporting pattern matched."
        results.append({
            "candidate_id": cluster["candidate_id"],
            "best_profile_match": best_pid,
            "match_quality": quality,
            "missing_fields": [] if cluster.get("receivers_seen") else ["receiver_ids"],
            "new_profile_required": quality == "NEW_PROFILE_NEEDED",
            "reason": reason,
        })
    return results


_LOG_TYPES_BY_FAMILY = {
    "stability": ["procmgr", "ktrap", "stbc_main"],
    "process_service": ["qt_ui", "sgs_handler", "stbc_main"],
    "video_dvr": ["var", "plc", "stbc_main"],
    "guide_epg_popup": ["dp_epg", "qt_ui", "stbc_main"],
    "networking_topology": ["netra", "ipll", "stbc_main"],
    "drm_playback": ["plc", "drm", "stbc_main"],
    "update_firmware": ["im", "config", "stbc_main"],
    "ml_anomaly": ["stbc_main"],
}


def draft_profile(cluster: dict[str, Any]) -> dict[str, Any]:
    profile_id = f"auto_{slugify(cluster.get('suspected_profile', cluster['candidate_id']))}"
    families = cluster.get("signal_families", [])
    core: list[str] = []
    for family in families:
        core.extend(_LOG_TYPES_BY_FAMILY.get(family, ["stbc_main"]))
    core = sorted(set(core))
    patterns = cluster.get("key_patterns", [])
    return {
        "profile_id": profile_id,
        "title": " ".join(profile_id.removeprefix("auto_").split("_")).title(),
        "short_description": f"Proposed profile for recurring {', '.join(families) or 'unknown'} signals.",
        "problem_statement": "Fleet evidence shows a repeatable signal not covered by a registered issue profile.",
        "user_visible_symptom": "Unknown until customer-impact evidence is collected.",
        "component_level_symptom": ", ".join(patterns[:5]),
        "likely_source_components": families,
        "excluded_symptoms": [],
        "owner_hint": "triage_required",
        "severity_default": cluster.get("severity_guess", "medium"),
        "patterns": {
            "direct": patterns,
            "supporting": families,
            "exclusion": [],
            "false_positive": [],
        },
        "required_log_types": {
            "core": core,
            "supplemental": [],
            "minimum_viable": core[:2],
            "receipt_gate": core[:1],
        },
        "topology_rules": {"anchor_only": True},
        "fix_lineage_rules": {"require_confirmed_fix_for_resolution": True},
        "rca_gating": {"root_cause_confirmed_requires_all_gates": True},
        "profile_status": "proposed_for_review",
        "evidence_classification": "HYPOTHESIS",
    }


def extract_candidate_receivers(value: Any) -> list[str]:
    rows = rows_from(value, "candidates", "receivers", "items", "results")
    out: list[str] = []
    for row in rows:
        rid = row.get("receiver_id") or row.get("rxid") or row.get("device_id")
        if rid:
            out.append(str(rid))
    return list(dict.fromkeys(out))


def coverage_summary(value: Any) -> dict[str, Any]:
    rows = rows_from(value, "coverage_matrix", "coverage", "receivers", "items", "results")
    if not rows and isinstance(value, dict):
        present = value.get("required_logs_present")
        return {"rows": [], "required_logs_present": present is True, "covered_receivers": 0}
    covered = 0
    explicit_flags: list[bool] = []
    complete_without_flag = False
    for row in rows:
        status = str(row.get("coverage_status") or row.get("status") or "").lower()
        logs = row.get("log_types") or row.get("available_log_types") or []
        if status not in {"none", "missing", "no_s3_data", "unavailable"} and (logs or status in {"ok", "complete", "covered", "ready", "ready_for_analysis"}):
            covered += 1
        if "required_logs_present" in row:
            explicit_flags.append(row.get("required_logs_present") is True)
        elif status in {"complete", "covered", "ready", "ready_for_analysis"}:
            complete_without_flag = True
    if explicit_flags:
        required = any(explicit_flags)
    else:
        required = complete_without_flag
    return {"rows": rows, "required_logs_present": required, "covered_receivers": covered}


def extract_hash(response: Any, *keys: str) -> str:
    if not isinstance(response, dict):
        return ""
    for key in keys:
        value = response.get(key)
        if value:
            return str(value)
    for nested_key in ("preview", "confirmation", "result"):
        nested = response.get(nested_key)
        if isinstance(nested, dict):
            found = extract_hash(nested, *keys)
            if found:
                return found
    return ""


def extract_identifier(response: Any, *keys: str) -> str:
    return extract_hash(response, *keys)


def blocking_codes(response: Any) -> list[str]:
    if not isinstance(response, dict):
        return ["UNSTRUCTURED_RESPONSE"]
    raw = response.get("blocking_codes") or response.get("blockers") or response.get("missing_gates") or []
    return [str(x) for x in as_list(raw) if x]


def classify_fix_lineage(response: Any) -> str:
    if not isinstance(response, dict):
        return "CANNOT_DETERMINE"
    fixes = as_list(response.get("jira_fix_ids"))
    prs = as_list(response.get("pr_ids"))
    commit = response.get("commit_sha")
    issues = as_list(response.get("jira_issue_ids"))
    if fixes and (prs or commit):
        return "NEWLY_RESOLVED"
    if fixes:
        return "PARTIALLY_ADDRESSED"
    if issues:
        return "FIX_LINEAGE_UPDATED"
    return "STILL_OPEN"


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# Validation-target selector (fairness-aware)
# ---------------------------------------------------------------------------

def _target_priority(target: dict[str, Any]) -> float:
    """Return a float score for a validation target (higher = more urgent).

    Scoring dimensions:
    - Resumed trackers start with a base bonus so they always outrank fresh
      clusters with equivalent signal strength.  The bonus decays by staleness
      so very old (≥14-day) trackers don't crowd out newly-critical signals.
    - Fresh clusters use the cluster's own priority_score from discover_clusters.
    - Severity multipliers give high-confidence / high-severity targets a
      further boost.
    """
    cluster: dict[str, Any] = target.get("cluster") or {}
    resumed = target.get("resumed_tracker") or {}

    base: float = 0.0

    if resumed:
        # Staleness: how many days since this tracker was created.
        raw_created = str(resumed.get("created_at") or resumed.get("requested_at") or "")
        try:
            created_dt = datetime.fromisoformat(raw_created.replace("Z", "+00:00"))
            age_days = max(0, (datetime.now(created_dt.tzinfo) - created_dt).days)
        except (ValueError, TypeError):
            age_days = 0
        # Freshness bonus decays from 30 → 0 over 14 days; floors at 0
        staleness_decay = max(0.0, 30.0 - age_days * (30.0 / 14))
        # Resumed trackers get a guaranteed floor so they're always picked
        # over fresh clusters when slot budget allows.
        base = 100.0 + staleness_decay
    else:
        base = float(cluster.get("priority_score") or 0.0)

    # Severity multiplier applied to both classes
    severity = str(cluster.get("severity_guess") or resumed.get("severity_guess") or "").lower()
    confidence = str(cluster.get("confidence") or resumed.get("confidence") or "").lower()
    severity_mult = {"high": 1.5, "medium": 1.15, "low": 1.0}.get(severity, 1.0)
    if confidence == "high" and not resumed:
        severity_mult = max(severity_mult, 1.15)

    return round(base * severity_mult, 4)


def select_validation_targets(
    targets: list[dict[str, Any]],
    max_total: int,
    resumed_floor: int | None = None,
) -> list[dict[str, Any]]:
    """Select up to *max_total* targets with fairness guarantees for resumed work.

    The selector partitions *targets* into two buckets:

    * **resumed** – targets carrying a ``resumed_tracker`` key (pending log-
      acquisition work from a prior run).
    * **fresh** – all other targets (newly discovered nightly clusters).

    A *resumed_floor* guarantees a minimum number of resumed tracker slots so
    that stale pending work cannot be starved by a large wave of fresh signals.
    The floor defaults to ``min(len(resumed_targets), max(1, max_total // 3))``,
    meaning up to a third of the run budget is reserved for resumption.

    Within each bucket, targets are sorted by ``_target_priority`` descending.
    After filling the resumed floor, remaining slots go to fresh targets in
    priority order, with any unfilled resumed slots released back to fresh.

    Args:
        targets:        Raw ``validation_targets`` list from phases.py phase 5.
        max_total:      Hard cap on returned list length (``max_profiles_per_run``).
        resumed_floor:  Minimum resumed slots.  Pass 0 to disable the guarantee.

    Returns:
        Sorted, deduplicated list of at most *max_total* targets, resumed items
        first (by priority), then fresh items (by priority).
    """
    if max_total <= 0:
        return []

    # Partition
    resumed: list[dict[str, Any]] = []
    fresh: list[dict[str, Any]] = []
    for t in targets:
        if t.get("resumed_tracker"):
            resumed.append(t)
        else:
            fresh.append(t)

    # Sort each bucket by priority descending
    resumed.sort(key=_target_priority, reverse=True)
    fresh.sort(key=_target_priority, reverse=True)

    # Compute floor
    if resumed_floor is None:
        resumed_floor = min(len(resumed), max(1, max_total // 3))
    resumed_floor = min(resumed_floor, max_total)

    # Fill resumed budget
    selected_resumed = resumed[:resumed_floor]
    leftover_resumed = resumed[resumed_floor:]  # unfilled slots → give to fresh

    # Remaining capacity
    fresh_capacity = max_total - len(selected_resumed)

    # Fill fresh slots; if still capacity remains, pull in leftover resumed
    selected_fresh = fresh[:fresh_capacity]
    extra_capacity = fresh_capacity - len(selected_fresh)
    if extra_capacity > 0:
        selected_fresh = selected_fresh + leftover_resumed[:extra_capacity]
    else:
        # No room for leftover resumed; they'll persist for the next run
        pass

    # Deduplicate by (profile_id, receiver_id) keeping first occurrence —
    # resumed entries always precede fresh due to ordering above.
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for t in selected_resumed + selected_fresh:
        cluster: dict[str, Any] = t.get("cluster") or {}
        receivers = cluster.get("receivers_seen") or []
        pid = str(t.get("profile_id") or "")
        # Use first receiver as dedup key (same behaviour as before for multi-rx)
        rx = str(receivers[0]) if receivers else str(t.get("candidate_id") or "")
        key = (pid, rx)
        if key not in seen:
            seen.add(key)
            result.append(t)
        if len(result) >= max_total:
            break

    return result
