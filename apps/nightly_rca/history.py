"""Bounded cross-run learning signals; detector/profile mutations remain human-gated."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


_COMPLETED_STATUSES = {"COMPLETE", "COMPLETE_PENDING", "COMPLETE_WITH_GAPS"}


def load_history_signals(output_dir: Path, current_run_id: str, limit: int) -> dict[str, dict[str, int]]:
    """Aggregate recurrence and validation outcomes from a bounded number of complete runs."""
    runs_dir = Path(output_dir) / "runs"
    recurrence: Counter[str] = Counter()
    validation_passes: Counter[str] = Counter()
    validation_failures: Counter[str] = Counter()
    if not runs_dir.exists():
        return {
            "recurrence": {}, "validation_passes": {}, "validation_failures": {},
        }

    seen = 0
    for state_path in sorted(runs_dir.glob("*/state.json"), reverse=True):
        if state_path.parent.name == current_run_id:
            continue
        try:
            with open(state_path, encoding="utf-8") as fh:
                state = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("status") not in _COMPLETED_STATUSES:
            continue
        data = state.get("data", {})
        for cluster in data.get("candidate_clusters", []):
            profile = cluster.get("suspected_profile")
            if profile:
                recurrence[str(profile)] += 1
        for validation in data.get("profile_validations", []):
            cluster = validation.get("cluster") if isinstance(validation.get("cluster"), dict) else {}
            signal = validation.get("suspected_profile") or cluster.get("suspected_profile")
            if not signal:
                profile_id = str(validation.get("profile_id") or "")
                signal = profile_id.removeprefix("auto_") if profile_id else ""
            if not signal:
                continue
            classification = str(validation.get("classification") or "")
            if classification == "PROFILE_VALIDATION_PASS":
                validation_passes[str(signal)] += 1
            elif classification in {
                "PROFILE_VALIDATION_FAIL", "PROFILE_VALIDATION_PARTIAL",
                "PROFILE_NEEDS_MORE_DATA", "PROFILE_DUPLICATE",
            }:
                validation_failures[str(signal)] += 1
        seen += 1
        if seen >= limit:
            break
    return {
        "recurrence": dict(recurrence),
        "validation_passes": dict(validation_passes),
        "validation_failures": dict(validation_failures),
    }


def load_history_counts(output_dir: Path, current_run_id: str, limit: int) -> dict[str, int]:
    """Compatibility helper for callers that only need recurrence counts."""
    return load_history_signals(output_dir, current_run_id, limit)["recurrence"]


def learning_recommendations(data: dict[str, Any]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for cluster in data.get("candidate_clusters", []):
        recurrence = int(cluster.get("recurrence_nights", 0))
        passes = int(cluster.get("prior_validation_passes", 0))
        failures = int(cluster.get("prior_validation_failures", 0))
        if failures >= 2:
            recommendations.append({
                "type": "FALSE_POSITIVE_OR_COVERAGE_REVIEW",
                "candidate_id": cluster.get("candidate_id"),
                "reason": f"The signal failed or lacked coverage in {failures} prior validation runs.",
                "action": "Review detector specificity, exclusions, and required-log mapping before retrying registration.",
            })
        elif recurrence >= 3 and passes >= 2 and cluster.get("new_profile_needed"):
            recommendations.append({
                "type": "STABLE_PROFILE_CANDIDATE",
                "candidate_id": cluster.get("candidate_id"),
                "reason": "The unmatched signal recurred and passed validation repeatedly.",
                "action": "Prioritize human review of the proposed profile and its false-positive exclusions.",
            })
        elif recurrence >= 3 and cluster.get("new_profile_needed"):
            recommendations.append({
                "type": "PROFILE_REVIEW_PRIORITY",
                "candidate_id": cluster.get("candidate_id"),
                "reason": "Unmatched cluster recurred on at least three prior complete runs.",
                "action": "Human-review proposed profile patterns and exclusions.",
            })
    for validation in data.get("profile_validations", []):
        if validation.get("classification") in {"PROFILE_VALIDATION_PARTIAL", "PROFILE_NEEDS_MORE_DATA"}:
            recommendations.append({
                "type": "COVERAGE_GAP",
                "profile_id": validation.get("profile_id"),
                "reason": validation.get("reason", "Insufficient validation evidence."),
                "action": "Improve upload coverage before profile registration or RCA promotion.",
            })
    return recommendations
