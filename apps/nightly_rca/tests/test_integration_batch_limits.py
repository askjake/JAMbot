"""Integration tests: pending-selector batch-size & validation-target limits."""
from __future__ import annotations
from datetime import datetime, timezone
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from apps.nightly_rca.pending_selector import (
    select_pending_investigations,
    selector_metrics,
)
from apps.nightly_rca.logic import select_validation_targets

_NOW = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)


def _make_pending_items(n: int) -> list[dict]:
    """Generate n distinct eligible pending items (unique logical keys)."""
    items = []
    for i in range(n):
        items.append({
            "tracker_id": f"TRK_{i:04d}",
            "profile_id": f"profile_{i % 5}",
            "receiver_id": f"R{i:05d}",
            "event_date": f"2026-07-{20 + (i % 7):02d}",
            "created_at": "2026-07-20T00:00:00+00:00",
            "last_checked_at": None,
            "next_check_at": None,
        })
    return items


def test_batch_size_25_caps_resume_receipt_attempts():
    """30 eligible pending, batch_size=25 => exactly 25 selected for receipt."""
    items = _make_pending_items(30)
    selected = select_pending_investigations(items, batch_size=25, now=_NOW)
    assert len(selected) == 25, f"Expected 25, got {len(selected)}"


def test_max_profiles_per_run_caps_validation_targets():
    """30 resumed tracker targets, max_profiles_per_run=10 => at most 10 validations."""
    targets = []
    for i in range(30):
        targets.append({
            "candidate_id": f"cand_{i}",
            "profile_id": f"profile_{i}",
            "profile_source": "resumed_tracker",
            "draft": None,
            "resumed_tracker": {"tracker_id": f"TRK_{i:04d}"},
            "cluster": {
                "candidate_id": f"cand_{i}",
                "suspected_profile": f"profile_{i}",
                "alert_name": "test",
                "key_patterns": ["test"],
                "receivers_seen": [f"R{i:05d}"],
                "dates": ["2026-07-25"],
            },
        })
    result = select_validation_targets(targets, max_total=10)
    assert len(result) <= 10, f"Expected <=10, got {len(result)}"


def test_combined_30_eligible_batch25_max10():
    """Combined: 30 eligible pending investigations, batch_size=25,
    max_profiles_per_run=10 => exactly 25 resume_receipt attempts,
    no more than 10 later profile validations."""
    items = _make_pending_items(30)
    selected_pending = select_pending_investigations(items, batch_size=25, now=_NOW)
    assert len(selected_pending) == 25, (
        f"resume_receipt attempts: expected exactly 25, got {len(selected_pending)}"
    )

    # Build validation targets from same items
    targets = []
    for i, item in enumerate(items):
        targets.append({
            "candidate_id": f"cand_{i}",
            "profile_id": item["profile_id"],
            "profile_source": "resumed_tracker",
            "draft": None,
            "resumed_tracker": item,
            "cluster": {
                "candidate_id": f"cand_{i}",
                "suspected_profile": item["profile_id"],
                "alert_name": "test",
                "key_patterns": ["test"],
                "receivers_seen": [item["receiver_id"]],
                "dates": [item.get("event_date", "")],
            },
        })
    validation_result = select_validation_targets(targets, max_total=10)
    assert len(validation_result) <= 10, (
        f"profile validations: expected no more than 10, got {len(validation_result)}"
    )

    metrics = selector_metrics(items, selected_pending, now=_NOW, batch_size=25)
    assert metrics["pending_selected"] == 25
    assert metrics["pending_eligible"] == 30


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
