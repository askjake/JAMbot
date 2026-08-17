"""Tests for pending backlog reporting — physical vs logical breakdown."""
from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from apps.nightly_rca.pending_selector import (
    select_pending_investigations,
    selector_metrics,
    _derive_logical_key,
)

_NOW = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)


def _item(
    tracker_id: str = "T1",
    profile: str = "profA",
    receiver: str = "R001",
    last_checked_at: str | None = None,
    next_check_at: str | None = None,
    receipt_error_count: int = 0,
    workflow_status: str = "",
    created_at: str = "2026-07-27T12:00:00+00:00",
    logical_case_key: str = "",
) -> dict:
    d = {
        "tracker_id": tracker_id,
        "profile_id": profile,
        "receiver_id": receiver,
        "last_checked_at": last_checked_at,
        "next_check_at": next_check_at,
        "receipt_error_count": receipt_error_count,
        "workflow_status": workflow_status,
        "created_at": created_at,
    }
    if logical_case_key:
        d["logical_case_key"] = logical_case_key
    return d


# ── Required identity: physical_total == selected + dup_suppressed + deferred_batch + backoff


def test_physical_total_identity_simple():
    """physical_total == selected + dup_suppressed + deferred_batch + backoff."""
    items = [_item(tracker_id=f"T{i}", receiver=f"R{i:03d}") for i in range(10)]
    selected = select_pending_investigations(items, batch_size=3, now=_NOW)
    m = selector_metrics(items, selected, now=_NOW, batch_size=3)
    assert m["pending_physical_total"] == (
        m["pending_selected_physical"]
        + m["pending_duplicate_suppressed_physical"]
        + m["pending_deferred_batch_physical"]
        + m["pending_backoff_physical"]
    ), f"Identity violated: {m}"


def test_physical_total_identity_with_backoff():
    """Items in backoff (future next_check_at) counted correctly."""
    future = (_NOW + timedelta(hours=2)).isoformat()
    items = [
        _item(tracker_id="T1", receiver="R001"),
        _item(tracker_id="T2", receiver="R002"),
        _item(tracker_id="T3", receiver="R003", next_check_at=future),
        _item(tracker_id="T4", receiver="R004", next_check_at=future),
    ]
    selected = select_pending_investigations(items, batch_size=5, now=_NOW)
    m = selector_metrics(items, selected, now=_NOW, batch_size=5)
    assert m["pending_physical_total"] == 4
    assert m["pending_backoff_physical"] == 2
    assert m["pending_selected_physical"] == 2
    identity = (
        m["pending_selected_physical"]
        + m["pending_duplicate_suppressed_physical"]
        + m["pending_deferred_batch_physical"]
        + m["pending_backoff_physical"]
    )
    assert m["pending_physical_total"] == identity


def test_physical_total_identity_with_duplicates():
    """Duplicate logical keys: only one physical per logical selected."""
    items = [
        _item(tracker_id="T1", profile="p1", receiver="R1", created_at="2026-07-26T00:00:00+00:00"),
        _item(tracker_id="T2", profile="p1", receiver="R1", created_at="2026-07-27T00:00:00+00:00"),
        _item(tracker_id="T3", profile="p2", receiver="R2"),
    ]
    selected = select_pending_investigations(items, batch_size=5, now=_NOW)
    m = selector_metrics(items, selected, now=_NOW, batch_size=5)
    assert m["pending_physical_total"] == 3
    assert m["pending_selected_physical"] == 2  # one from each logical key
    assert m["pending_duplicate_suppressed_physical"] == 1
    identity = (
        m["pending_selected_physical"]
        + m["pending_duplicate_suppressed_physical"]
        + m["pending_deferred_batch_physical"]
        + m["pending_backoff_physical"]
    )
    assert m["pending_physical_total"] == identity


def test_physical_total_identity_with_batch_limit():
    """Batch limit causes deferred items."""
    items = [_item(tracker_id=f"T{i}", receiver=f"R{i:03d}") for i in range(10)]
    selected = select_pending_investigations(items, batch_size=3, now=_NOW)
    m = selector_metrics(items, selected, now=_NOW, batch_size=3)
    assert m["pending_selected_physical"] == 3
    assert m["pending_deferred_batch_physical"] == 7
    identity = (
        m["pending_selected_physical"]
        + m["pending_duplicate_suppressed_physical"]
        + m["pending_deferred_batch_physical"]
        + m["pending_backoff_physical"]
    )
    assert m["pending_physical_total"] == identity


def test_physical_total_identity_mixed():
    """Combined backoff, duplicates, and batch limit."""
    future = (_NOW + timedelta(hours=2)).isoformat()
    items = [
        # 2 items with same logical key (profA|R001|2026-07-27)
        _item(tracker_id="T1", profile="profA", receiver="R001", created_at="2026-07-26T00:00:00+00:00"),
        _item(tracker_id="T2", profile="profA", receiver="R001", created_at="2026-07-27T00:00:00+00:00"),
        # 3 items with unique logical keys
        _item(tracker_id="T3", profile="profB", receiver="R002"),
        _item(tracker_id="T4", profile="profC", receiver="R003"),
        _item(tracker_id="T5", profile="profD", receiver="R004"),
        # 2 items in backoff
        _item(tracker_id="T6", profile="profE", receiver="R005", next_check_at=future),
        _item(tracker_id="T7", profile="profF", receiver="R006", next_check_at=future),
    ]
    selected = select_pending_investigations(items, batch_size=3, now=_NOW)
    m = selector_metrics(items, selected, now=_NOW, batch_size=3)
    assert m["pending_physical_total"] == 7
    assert m["pending_backoff_physical"] == 2
    identity = (
        m["pending_selected_physical"]
        + m["pending_duplicate_suppressed_physical"]
        + m["pending_deferred_batch_physical"]
        + m["pending_backoff_physical"]
    )
    assert m["pending_physical_total"] == identity, f"Identity violated: {m}"


# ── Required identity: eligible_logical == selected_logical + deferred_batch_logical


def test_eligible_logical_identity():
    """eligible_logical == selected_logical + deferred_batch_logical."""
    items = [_item(tracker_id=f"T{i}", receiver=f"R{i:03d}") for i in range(8)]
    selected = select_pending_investigations(items, batch_size=3, now=_NOW)
    m = selector_metrics(items, selected, now=_NOW, batch_size=3)
    assert m["pending_eligible_logical"] == (
        m["pending_selected_logical"] + m["pending_deferred_batch_logical"]
    ), f"Logical identity violated: {m}"


def test_eligible_logical_identity_with_backoff():
    future = (_NOW + timedelta(hours=2)).isoformat()
    items = [
        _item(tracker_id="T1", receiver="R001"),
        _item(tracker_id="T2", receiver="R002"),
        _item(tracker_id="T3", receiver="R003", next_check_at=future),
    ]
    selected = select_pending_investigations(items, batch_size=1, now=_NOW)
    m = selector_metrics(items, selected, now=_NOW, batch_size=1)
    assert m["pending_eligible_logical"] == (
        m["pending_selected_logical"] + m["pending_deferred_batch_logical"]
    )
    assert m["pending_backoff_logical"] == 1


# ── Logical counts


def test_logical_counts_distinct_from_physical():
    """Logical counts deduplicate by logical_case_key."""
    items = [
        _item(tracker_id="T1", profile="p1", receiver="R1"),
        _item(tracker_id="T2", profile="p1", receiver="R1"),  # same logical key
        _item(tracker_id="T3", profile="p2", receiver="R2"),
    ]
    selected = select_pending_investigations(items, batch_size=5, now=_NOW)
    m = selector_metrics(items, selected, now=_NOW, batch_size=5)
    assert m["pending_physical_total"] == 3
    assert m["pending_unique_logical_total"] == 2
    assert m["pending_selected_logical"] == 2


# ── Report and notification wording tests


def test_report_includes_pending_breakdown():
    """FINAL_REPORT.md includes the pending backlog section."""
    from apps.nightly_rca.report import render_report
    from apps.nightly_rca.state import RunState

    state = RunState(
        schema_version=1, run_id="test", mode="commit", role="nightly_rca",
        started_at="2026-07-29T00:00:00Z", completed_at="2026-07-29T01:00:00Z",
        status="COMPLETE",
    )
    state.data["final"] = {
        "executive_verdict": "PASS",
        "final_numbers": {"pending_investigations": 5},
        "remaining_blockers": [],
    }
    state.data["pending_selector_metrics"] = {
        "pending_physical_total": 50,
        "pending_unique_logical_total": 40,
        "pending_selected_physical": 25,
        "pending_selected_logical": 25,
        "pending_eligible_physical": 40,
        "pending_eligible_logical": 35,
        "pending_backoff_physical": 10,
        "pending_backoff_logical": 5,
        "pending_deferred_batch_physical": 10,
        "pending_deferred_batch_logical": 10,
        "pending_duplicate_suppressed_physical": 5,
    }
    report = render_report(state)
    assert "Pending investigation backlog" in report
    assert "25 logical investigations selected" in report
    assert "10 eligible logical investigations were deferred by the batch limit" in report
    assert "10 physical tracker rows are in backoff" in report
    assert "5 additional physical tracker rows represent duplicate logical investigations" in report


def test_notification_uses_accurate_wording():
    """Google Chat notification avoids ambiguous 'N remain pending' wording."""
    from unittest.mock import MagicMock
    from apps.nightly_rca.notify import summarize
    from apps.nightly_rca.state import RunState, RunStore

    state = RunState(
        schema_version=1, run_id="test", mode="commit", role="nightly_rca",
        started_at="2026-07-29T00:00:00Z", completed_at="2026-07-29T01:00:00Z",
        status="COMPLETE",
    )
    state.completed_phases = ["0", "1"]
    state.data["final"] = {
        "executive_verdict": "DEFERRED",
        "final_numbers": {"pending_investigations": 50},
        "remaining_blockers": [],
    }
    state.data["pending_investigations"] = [{"tracker_id": f"T{i}"} for i in range(50)]
    state.data["pending_selector_metrics"] = {
        "pending_physical_total": 50,
        "pending_selected_logical": 25,
        "pending_deferred_batch_logical": 10,
        "pending_backoff_physical": 8,
        "pending_duplicate_suppressed_physical": 7,
        "pending_selected": 25,
    }
    store = MagicMock(spec=RunStore)
    store.run_dir = MagicMock()
    store.run_dir.__truediv__ = lambda self, other: MagicMock(exists=lambda: False)

    text = summarize(state, store)
    # Must NOT contain the ambiguous wording
    assert "upload tracker(s) remain pending" not in text
    # Must contain accurate wording
    assert "25 logical investigations selected" in text
    assert "10 eligible logical investigations were deferred by the batch limit" in text
    assert "8 physical tracker rows are in backoff" in text
    assert "7 additional physical tracker rows represent duplicate logical investigations" in text


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
