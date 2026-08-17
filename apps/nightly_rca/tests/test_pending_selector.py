"""Tests for pending_selector.py — fair rotation selection logic."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from apps.nightly_rca.pending_selector import (
    select_pending_investigations,
    compute_next_check_at,
    update_metadata_after_attempt,
    selector_metrics,
    _is_eligible,
    _derive_logical_key,
)


_NOW = datetime(2026, 7, 28, 12, 0, 0, tzinfo=timezone.utc)


def _item(
    tracker_id: str = "T1",
    profile: str = "profA",
    receiver: str = "R001",
    last_checked_at: str | None = None,
    next_check_at: str | None = None,
    receipt_error_count: int = 0,
    workflow_status: str = "",
    created_at: str = "2026-07-27T12:00:00+00:00",
) -> dict:
    return {
        "tracker_id": tracker_id,
        "profile_id": profile,
        "receiver_id": receiver,
        "last_checked_at": last_checked_at,
        "next_check_at": next_check_at,
        "receipt_error_count": receipt_error_count,
        "workflow_status": workflow_status,
        "created_at": created_at,
    }


# ── eligibility ──────────────────────────────────────────────────────────────

def test_eligible_no_next_check():
    assert _is_eligible(_item(next_check_at=None), _NOW)


def test_eligible_past_next_check():
    past = (_NOW - timedelta(hours=1)).isoformat()
    assert _is_eligible(_item(next_check_at=past), _NOW)


def test_ineligible_future_next_check():
    future = (_NOW + timedelta(hours=1)).isoformat()
    assert not _is_eligible(_item(next_check_at=future), _NOW)


def test_eligible_exactly_now():
    assert _is_eligible(_item(next_check_at=_NOW.isoformat()), _NOW)


# ── logical key derivation ───────────────────────────────────────────────────

def test_logical_key_from_explicit():
    item = {"logical_case_key": "prof|rx|2026-07-27"}
    assert _derive_logical_key(item) == "prof|rx|2026-07-27"


def test_logical_key_derived():
    item = {"profile_id": "p1", "receiver_id": "R9", "selected_date": "2026-07-27T10:00Z"}
    assert _derive_logical_key(item) == "p1|R9|2026-07-27"


# ── selection basics ─────────────────────────────────────────────────────────

def test_empty_returns_empty():
    result = select_pending_investigations([], batch_size=5, now=_NOW)
    assert result == []


def test_selects_up_to_batch_size():
    items = [_item(tracker_id=f"T{i}", receiver=f"R{i:03d}") for i in range(10)]
    result = select_pending_investigations(items, batch_size=3, now=_NOW)
    assert len(result) == 3


def test_does_not_exceed_available():
    items = [_item(tracker_id="T1", receiver="R001"), _item(tracker_id="T2", receiver="R002")]
    result = select_pending_investigations(items, batch_size=10, now=_NOW)
    assert len(result) == 2


def test_deferred_items_excluded():
    future = (_NOW + timedelta(hours=2)).isoformat()
    items = [
        _item(tracker_id="T1", receiver="R001", next_check_at=None),
        _item(tracker_id="T2", receiver="R002", next_check_at=future),
    ]
    result = select_pending_investigations(items, batch_size=5, now=_NOW)
    assert len(result) == 1
    assert result[0]["tracker_id"] == "T1"


def test_deduplication_same_logical_key():
    # Two physical trackers for the same logical key: only one selected per cycle
    items = [
        _item(tracker_id="T1", profile="p1", receiver="R1", created_at="2026-07-26T00:00:00+00:00"),
        _item(tracker_id="T2", profile="p1", receiver="R1", created_at="2026-07-27T00:00:00+00:00"),
    ]
    result = select_pending_investigations(items, batch_size=5, now=_NOW)
    assert len(result) == 1


# ── priority ordering ────────────────────────────────────────────────────────

def test_never_checked_selected_over_old_checked():
    never = _item(tracker_id="T_NEVER", receiver="R001", last_checked_at=None)
    old = _item(tracker_id="T_OLD", receiver="R002",
                last_checked_at="2026-07-20T00:00:00+00:00")
    result = select_pending_investigations([old, never], batch_size=1, now=_NOW)
    assert result[0]["tracker_id"] == "T_NEVER"


def test_retry_upload_prioritized():
    normal = _item(tracker_id="T_NORM", receiver="R001")
    retry = _item(tracker_id="T_RETRY", receiver="R002", workflow_status="RETRY_UPLOAD")
    result = select_pending_investigations([normal, retry], batch_size=1, now=_NOW)
    assert result[0]["tracker_id"] == "T_RETRY"


def test_error_count_prioritized_over_clean():
    clean = _item(tracker_id="T_CLEAN", receiver="R001", last_checked_at="2026-07-27T00:00:00+00:00")
    errored = _item(tracker_id="T_ERR", receiver="R002",
                    last_checked_at="2026-07-27T00:00:00+00:00", receipt_error_count=2)
    result = select_pending_investigations([clean, errored], batch_size=1, now=_NOW)
    assert result[0]["tracker_id"] == "T_ERR"


# ── cross-profile round-robin ────────────────────────────────────────────────

def test_cross_profile_fairness():
    # 10 items from profA, 2 items from profB; batch=4 -> should include profB
    items_a = [_item(tracker_id=f"TA{i}", profile="profA", receiver=f"R{100+i}") for i in range(10)]
    items_b = [_item(tracker_id=f"TB{i}", profile="profB", receiver=f"R{200+i}") for i in range(2)]
    result = select_pending_investigations(items_a + items_b, batch_size=4, now=_NOW)
    profiles_selected = {r["profile_id"] for r in result}
    assert "profB" in profiles_selected, "profB should get at least one slot in round-robin"


# ── determinism ──────────────────────────────────────────────────────────────

def test_deterministic_ordering():
    items = [_item(tracker_id=f"T{i}", receiver=f"R{i:03d}") for i in range(20)]
    r1 = [x["tracker_id"] for x in select_pending_investigations(items, batch_size=5, now=_NOW)]
    r2 = [x["tracker_id"] for x in select_pending_investigations(items, batch_size=5, now=_NOW)]
    assert r1 == r2


# ── compute_next_check_at ────────────────────────────────────────────────────

def test_next_check_no_error():
    item = _item()
    result = compute_next_check_at(item, now=_NOW, was_error=False)
    dt = datetime.fromisoformat(result)
    delta = dt - _NOW
    assert abs(delta.total_seconds() - 86400) < 60  # ~24h


def test_next_check_error_backoff():
    item = _item(receipt_error_count=1)
    result = compute_next_check_at(item, now=_NOW, was_error=True)
    dt = datetime.fromisoformat(result)
    delta = dt - _NOW
    # error_count=2 after increment -> 3h
    assert abs(delta.total_seconds() - 10800) < 60


# ── update_metadata_after_attempt ────────────────────────────────────────────

def test_update_sets_last_checked_at():
    item = _item()
    update_metadata_after_attempt(item, now=_NOW, was_error=False)
    assert item["last_checked_at"] == _NOW.isoformat()


def test_update_increments_check_count():
    item = _item()
    update_metadata_after_attempt(item, now=_NOW, was_error=False)
    update_metadata_after_attempt(item, now=_NOW, was_error=False)
    assert item["receipt_check_count"] == 2


def test_update_increments_error_count():
    item = _item()
    update_metadata_after_attempt(item, now=_NOW, was_error=True)
    assert item["receipt_error_count"] == 1


def test_update_resets_error_count_on_success():
    item = _item(receipt_error_count=3)
    update_metadata_after_attempt(item, now=_NOW, was_error=False)
    assert item["receipt_error_count"] == 0


def test_update_sets_logical_key():
    item = _item(profile="pX", receiver="RX", tracker_id="TX")
    update_metadata_after_attempt(item, now=_NOW, was_error=False)
    assert item["logical_case_key"] == "pX|RX|"


# ── selector_metrics ─────────────────────────────────────────────────────────

def test_selector_metrics_keys():
    future = (_NOW + timedelta(hours=2)).isoformat()
    all_items = [
        _item(tracker_id="T1", receiver="R001"),
        _item(tracker_id="T2", receiver="R002", next_check_at=future),
    ]
    selected = [all_items[0]]
    m = selector_metrics(all_items, selected, now=_NOW, batch_size=5)
    assert m["pending_total"] == 2
    assert m["pending_eligible"] == 1
    assert m["pending_deferred"] == 1
    assert m["pending_selected"] == 1
    assert isinstance(m["selector_starvation_guard_pass"], bool)


def test_selector_metrics_never_checked_selected():
    items = [_item(tracker_id="T1", receiver="R001", last_checked_at=None)]
    m = selector_metrics(items, items[:1], now=_NOW, batch_size=5)
    assert m["selector_never_checked_selected"] == 1


# ── starvation guard ─────────────────────────────────────────────────────────

def test_starvation_guard_never_checked_always_selected():
    """Never-checked items must always be in the batch (until batch_size exhausted)."""
    old_checked = [
        _item(tracker_id=f"TC{i}", receiver=f"R{i:03d}",
              last_checked_at="2026-07-01T00:00:00+00:00")
        for i in range(20)
    ]
    never = _item(tracker_id="T_NEVER", receiver="R999", last_checked_at=None)
    items = old_checked + [never]
    result = select_pending_investigations(items, batch_size=5, now=_NOW)
    selected_ids = {r["tracker_id"] for r in result}
    assert "T_NEVER" in selected_ids

# ── duplicate logical-group rotation fairness ────────────────────────────────

def test_duplicate_logical_group_rotation():
    """Verify that duplicate physical rows sharing a logical_case_key are treated
    as one logical group: only one representative is selected per cycle, and the
    entire logical group is excluded from cycle 2 while never-checked logical
    groups remain eligible.

    Fixture:
      - A1 and A2 share logical_case_key 'LOGICAL_A' (distinct tracker rows)
      - B through Z are 25 distinct never-checked logical investigations
      - batch_size=25
    """
    logical_key_a = "LOGICAL_A"

    # Two physical rows sharing the same logical key
    a1 = _item(tracker_id="A1", profile="profA", receiver="R_A1")
    a1["logical_case_key"] = logical_key_a
    a2 = _item(tracker_id="A2", profile="profA", receiver="R_A2")
    a2["logical_case_key"] = logical_key_a

    # 25 distinct never-checked logical investigations (B through Z)
    others = []
    for i in range(25):
        letter = chr(ord("B") + i)
        item = _item(
            tracker_id=f"T_{letter}",
            profile="profA",
            receiver=f"R_{letter}",
        )
        item["logical_case_key"] = f"LOGICAL_{letter}"
        others.append(item)

    all_items = [a1, a2] + others
    physical_count_before = len(all_items)

    # ── Cycle 1 ──────────────────────────────────────────────────────────────
    cycle_1_time = _NOW
    cycle_1_selected = select_pending_investigations(
        all_items, batch_size=25, now=cycle_1_time
    )

    # Extract logical keys selected in cycle 1
    cycle_1_logical_keys = [_derive_logical_key(item) for item in cycle_1_selected]

    # One representative for logical key A selected (not both)
    a_in_cycle_1 = [k for k in cycle_1_logical_keys if k == logical_key_a]
    assert len(a_in_cycle_1) <= 1, (
        f"Both A1 and A2 selected in cycle 1: logical key A appears {len(a_in_cycle_1)} times"
    )

    # No duplicate logical keys within cycle 1
    assert len(cycle_1_logical_keys) == len(set(cycle_1_logical_keys)), (
        "Duplicate logical keys within cycle 1 batch"
    )

    # Simulate completed receipt attempt on the selected representative for A
    # (mark whichever representative was selected as checked)
    for item in cycle_1_selected:
        if _derive_logical_key(item) == logical_key_a:
            update_metadata_after_attempt(
                item, now=cycle_1_time, was_error=False
            )
            break

    # All physical records remain
    assert len(all_items) == physical_count_before, "Physical records were deleted"

    # ── Cycle 2 ──────────────────────────────────────────────────────────────
    # Reference time: cycle 1 + 24h5m (past the 24h next_check_at)
    cycle_2_time = cycle_1_time + timedelta(hours=24, minutes=5)

    cycle_2_selected = select_pending_investigations(
        all_items, batch_size=25, now=cycle_2_time
    )

    cycle_2_logical_keys = [_derive_logical_key(item) for item in cycle_2_selected]

    # Count remaining never-checked logical keys (not counting logical A group)
    never_checked_non_a = set()
    for item in all_items:
        lk = _derive_logical_key(item)
        if lk != logical_key_a and item.get("last_checked_at") is None:
            never_checked_non_a.add(lk)

    # KEY ASSERTION: A must not appear while never-checked work remains
    assert never_checked_non_a, (
        "Test setup error: expected some never-checked non-A logical keys"
    )
    assert logical_key_a not in cycle_2_logical_keys, (
        f"key A appears in cycle 2 — logical group was re-selected despite "
        f"{len(never_checked_non_a)} remaining never-checked logical groups"
    )

    # No duplicate logical keys within cycle 2
    assert len(cycle_2_logical_keys) == len(set(cycle_2_logical_keys)), (
        "Duplicate logical keys within cycle 2 batch"
    )

    # Both physical records A1 and A2 still present
    all_tracker_ids = {item["tracker_id"] for item in all_items}
    assert "A1" in all_tracker_ids, "A1 physical record deleted"
    assert "A2" in all_tracker_ids, "A2 physical record deleted"

    # Physical count unchanged
    assert len(all_items) == physical_count_before, "Physical records were deleted"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
