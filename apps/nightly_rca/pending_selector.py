"""Fair rotation selector for pending receipt investigations.

Ensures all pending investigations get receipt-checked over multiple cycles
without age-based deletion or implicit TTL. Every physical record is preserved;
only a batch_size subset is selected per run for resume_receipt calls.

Strict global tier exhaustion:
  Tier 0: RETRY_UPLOAD
  Tier 1: due receipt errors
  Tier 2: due never-checked ordinary investigations
  Tier 3: due previously checked ordinary investigations

Process one tier at a time. Round-robin only among entries in the current tier.
A profile with only Tier 3 entries must not consume a slot while Tier 2 entries
exist in another profile.

Tiers 0 and 1 are urgent and always preempt. After filling urgent slots,
the selector processes the highest-priority ordinary tier that has entries.
It does NOT mix tier 2 and tier 3 in the same batch.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


# ── Priority tiers (lower = higher priority) ────────────────────────────────

_TIER_RETRY_UPLOAD = 0
_TIER_RECEIPT_ERROR = 1
_TIER_NEVER_CHECKED = 2
_TIER_DUE_PENDING = 3

_URGENT_TIERS = frozenset({_TIER_RETRY_UPLOAD, _TIER_RECEIPT_ERROR})


def _stable_sort_key(item: dict[str, Any]) -> str:
    """Deterministic tiebreaker from logical key + tracker ID."""
    payload = json.dumps(
        [item.get("logical_case_key", ""), _tracker_id(item)],
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _tracker_id(item: dict[str, Any]) -> str:
    return str(item.get("tracker_id") or item.get("request_id") or "")


def _derive_logical_key(item: dict[str, Any]) -> str:
    """Use existing logical_case_key or derive from receiver+profile+date."""
    if item.get("logical_case_key"):
        return str(item["logical_case_key"])
    receiver = str(item.get("receiver_id") or "")
    profile = str(item.get("profile_id") or item.get("profile") or item.get("issue_profile") or "")
    event_date = str(
        item.get("selected_date")
        or item.get("event_date")
        or item.get("investigation_date")
        or ""
    )[:10]
    return f"{profile}|{receiver}|{event_date}"


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        s = str(value)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _is_eligible(item: dict[str, Any], now: datetime) -> bool:
    """An item is eligible unless it has a future next_check_at."""
    next_check = _parse_iso(item.get("next_check_at"))
    if next_check is None:
        return True  # missing next_check_at -> eligible
    if next_check.tzinfo is None:
        next_check = next_check.replace(tzinfo=timezone.utc)
    return now >= next_check


def _priority_tier(item: dict[str, Any]) -> int:
    """Assign priority tier number based on item status."""
    status = str(item.get("workflow_status") or item.get("status") or "").upper()
    if status == "RETRY_UPLOAD":
        return _TIER_RETRY_UPLOAD
    error_count = int(item.get("receipt_error_count") or 0)
    if error_count > 0:
        return _TIER_RECEIPT_ERROR
    if item.get("last_checked_at") is None:
        return _TIER_NEVER_CHECKED
    return _TIER_DUE_PENDING


def _intra_profile_sort_value(item: dict[str, Any]) -> tuple:
    """Stable ordering within a profile inside one tier.

    Sorts by: oldest created_at/selected_date, logical_case_key, tracker_id.
    """
    created = _parse_iso(item.get("created_at") or item.get("selected_date"))
    created_sort = created.isoformat() if created else ""
    logical_key = _derive_logical_key(item)
    tid = _tracker_id(item)
    return (created_sort, logical_key, tid)


def _round_robin_from_queues(
    profile_queues: dict[str, list[dict[str, Any]]],
    selected: list[dict[str, Any]],
    selected_logical_keys: set[str],
    budget: int,
) -> None:
    """Round-robin across profile queues, mutating selected in place."""
    sorted_profiles = sorted(profile_queues.keys())
    changed = True
    while len(selected) < budget and changed:
        changed = False
        for profile in sorted_profiles:
            if len(selected) >= budget:
                break
            queue = profile_queues[profile]
            while queue:
                candidate = queue.pop(0)
                logical_key = _derive_logical_key(candidate)
                if logical_key in selected_logical_keys:
                    continue
                selected.append(candidate)
                selected_logical_keys.add(logical_key)
                changed = True
                break


def _profile_from_item(item: dict[str, Any]) -> str:
    return str(item.get("profile_id") or item.get("profile") or item.get("issue_profile") or "unknown")


def select_pending_investigations(
    items: list[dict[str, Any]],
    *,
    batch_size: int,
    now: datetime,
) -> list[dict[str, Any]]:
    """Select up to batch_size pending investigations for receipt checking.

    Strict global tier exhaustion:
    - Tiers 0 and 1 (urgent) always fill first via round-robin.
    - After urgent tiers, select from exactly ONE ordinary tier (2 or 3).
    - Tier 2 (never-checked) takes priority over Tier 3 (previously checked).
    - If tier 2 has entries, ONLY tier 2 is selected (no mixing with tier 3).
    - Only one physical tracker per logical investigation per cycle.
    - Deterministic cross-profile round-robin within each tier.

    Returns the selected subset (references to the original dicts).
    """
    batch_size = max(1, min(50, batch_size))

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # Build logical-group checked-state: if ANY physical row in a logical group
    # has been checked, the entire group is considered previously-checked for
    # tier assignment. This prevents un-attempted physical siblings from being
    # classified as never-checked when their logical group was already attempted.
    logical_group_checked: set[str] = set()
    for item in items:
        if item.get("last_checked_at") is not None:
            logical_group_checked.add(_derive_logical_key(item))

    # Separate eligible items by tier
    tier_items: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        if _is_eligible(item, now):
            tier = _priority_tier(item)
            # Override: if this item appears never-checked but its logical group
            # has a checked sibling, demote to DUE_PENDING tier.
            if tier == _TIER_NEVER_CHECKED and _derive_logical_key(item) in logical_group_checked:
                tier = _TIER_DUE_PENDING
            tier_items.setdefault(tier, []).append(item)

    selected: list[dict[str, Any]] = []
    selected_logical_keys: set[str] = set()

    # Phase 1: Process urgent tiers (0, 1) — always fill first
    for urgent_tier in sorted(t for t in tier_items if t in _URGENT_TIERS):
        if len(selected) >= batch_size:
            break
        profile_queues: dict[str, list[dict[str, Any]]] = {}
        for item in tier_items[urgent_tier]:
            profile = _profile_from_item(item)
            profile_queues.setdefault(profile, []).append(item)
        for profile in profile_queues:
            profile_queues[profile].sort(key=_intra_profile_sort_value)
        _round_robin_from_queues(profile_queues, selected, selected_logical_keys, batch_size)

    # Phase 2: Process exactly ONE ordinary tier (2 before 3)
    # Do not mix tier 2 and tier 3 in the same batch.
    ordinary_tiers = sorted(t for t in tier_items if t not in _URGENT_TIERS)
    for ordinary_tier in ordinary_tiers:
        if len(selected) >= batch_size:
            break
        tier_entries = tier_items[ordinary_tier]
        if not tier_entries:
            continue
        # Found the highest-priority ordinary tier with entries
        profile_queues = {}
        for item in tier_entries:
            profile = _profile_from_item(item)
            profile_queues.setdefault(profile, []).append(item)
        for profile in profile_queues:
            profile_queues[profile].sort(key=_intra_profile_sort_value)
        _round_robin_from_queues(profile_queues, selected, selected_logical_keys, batch_size)
        break  # Only process ONE ordinary tier

    return selected


def compute_next_check_at(
    item: dict[str, Any],
    *,
    now: datetime,
    was_error: bool,
) -> str:
    """Compute next_check_at based on receipt result.

    For valid no-log results: now + 24 hours.
    For errors: exponential backoff (1h, 3h, 6h, 24h).
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    if was_error:
        error_count = int(item.get("receipt_error_count") or 0) + 1
        delays = {1: 3600, 2: 10800, 3: 21600}  # 1h, 3h, 6h
        delay_seconds = delays.get(error_count, 86400)  # 4+ -> 24h
    else:
        delay_seconds = 86400  # 24 hours

    from datetime import timedelta
    next_dt = now + timedelta(seconds=delay_seconds)
    return next_dt.isoformat()


def update_metadata_after_attempt(
    item: dict[str, Any],
    *,
    now: datetime,
    was_error: bool,
    selector_generation: int = 0,
) -> None:
    """Update durable metadata on a pending item AFTER a receipt call was attempted.

    Mutates item in place.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    item["last_checked_at"] = now.isoformat()
    item["receipt_check_count"] = int(item.get("receipt_check_count") or 0) + 1
    item["selector_generation"] = selector_generation
    if not item.get("logical_case_key"):
        item["logical_case_key"] = _derive_logical_key(item)

    if was_error:
        item["receipt_error_count"] = int(item.get("receipt_error_count") or 0) + 1
    else:
        item["receipt_error_count"] = 0

    item["next_check_at"] = compute_next_check_at(item, now=now, was_error=was_error)


def selector_metrics(
    all_items: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    *,
    now: datetime,
    batch_size: int,
) -> dict[str, Any]:
    """Produce metrics dict describing the selector run.

    Reports physical and logical unit counts separately to avoid ambiguity.
    Required identities:
      physical_total == selected_physical + duplicate_suppressed_physical
                        + deferred_batch_physical + backoff_physical
      eligible_logical == selected_logical + deferred_batch_logical
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # Physical totals
    physical_total = len(all_items)
    all_logical_keys = set(_derive_logical_key(item) for item in all_items)
    unique_logical_total = len(all_logical_keys)

    # Selected physical/logical
    selected_physical = len(selected)
    selected_logical_keys = set(_derive_logical_key(item) for item in selected)
    selected_logical = len(selected_logical_keys)

    # Eligible: items whose next_check_at is not in the future
    eligible_items = [item for item in all_items if _is_eligible(item, now)]
    eligible_physical = len(eligible_items)
    eligible_logical_keys = set(_derive_logical_key(item) for item in eligible_items)
    eligible_logical = len(eligible_logical_keys)

    # Backoff: items not eligible (next_check_at is in the future)
    backoff_physical = physical_total - eligible_physical

    # Duplicate suppressed: eligible physical rows whose logical key was already
    # selected (so only one physical per logical key gets processed).
    # This also counts eligible physical rows with a logical key that appears on
    # multiple physical rows -- only one is selected per cycle.
    duplicate_suppressed_physical = 0
    selected_tracker_ids = frozenset(
        str(item.get("tracker_id") or item.get("request_id") or "") for item in selected
    )
    eligible_non_selected = [
        item for item in eligible_items
        if str(item.get("tracker_id") or item.get("request_id") or "") not in selected_tracker_ids
    ]
    for item in eligible_non_selected:
        lk = _derive_logical_key(item)
        if lk in selected_logical_keys:
            duplicate_suppressed_physical += 1

    # Deferred by batch limit: eligible physical rows not selected, not duplicate suppressed
    deferred_batch_physical = eligible_physical - selected_physical - duplicate_suppressed_physical

    # Deferred batch logical: eligible logical keys not selected
    deferred_batch_logical = eligible_logical - selected_logical

    # Backoff logical: logical keys where ALL physical rows are in backoff
    eligible_logical_set = eligible_logical_keys
    backoff_logical = unique_logical_total - eligible_logical

    # Never-checked stats
    never_checked_total = sum(1 for item in all_items if item.get("last_checked_at") is None and _is_eligible(item, now))
    never_checked_selected = sum(1 for item in selected if item.get("last_checked_at") is None)

    # Oldest last_checked_at among selected
    oldest_checked = None
    for item in selected:
        lc = _parse_iso(item.get("last_checked_at"))
        if lc is not None:
            if oldest_checked is None or lc < oldest_checked:
                oldest_checked = lc

    return {
        # Physical/logical breakdown
        "pending_physical_total": physical_total,
        "pending_unique_logical_total": unique_logical_total,
        "pending_eligible_physical": eligible_physical,
        "pending_eligible_logical": eligible_logical,
        "pending_selected_physical": selected_physical,
        "pending_selected_logical": selected_logical,
        "pending_backoff_physical": backoff_physical,
        "pending_backoff_logical": backoff_logical,
        "pending_deferred_batch_physical": deferred_batch_physical,
        "pending_deferred_batch_logical": deferred_batch_logical,
        "pending_duplicate_suppressed_physical": duplicate_suppressed_physical,
        # Legacy compatible keys
        "pending_total": physical_total,
        "pending_eligible": eligible_physical,
        "pending_selected": selected_physical,
        "pending_deferred": backoff_physical,
        "pending_unique_logical_keys": unique_logical_total,
        "pending_duplicate_physical_trackers": physical_total - unique_logical_total,
        # Starvation/selector stats
        "selector_never_checked_total": never_checked_total,
        "selector_never_checked_selected": never_checked_selected,
        "selector_oldest_last_checked_at": oldest_checked.isoformat() if oldest_checked else None,
        "selector_batch_size": batch_size,
        "selector_starvation_guard_pass": never_checked_selected > 0 or never_checked_total == 0 or eligible_physical == 0,
    }
