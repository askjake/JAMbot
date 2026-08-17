"""
Tests for Phase 5 pending-tracker reconciliation isolation.

Bug fixed: When pending_batch_size >= max_profiles_per_run, selected physical
trackers were reconciled once in the dedicated pending pass (Step 3) AND again
in the downstream validation loop (Step 7). This produced duplicate step IDs
like ``5.3.resume_receipt`` appearing twice in state.steps, causing the executor
to write conflicting records and making dry-run audit reports non-deterministic.

The fix: _reconcile_upload_tracker is called ONLY from the pending pass (Step 3)
using step IDs ``5.P{nn:02d}.resume_receipt``. Results are cached in
``_reconciliation_cache``. The validation loop consumes the cache — it never
calls _reconcile_upload_tracker directly.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from nightly_rca.config import Settings
from nightly_rca.pipeline import NightlyPipeline
from nightly_rca.state import RunState, RunStore
from nightly_rca.tests.fake_client import FakeToolClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(run_id: str) -> RunState:
    return RunState(
        schema_version=1,
        run_id=run_id,
        mode="test",
        role="operator",
        started_at="2026-07-21T08:00:00+00:00",
    )


def _make_pending_item(i: int) -> dict[str, Any]:
    return {
        "tracker_id": f"TRK_{i:04d}",
        "receiver_id": f"R{i:05d}",
        "profile_id": f"profile_{i % 3}",
        "candidate_id": f"cand_{i:04d}",
        "alert_name": "test_alert",
        "event_date": "2026-07-20",
        "workflow_status": "WAITING_FOR_LOGS",
        "created_at": "2026-07-20T00:00:00+00:00",
        "suspected_profile": f"profile_{i % 3}",
    }


class AlwaysWaitingClient(FakeToolClient):
    """All upload trackers remain WAITING — no logs ever arrive."""

    def _response(self, tool: str, args: dict[str, Any]) -> Any:
        if tool == "update_upload_tracker_from_s3":
            tracker_id = str(args.get("tracker_id") or "")
            return {
                "ok": True,
                "result": "UPLOAD_TRACKER_UPDATED",
                "tracker_id": tracker_id,
                "receipt_status": "upload_requested_pending_receipt",
                "workflow_status": "WAITING_FOR_LOGS",
                "landed_log_types": [],
                "missing_log_types": ["procmgr"],
                "dry_run": args.get("dry_run", False),
                "write_performed": not args.get("dry_run", False),
            }
        return super()._response(tool, args)


# ---------------------------------------------------------------------------
# Core invariant: resume_receipt is called exactly once per selected tracker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resume_receipt_called_exactly_once_per_selected_tracker(tmp_path: Path):
    """Each selected pending tracker must produce exactly one resume_receipt call.

    With pending_batch_size=5 and 5 pending items, the pending pass (Step 3)
    issues exactly 5 update_upload_tracker_from_s3 calls.  No additional
    reconciliation calls must appear anywhere in state.steps.
    """
    n_pending = 5
    settings = replace(
        Settings(), output_dir=tmp_path, commit=False, notify=False,
        pending_batch_size=n_pending, max_profiles_per_run=10,
    )
    client = AlwaysWaitingClient()
    state = _state("no-dup-receipt")

    # Seed pending investigations
    store = RunStore(tmp_path, state.run_id)
    store.save_pending_investigations([_make_pending_item(i) for i in range(n_pending)])
    state.data["pending_investigations"] = store.load_pending_investigations()

    rc = await NightlyPipeline(settings=settings, client=client, state=state).run(stop_after="5")
    assert rc == 0, f"Pipeline returned non-zero exit code: {rc}"

    tracker_calls = [
        s for s in state.steps
        if s.get("tool") == "update_upload_tracker_from_s3"
    ]
    assert len(tracker_calls) == n_pending, (
        f"Expected exactly {n_pending} update_upload_tracker_from_s3 calls, "
        f"got {len(tracker_calls)}: {[s.get('step') for s in tracker_calls]}"
    )


@pytest.mark.asyncio
async def test_resume_receipt_step_ids_use_pending_namespace(tmp_path: Path):
    """reconcile_upload_tracker step IDs must use the 5.P{nn:02d} namespace.

    Before the fix, step IDs were ``5.{idx}.resume_receipt`` where idx came
    from the validation loop counter — colliding with Step 3 IDs.
    After the fix, Step 3 produces ``5.P{nn:02d}.resume_receipt`` and the
    validation loop NEVER calls update_upload_tracker_from_s3.
    """
    n_pending = 3
    settings = replace(
        Settings(), output_dir=tmp_path, commit=False, notify=False,
        pending_batch_size=n_pending, max_profiles_per_run=10,
    )
    client = AlwaysWaitingClient()
    state = _state("pending-namespace")

    store = RunStore(tmp_path, state.run_id)
    store.save_pending_investigations([_make_pending_item(i) for i in range(n_pending)])
    state.data["pending_investigations"] = store.load_pending_investigations()

    rc = await NightlyPipeline(settings=settings, client=client, state=state).run(stop_after="5")
    assert rc == 0

    tracker_steps = [
        s for s in state.steps
        if s.get("tool") == "update_upload_tracker_from_s3"
    ]
    # All step IDs must start with "5.P" (pending pass namespace)
    for s in tracker_steps:
        step_id = str(s.get("step") or "")
        assert step_id.startswith("5.P"), (
            f"Expected pending-pass step ID (5.P...) but got: {step_id!r}. "
            "The validation loop must not call _reconcile_upload_tracker."
        )

    # Step IDs must be unique (no duplicates)
    step_ids = [s.get("step") for s in tracker_steps]
    assert len(step_ids) == len(set(step_ids)), (
        f"Duplicate step IDs detected: {step_ids}"
    )


@pytest.mark.asyncio
async def test_no_duplicate_step_ids_with_large_pending_batch(tmp_path: Path):
    """30 pending trackers, batch_size=25: no duplicate step IDs anywhere in phase 5.

    This reproduces the original bug scenario: with a large pending batch, the
    validation loop would re-reconcile trackers at idx 1-10 producing duplicate
    step IDs ``5.1.resume_receipt`` through ``5.10.resume_receipt``.
    """
    n_pending = 30
    settings = replace(
        Settings(), output_dir=tmp_path, commit=False, notify=False,
        pending_batch_size=25, max_profiles_per_run=10,
    )
    client = AlwaysWaitingClient()
    state = _state("no-dup-large-batch")

    store = RunStore(tmp_path, state.run_id)
    store.save_pending_investigations([_make_pending_item(i) for i in range(n_pending)])
    state.data["pending_investigations"] = store.load_pending_investigations()

    rc = await NightlyPipeline(settings=settings, client=client, state=state).run(stop_after="5")
    assert rc == 0

    # Collect all phase-5 step IDs
    phase5_steps = [
        s for s in state.steps
        if str(s.get("phase") or s.get("step") or "").startswith("5")
    ]
    step_ids = [s.get("step") for s in phase5_steps]
    duplicates = [sid for sid in set(step_ids) if step_ids.count(sid) > 1]
    assert not duplicates, (
        f"Duplicate phase-5 step IDs detected (was the fix reverted?): {duplicates}"
    )

    # update_upload_tracker_from_s3 must be called exactly pending_batch_size times
    tracker_calls = [s for s in state.steps if s.get("tool") == "update_upload_tracker_from_s3"]
    assert len(tracker_calls) == 25, (
        f"Expected exactly 25 tracker reconciliation calls, got {len(tracker_calls)}: "
        f"{[s.get('step') for s in tracker_calls]}"
    )


@pytest.mark.asyncio
async def test_reconciliation_cache_prevents_extra_tracker_calls_with_mixed_pending_and_anomalies(
    tmp_path: Path,
):
    """When pending trackers coexist with fresh anomaly candidates, reconciliation
    must not exceed pending_batch_size even if those trackers appear in validation_targets.

    This is the scenario from the original bug: a large pending batch with
    batch_size=5 and max_profiles_per_run=5 would previously call
    update_upload_tracker_from_s3 for the SAME trackers in BOTH Step 3 and Step 7,
    producing duplicate step IDs and double the expected call count.
    """
    n_pending = 5
    settings = replace(
        Settings(), output_dir=tmp_path, commit=False, notify=False,
        pending_batch_size=n_pending, max_profiles_per_run=5,
    )
    client = AlwaysWaitingClient()
    state = _state("no-double-call")

    store = RunStore(tmp_path, state.run_id)
    store.save_pending_investigations([_make_pending_item(i) for i in range(n_pending)])
    state.data["pending_investigations"] = store.load_pending_investigations()

    rc = await NightlyPipeline(settings=settings, client=client, state=state).run(stop_after="5")
    assert rc == 0, f"Pipeline exited with non-zero code {rc}"

    tracker_calls = [
        s for s in state.steps
        if s.get("tool") == "update_upload_tracker_from_s3"
    ]
    # Must be EXACTLY n_pending — not 2×n_pending (which the bug produced)
    assert len(tracker_calls) == n_pending, (
        f"Expected exactly {n_pending} tracker reconciliation calls, got {len(tracker_calls)}. "
        f"Step IDs: {[s.get('step') for s in tracker_calls]}"
    )

    # Step IDs in phase 5 must be unique (filter to phase-5 steps only)
    phase5_step_ids = [
        s.get("step") for s in state.steps
        if str(s.get("step") or "").startswith("5.")
    ]
    duplicates = [sid for sid in set(phase5_step_ids) if phase5_step_ids.count(sid) > 1]
    assert not duplicates, f"Duplicate phase-5 step IDs: {duplicates}"


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-v"]))
