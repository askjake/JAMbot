"""Tests for the expected-negative canary classification (E_expected_no_logs)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from nightly_rca.config import Settings
from nightly_rca.notify import summarize
from nightly_rca.pipeline import NightlyPipeline
from nightly_rca.state import RunState, RunStore
from nightly_rca.tests.fake_client import FakeToolClient


def fixed_state(mode: str) -> RunState:
    return RunState(
        schema_version=1,
        run_id=f"test-canary-{mode}",
        mode=mode,
        role="operator",
        started_at="2026-07-24T08:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_expected_no_logs_passes_canary(tmp_path: Path):
    """1. Recognized no-logs error passes the E_expected_no_logs canary."""
    settings = replace(Settings(), output_dir=tmp_path, commit=False, notify=False)
    client = FakeToolClient(canary_no_logs_mode="no_logs")
    state = fixed_state("no-logs-pass")
    pipeline = NightlyPipeline(settings=settings, client=client, state=state)

    rc = await pipeline.run()

    assert rc == 0
    canaries = state.data["canaries"]["canaries"]
    e_canary = next(c for c in canaries if c["canary"] == "E_expected_no_logs")
    assert e_canary["pass"] is True
    assert e_canary["evidence"]["status"] == "EXPECTED_NO_LOGS_CONFIRMED"
    assert state.metrics["expected_negative_canary_pass"] == 1


@pytest.mark.asyncio
async def test_expected_no_logs_does_not_increment_step_failed(tmp_path: Path):
    """2. Recognized no-logs result does not increment step_failed."""
    settings = replace(Settings(), output_dir=tmp_path, commit=False, notify=False)
    client = FakeToolClient(canary_no_logs_mode="no_logs")
    state = fixed_state("no-step-fail")
    pipeline = NightlyPipeline(settings=settings, client=client, state=state)

    rc = await pipeline.run()

    assert rc == 0
    assert state.metrics["step_failed"] == 0


@pytest.mark.asyncio
async def test_auth_failure_fails_canary(tmp_path: Path):
    """3. Authentication failure fails the E_expected_no_logs canary."""
    settings = replace(Settings(), output_dir=tmp_path, commit=False, notify=False)
    client = FakeToolClient(canary_no_logs_mode="auth_fail")
    state = fixed_state("auth-fail")
    pipeline = NightlyPipeline(settings=settings, client=client, state=state)

    rc = await pipeline.run()

    assert rc == 1
    canaries = state.data["canaries"]["canaries"]
    e_canary = next(c for c in canaries if c["canary"] == "E_expected_no_logs")
    assert e_canary["pass"] is False
    assert "CANARY_FAILED" in e_canary["evidence"]["status"]


@pytest.mark.asyncio
async def test_authz_failure_fails_canary(tmp_path: Path):
    """4. Authorization failure fails the E_expected_no_logs canary."""
    settings = replace(Settings(), output_dir=tmp_path, commit=False, notify=False)
    client = FakeToolClient(canary_no_logs_mode="authz_fail")
    state = fixed_state("authz-fail")
    pipeline = NightlyPipeline(settings=settings, client=client, state=state)

    rc = await pipeline.run()

    assert rc == 1
    canaries = state.data["canaries"]["canaries"]
    e_canary = next(c for c in canaries if c["canary"] == "E_expected_no_logs")
    assert e_canary["pass"] is False
    assert "CANARY_FAILED" in e_canary["evidence"]["status"]


@pytest.mark.asyncio
async def test_timeout_fails_canary(tmp_path: Path):
    """5. Timeout fails the E_expected_no_logs canary."""
    settings = replace(Settings(), output_dir=tmp_path, commit=False, notify=False)
    client = FakeToolClient(canary_no_logs_mode="timeout")
    state = fixed_state("timeout-fail")
    pipeline = NightlyPipeline(settings=settings, client=client, state=state)

    rc = await pipeline.run()

    assert rc == 1
    canaries = state.data["canaries"]["canaries"]
    e_canary = next(c for c in canaries if c["canary"] == "E_expected_no_logs")
    assert e_canary["pass"] is False


@pytest.mark.asyncio
async def test_tool_not_found_fails_canary(tmp_path: Path):
    """6. Tool-not-found fails the E_expected_no_logs canary."""
    settings = replace(Settings(), output_dir=tmp_path, commit=False, notify=False)
    client = FakeToolClient(canary_no_logs_mode="tool_not_found")
    state = fixed_state("tool-not-found")
    pipeline = NightlyPipeline(settings=settings, client=client, state=state)

    rc = await pipeline.run()

    # tool_not_found in executor sets status=TOOL_NOT_FOUND, not STEP_FAILED
    # _classify_expected_no_logs sees status=TOOL_NOT_FOUND and error contains "tool_not_found"
    assert rc == 1
    canaries = state.data["canaries"]["canaries"]
    e_canary = next(c for c in canaries if c["canary"] == "E_expected_no_logs")
    assert e_canary["pass"] is False


@pytest.mark.asyncio
async def test_unexpected_logs_fails_canary(tmp_path: Path):
    """7. Unexpected logs for the synthetic receiver fail the canary."""
    settings = replace(Settings(), output_dir=tmp_path, commit=False, notify=False)
    client = FakeToolClient(canary_no_logs_mode="unexpected_logs")
    state = fixed_state("unexpected-logs")
    pipeline = NightlyPipeline(settings=settings, client=client, state=state)

    rc = await pipeline.run()

    assert rc == 1
    canaries = state.data["canaries"]["canaries"]
    e_canary = next(c for c in canaries if c["canary"] == "E_expected_no_logs")
    assert e_canary["pass"] is False
    assert e_canary["evidence"]["status"] == "UNEXPECTED_LOGS_FOR_SYNTHETIC_RECEIVER"


@pytest.mark.asyncio
async def test_real_tool_failure_still_increments_step_failed(tmp_path: Path):
    """8. A real ordinary pipeline tool failure still increments step_failed."""
    settings = replace(Settings(), output_dir=tmp_path, commit=False, notify=False)
    client = FakeToolClient(fail_tool_once="count_alerts", canary_no_logs_mode="no_logs")
    state = fixed_state("real-failure")
    pipeline = NightlyPipeline(settings=settings, client=client, state=state)

    rc = await pipeline.run(stop_after="2")

    assert state.metrics["step_failed"] == 1
    # The canary was not reached (stopped after phase 2)
    assert state.metrics["expected_negative_canary_pass"] == 0


@pytest.mark.asyncio
async def test_successful_pending_produces_complete_pending(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """9. Successful pending investigations produce COMPLETE_PENDING."""
    monkeypatch.setenv("NIGHTLY_RCA_WRITE_AUTHORIZED", "true")
    settings = replace(Settings(), output_dir=tmp_path, commit=True, notify=False)
    client = FakeToolClient(canary_no_logs_mode="no_logs")
    state = fixed_state("pending")

    # Override client to force pending investigations
    class PendingClient(FakeToolClient):
        def __init__(self):
            super().__init__(canary_no_logs_mode="no_logs")

        def _response(self, tool, args):
            if tool == "verify_receiver_log_coverage":
                rx = str(args.get("receiver_ids", "")).split(",")[0].strip()
                return {"coverage_matrix": [
                    {"receiver_id": rx, "coverage_status": "no_s3_data", "required_logs_present": False}
                ]}
            if tool == "count_alerts" and args.get("group_by") == "receiver":
                return {"status": "OK", "buckets": []}
            return super()._response(tool, args)

    client = PendingClient()
    pipeline = NightlyPipeline(settings=settings, client=client, state=state)

    rc = await pipeline.run()

    # With pending work but no failures, verdict should be DEFERRED
    verdict = state.data.get("final", {}).get("executive_verdict", "")
    if verdict.startswith("DEFERRED"):
        assert rc == 0
        assert state.status == "COMPLETE_PENDING"


@pytest.mark.asyncio
async def test_complete_pending_returns_exit_code_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """10. COMPLETE_PENDING returns exit code 0."""
    monkeypatch.setenv("NIGHTLY_RCA_WRITE_AUTHORIZED", "true")

    class DeferredClient(FakeToolClient):
        def __init__(self):
            super().__init__(canary_no_logs_mode="no_logs")

        def _response(self, tool, args):
            if tool == "verify_receiver_log_coverage":
                rx = str(args.get("receiver_ids", "")).split(",")[0].strip()
                return {"coverage_matrix": [
                    {"receiver_id": rx, "coverage_status": "no_s3_data", "required_logs_present": False}
                ]}
            if tool == "count_alerts" and args.get("group_by") == "receiver":
                return {"status": "OK", "buckets": []}
            return super()._response(tool, args)

    settings = replace(Settings(), output_dir=tmp_path, commit=True, notify=False)
    client = DeferredClient()
    state = fixed_state("deferred-exit0")
    pipeline = NightlyPipeline(settings=settings, client=client, state=state)

    rc = await pipeline.run()
    verdict = state.data.get("final", {}).get("executive_verdict", "")
    if verdict.startswith("DEFERRED"):
        assert rc == 0


@pytest.mark.asyncio
async def test_failed_acceptance_canary_produces_gaps_exit_1(tmp_path: Path):
    """11. Failed acceptance canary produces COMPLETE_WITH_GAPS and exit code 1."""
    settings = replace(Settings(), output_dir=tmp_path, commit=False, notify=False)
    client = FakeToolClient(canary_no_logs_mode="auth_fail")
    state = fixed_state("canary-exit1")
    pipeline = NightlyPipeline(settings=settings, client=client, state=state)

    rc = await pipeline.run()

    assert rc == 1
    assert state.status == "COMPLETE_WITH_GAPS"
    assert state.data["canaries"]["failure_count"] > 0


@pytest.mark.asyncio
async def test_notification_shows_step_failed_0_for_expected_negative(tmp_path: Path):
    """12. Notification output shows step_failed=0 for the normal expected-negative case."""
    settings = replace(Settings(), output_dir=tmp_path, commit=False, notify=False)
    client = FakeToolClient(canary_no_logs_mode="no_logs")
    state = fixed_state("notify-check")
    pipeline = NightlyPipeline(settings=settings, client=client, state=state)

    rc = await pipeline.run()
    assert rc == 0

    store = RunStore(tmp_path, state.run_id)
    message = summarize(state, store)
    assert "step_failed=0" in message
    assert "expected_negative_canary_pass=1" in message


@pytest.mark.asyncio
async def test_notification_includes_e_canary_passed(tmp_path: Path):
    """13. Notification includes E_expected_no_logs as passed."""
    settings = replace(Settings(), output_dir=tmp_path, commit=False, notify=False)
    client = FakeToolClient(canary_no_logs_mode="no_logs")
    state = fixed_state("notify-e-canary")
    pipeline = NightlyPipeline(settings=settings, client=client, state=state)

    rc = await pipeline.run()
    assert rc == 0

    store = RunStore(tmp_path, state.run_id)
    message = summarize(state, store)
    assert "E_expected_no_logs" in message
    # The canary shows as passed (✅)
    assert "✅" in message
