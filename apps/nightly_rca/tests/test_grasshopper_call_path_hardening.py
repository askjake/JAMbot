from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from nightly_rca.config import Settings
from nightly_rca.executor import SerialExecutor
from nightly_rca.grasshopper_contract import AcquisitionState
from nightly_rca.phases import PhaseRunner
from nightly_rca.state import RunState, RunStore
from nightly_rca.tests.fake_client import FakeToolClient


ATV_PROFILE = {
    "issue_profile": "atv_reboot_instability",
    "grasshopper_profile": "atv_core",
    "upload_mode": "balanced",
    "max_files_per_type": 10,
    "max_total_files": 80,
    "core_log_types": ["procmgr", "android_main", "android_system", "launcher"],
    "supplemental_log_types": ["qt_gui", "reactuijava", "sddp_log", "invidiDebugLog", "updater"],
}


def make_runner(tmp_path: Path, client: FakeToolClient, *, commit: bool = True) -> tuple[PhaseRunner, RunState]:
    settings = replace(Settings(), output_dir=tmp_path, commit=commit, notify=False, max_upload_files=50)
    state = RunState(
        schema_version=1,
        run_id="test-grasshopper-call-path",
        mode="commit" if commit else "dry_run",
        role="operator",
        started_at="2026-08-06T00:00:00+00:00",
    )
    state.data["profile_catalog"] = {"atv_reboot_instability": dict(ATV_PROFILE)}
    state.data["source_inventory"] = {
        "source_availability": {"list_upload_trackers": True, "local_pending_ledger": True}
    }
    store = RunStore(tmp_path, state.run_id)
    executor = SerialExecutor(client, settings, state, store)
    return PhaseRunner(settings, state, store, executor), state


def calls_for(client: FakeToolClient, tool: str) -> list[dict]:
    return [row for row in client.calls if row["tool"] == tool]


@pytest.mark.asyncio
async def test_phase5_wires_catalog_metadata_to_plan_upload_and_tracker(tmp_path: Path):
    client = FakeToolClient()
    runner, _state = make_runner(tmp_path, client, commit=True)

    result = await runner._request_grasshopper_upload(
        1,
        "R1955706171",
        "atv_reboot_instability",
        "PMGR_UNEXPECTED_EXIT",
        "candidate-atv",
    )

    plan_args = calls_for(client, "grasshopper_plan_profile_upload")[0]["arguments"]
    upload_args = calls_for(client, "grasshopper_upload_profile_logs")[0]["arguments"]
    tracker_args = calls_for(client, "record_upload_tracker")[0]["arguments"]

    expected_types = ATV_PROFILE["core_log_types"] + ATV_PROFILE["supplemental_log_types"]
    assert plan_args == {
        "profile": "atv_core",
        "receiver_id": "R1955706171",
        "upload_mode": "balanced",
        "max_files_per_type": 10,
        "max_total_files": 50,
    }
    assert upload_args == {**plan_args, "dry_run": False, "allow_live_upload": True}
    assert tracker_args["issue_profile"] == "atv_reboot_instability"
    assert tracker_args["grasshopper_profile"] == "atv_core"
    assert tracker_args["requested_log_types"] == ",".join(expected_types)
    assert result["status"] == AcquisitionState.UPLOAD_SUBMITTED_TRACKED.value
    assert result["profile_metadata"]["grasshopper_profile"] == "atv_core"
    assert result["grasshopper_request_id"] == "gh-req-abc123"


@pytest.mark.asyncio
async def test_phase5_accepted_upload_with_tracker_write_failure_is_partial_success(tmp_path: Path):
    class TrackerFailureClient(FakeToolClient):
        def _response(self, tool, args):
            if tool == "record_upload_tracker":
                return {"ok": False, "error": "simulated tracker persistence failure"}
            return super()._response(tool, args)

    client = TrackerFailureClient()
    runner, _state = make_runner(tmp_path, client, commit=True)

    result = await runner._request_grasshopper_upload(
        2,
        "R1955706171",
        "atv_reboot_instability",
        "PMGR_UNEXPECTED_EXIT",
        "candidate-atv",
    )

    assert result["status"] == AcquisitionState.UPLOAD_ACCEPTED_TRACKER_WRITE_FAILED.value
    assert result["submitted"] is True
    assert result["tracker_write_succeeded"] is False
    assert result["grasshopper_request_id"] == "gh-req-abc123"
    assert result["tracker_id"] == ""
    assert result["pending"]["workflow_status"] == "TRACKER_WRITE_FAILED_RECONCILIATION_REQUIRED"


@pytest.mark.asyncio
async def test_phase6_uses_typed_nested_contract_and_catalog_profile_metadata(tmp_path: Path):
    client = FakeToolClient()
    runner, state = make_runner(tmp_path, client, commit=True)
    state.active_date_window = ["2026-08-06"]
    state.data["actionable_profiles"] = [
        {
            "profile_id": "atv_reboot_instability",
            "best_receiver": "R1955706171",
            "required_logs_present": False,
        }
    ]

    payload = await runner.phase_6_data_collection()
    row = payload["uploads"][0]

    plan_args = calls_for(client, "grasshopper_plan_profile_upload")[0]["arguments"]
    upload_args = calls_for(client, "grasshopper_upload_profile_logs")[0]["arguments"]
    assert plan_args["profile"] == "atv_core"
    assert plan_args["max_total_files"] == 50
    assert upload_args["profile"] == "atv_core"
    assert row["file_count"] == 10
    assert row["grasshopper_request_id"] == "gh-req-abc123"
    assert row["submission_attempted"] is True
    assert row["submission_accepted"] is True
    assert row["status"] == "UPLOADED"
    assert row["receipt_verified"] is True


@pytest.mark.asyncio
async def test_phase6_zero_files_is_skipped_not_mislabeled_as_rejected(tmp_path: Path):
    class ZeroPlanClient(FakeToolClient):
        def _response(self, tool, args):
            if tool == "grasshopper_plan_profile_upload":
                return {
                    "status": "success",
                    "profile": args["profile"],
                    "receiver_id": args["receiver_id"],
                    "plan": {
                        "selected_file_count": 0,
                        "selected_file_ids": [],
                        "expanded_log_types": [],
                        "missing_profile_log_types": [],
                        "deferred_profile_log_types_by_upload_mode": [],
                        "files_available_by_type": {},
                    },
                }
            return super()._response(tool, args)

    client = ZeroPlanClient()
    runner, state = make_runner(tmp_path, client, commit=True)
    state.active_date_window = ["2026-08-06"]
    state.data["actionable_profiles"] = [
        {
            "profile_id": "atv_reboot_instability",
            "best_receiver": "R1955706171",
            "required_logs_present": False,
        }
    ]

    payload = await runner.phase_6_data_collection()
    row = payload["uploads"][0]

    assert row["status"] == AcquisitionState.UPLOAD_SKIPPED_NO_FILES.value
    assert row["submission_attempted"] is False
    assert row["submission_accepted"] is False
    assert row["current_s3_coverage"] == "UNKNOWN"
    assert "selected zero" in row["reason"].lower()
    assert not calls_for(client, "grasshopper_upload_profile_logs")
    assert not calls_for(client, "verify_profile_upload_receipt")


@pytest.mark.asyncio
async def test_phase5_duplicate_inventory_unknown_fails_closed_before_plan(tmp_path: Path):
    client = FakeToolClient()
    runner, state = make_runner(tmp_path, client, commit=True)
    state.data["source_inventory"] = {"source_availability": {}}

    result = await runner._request_grasshopper_upload(
        9,
        "R1955706171",
        "atv_reboot_instability",
        "PMGR_UNEXPECTED_EXIT",
        "candidate-atv",
    )

    assert result["status"] == AcquisitionState.UPLOAD_BLOCKED_DUPLICATE_PREFLIGHT_UNAVAILABLE.value
    assert result["submitted"] is False
    assert result["submission_attempted"] is False
    assert result["duplicate_preflight"]["status"] == "INCOMPLETE"
    assert not calls_for(client, "grasshopper_plan_profile_upload")
    assert not calls_for(client, "grasshopper_upload_profile_logs")


@pytest.mark.asyncio
async def test_phase5_result_retains_non_secret_plan_identity(tmp_path: Path):
    client = FakeToolClient()
    runner, _state = make_runner(tmp_path, client, commit=True)

    result = await runner._request_grasshopper_upload(
        10,
        "R1955706171",
        "atv_reboot_instability",
        "PMGR_UNEXPECTED_EXIT",
        "candidate-atv",
    )

    assert result["selected_file_ids"] == list(range(1, 11))
    assert result["file_count"] == 10
    assert result["plan_fingerprint"].startswith("sha256:")
    assert result["grasshopper_profile"] == "atv_core"
    assert result["grasshopper_request_id"] == "gh-req-abc123"
    assert result["identifier_provenance"] == "proven_from_nested_response"


@pytest.mark.asyncio
async def test_phase5_local_pending_ledger_error_fails_closed_before_plan(tmp_path: Path):
    client = FakeToolClient()
    runner, state = make_runner(tmp_path, client, commit=True)
    state.data["source_inventory"]["source_availability"]["local_pending_ledger"] = False

    result = await runner._request_grasshopper_upload(
        11,
        "R1955706171",
        "atv_reboot_instability",
        "PMGR_UNEXPECTED_EXIT",
        "candidate-atv",
    )

    assert result["status"] == AcquisitionState.UPLOAD_BLOCKED_DUPLICATE_PREFLIGHT_UNAVAILABLE.value
    assert result["duplicate_preflight"]["reason"] == "TRACKER_INVENTORY_UNAVAILABLE"
    assert not calls_for(client, "grasshopper_plan_profile_upload")
    assert not calls_for(client, "grasshopper_upload_profile_logs")



@pytest.mark.asyncio
async def test_phase6_duplicate_inventory_unavailable_blocks_before_plan(tmp_path: Path):
    client = FakeToolClient()
    runner, state = make_runner(tmp_path, client, commit=True)
    state.active_date_window = ["2026-08-06"]
    state.data["source_inventory"]["source_availability"]["local_pending_ledger"] = False
    state.data["actionable_profiles"] = [
        {
            "profile_id": "atv_reboot_instability",
            "best_receiver": "R1955706171",
            "required_logs_present": False,
        }
    ]

    payload = await runner.phase_6_data_collection()
    row = payload["uploads"][0]

    assert row["status"] == AcquisitionState.UPLOAD_BLOCKED_DUPLICATE_PREFLIGHT_UNAVAILABLE.value
    assert row["submission_attempted"] is False
    assert not calls_for(client, "grasshopper_plan_profile_upload")
    assert not calls_for(client, "grasshopper_upload_profile_logs")


@pytest.mark.asyncio
async def test_phase1_corrupt_local_ledger_is_preserved_and_reported_unavailable(tmp_path: Path):
    client = FakeToolClient()
    runner, state = make_runner(tmp_path, client, commit=True)
    original = "{corrupt-ledger-do-not-overwrite"
    runner.store.pending_path.write_text(original, encoding="utf-8")

    payload = await runner.phase_1_source_inventory()

    assert payload["status"] in {"OK", "PARTIAL"}
    assert runner.store.pending_path.read_text(encoding="utf-8") == original
    availability = state.data["source_inventory"]["source_availability"]
    assert availability["list_upload_trackers"] is True
    assert availability["local_pending_ledger"] is False
    assert state.data["source_inventory"]["source_error_classes"]["local_pending_ledger"] == "JSONDecodeError"
