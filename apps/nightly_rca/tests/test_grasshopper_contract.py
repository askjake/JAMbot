"""
Regression tests for grasshopper_contract typed adapter.

These tests verify:
- Nested plan structure parsing (selected_file_count, file_ids)
- Inner vs outer status distinction
- Profile metadata propagation
- Identifier provenance separation
- Nightly cap enforcement
- Accurate state machine transitions

Phase 2: These tests MUST fail before the source fix is applied.
"""
from __future__ import annotations

import pytest

from nightly_rca.grasshopper_contract import (
    AcquisitionState,
    ACQUISITION_STATE_META,
    GrasshopperPlanResult,
    GrasshopperUploadResult,
    IdentifierSet,
    NIGHTLY_MAX_UPLOAD_FILES,
    ProfileMetadata,
    UNKNOWN_PROVENANCE,
    build_identifier_set,
    build_profile_metadata,
    determine_acquisition_state,
    parse_plan_response,
    parse_upload_response,
)


# ── Fixtures: Real nested Grasshopper contract ────────────────────────────────

PLAN_RESPONSE_ZERO_FILES = {
    "status": "OK",
    "response": {
        "status": "success",
        "profile": "atv_reboot_instability",
        "receiver_id": "R-SANITIZED",
        "plan": {
            "selected_file_count": 0,
            "selected_file_ids": [],
            "expanded_log_types": ["atv_reboot_instability"],
            "missing_profile_log_types": ["atv_reboot_instability"],
            "files_available_by_type": {
                "nal": 242,
                "stbCtrl": 169,
                "procmgr": 45,
                "sg_server": 51,
                "qt_gui": 22,
            },
        },
    },
}

PLAN_RESPONSE_POSITIVE_FILES = {
    "status": "OK",
    "response": {
        "status": "success",
        "profile": "atv_core",
        "receiver_id": "R-SANITIZED",
        "plan": {
            "selected_file_count": 10,
            "selected_file_ids": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "expanded_log_types": ["procmgr", "android_main", "qt_gui"],
            "missing_profile_log_types": [],
            "files_available_by_type": {
                "procmgr": 45,
                "android_main": 30,
                "qt_gui": 22,
            },
        },
    },
}

UPLOAD_RESPONSE_SUCCESS = {
    "status": "OK",
    "response": {
        "status": "success",
        "profile": "atv_core",
        "receiver_id": "R-SANITIZED",
        "uploaded_file_ids": [1, 2, 3],
        "response": {
            "status": "success",
            "request_id": "SANITIZED-EXTERNAL-ID",
        },
        "destination": "s3",
    },
}

UPLOAD_RESPONSE_INNER_ERROR = {
    "status": "OK",
    "response": {
        "status": "success",
        "profile": "atv_core",
        "receiver_id": "R-SANITIZED",
        "uploaded_file_ids": [],
        "response": {
            "status": "error",
            "request_id": "",
            "error": "INVALID_PROFILE",
        },
        "destination": "s3",
    },
}

UPLOAD_RESPONSE_NO_REQUEST_ID = {
    "status": "OK",
    "response": {
        "status": "success",
        "profile": "atv_core",
        "receiver_id": "R-SANITIZED",
        "uploaded_file_ids": [1, 2, 3],
        "response": {
            "status": "success",
            "request_id": "",
        },
        "destination": "s3",
    },
}

PLAN_RESPONSE_EXECUTOR_ERROR = {
    "status": "ERROR",
    "error": "timeout connecting to grasshopper",
    "response": None,
}


# ── Test: Nested Plan Parsing ─────────────────────────────────────────────────

class TestNestedPlanParsing:

    def test_nested_plan_selected_file_count_zero(self):
        result = parse_plan_response(PLAN_RESPONSE_ZERO_FILES)
        assert result.is_executor_success is True
        assert result.is_inner_success is True
        assert result.selected_file_count == 0
        assert result.file_count_is_zero is True
        assert result.file_count_is_positive is False

    def test_nested_plan_selected_file_count_positive(self):
        result = parse_plan_response(PLAN_RESPONSE_POSITIVE_FILES)
        assert result.is_executor_success is True
        assert result.is_inner_success is True
        assert result.selected_file_count == 10
        assert result.file_count_is_positive is True
        assert result.file_count_is_zero is False

    def test_nested_plan_selected_file_ids_preserved(self):
        result = parse_plan_response(PLAN_RESPONSE_POSITIVE_FILES)
        assert result.selected_file_ids == (101, 102, 103, 104, 105, 106, 107, 108, 109, 110)
        assert len(result.selected_file_ids) == 10

    def test_unknown_plan_count_is_not_zero_files(self):
        """A plan without a readable count is UNKNOWN, not zero."""
        bad_plan = {
            "status": "OK",
            "response": {
                "status": "success",
                "profile": "atv_core",
                "plan": {},  # no selected_file_count
            },
        }
        result = parse_plan_response(bad_plan)
        assert result.selected_file_count is None
        assert result.has_valid_file_count is False
        # Must not be treated as zero files
        assert result.file_count_is_zero is False

    def test_zero_files_is_not_unknown_plan_count(self):
        """Zero files is a known value, not unknown."""
        result = parse_plan_response(PLAN_RESPONSE_ZERO_FILES)
        assert result.selected_file_count == 0
        assert result.has_valid_file_count is True
        assert result.file_count_is_zero is True


# ── Test: Inner vs Outer Status ───────────────────────────────────────────────

class TestStatusLayers:

    def test_inner_grasshopper_error_status_is_rejected(self):
        result = parse_upload_response(UPLOAD_RESPONSE_INNER_ERROR)
        assert result.is_executor_success is True
        assert result.is_inner_success is False
        assert result.write_state == "inner_rejected"

    def test_outer_executor_ok_and_inner_success_are_distinct_layers(self):
        result = parse_upload_response(UPLOAD_RESPONSE_SUCCESS)
        # Executor is OK
        assert result.is_executor_success is True
        # Inner response is success
        assert result.is_inner_success is True
        # These are different layers
        assert result.executor_ok is True
        assert result.inner_status == "success"
        assert result.outer_status == "success"


# ── Test: Acquisition State Machine ──────────────────────────────────────────

class TestAcquisitionStateMachine:

    def test_zero_files_never_says_upload_not_accepted(self):
        plan = parse_plan_response(PLAN_RESPONSE_ZERO_FILES)
        state = determine_acquisition_state(plan, None, dry_run=False)
        assert state == AcquisitionState.UPLOAD_SKIPPED_NO_FILES
        meta = ACQUISITION_STATE_META[state]
        assert "not accepted" not in meta["operator_message"].lower()
        assert "rejected" not in meta["operator_message"].lower()

    def test_unknown_size_never_says_upload_not_accepted(self):
        bad_plan = {
            "status": "OK",
            "response": {"status": "success", "plan": {}},
        }
        plan = parse_plan_response(bad_plan)
        state = determine_acquisition_state(plan, None, dry_run=False)
        assert state == AcquisitionState.UPLOAD_BLOCKED_UNKNOWN_BATCH_SIZE
        meta = ACQUISITION_STATE_META[state]
        assert "not accepted" not in meta["operator_message"].lower()
        assert "rejected" not in meta["operator_message"].lower()

    def test_submit_not_called_never_reports_rejection(self):
        """If no upload was submitted, the state must not be REJECTED."""
        plan = parse_plan_response(PLAN_RESPONSE_ZERO_FILES)
        # upload=None means no submission attempt
        state = determine_acquisition_state(plan, None, dry_run=False)
        assert state != AcquisitionState.UPLOAD_REJECTED
        assert state != AcquisitionState.UPLOAD_ACCEPTANCE_UNKNOWN


# ── Test: Request ID Extraction ───────────────────────────────────────────────

class TestRequestIdExtraction:

    def test_nested_upload_request_id_is_extracted(self):
        result = parse_upload_response(UPLOAD_RESPONSE_SUCCESS)
        assert result.inner_request_id == "SANITIZED-EXTERNAL-ID"
        assert result.has_proven_request_id is True

    def test_success_without_external_request_id_is_acceptance_unknown(self):
        result = parse_upload_response(UPLOAD_RESPONSE_NO_REQUEST_ID)
        assert result.is_executor_success is True
        assert result.is_inner_success is True
        assert result.inner_request_id == ""
        assert result.has_proven_request_id is False
        assert result.write_state == "acceptance_unknown"


# ── Test: Identifier Provenance ───────────────────────────────────────────────

class TestIdentifierProvenance:

    def test_local_correlation_and_external_request_ids_are_separate(self):
        upload = parse_upload_response(UPLOAD_RESPONSE_SUCCESS)
        ids = build_identifier_set(
            local_correlation_id="nightly-rca-phase5-cand-R100-run123",
            upload_result=upload,
            tracker_id="tracker-abc",
            origin_run_id="run123",
        )
        assert ids.local_correlation_id == "nightly-rca-phase5-cand-R100-run123"
        assert ids.grasshopper_request_id == "SANITIZED-EXTERNAL-ID"
        assert ids.tracker_id == "tracker-abc"
        assert ids.origin_run_id == "run123"
        assert ids.identifier_provenance == "proven_from_nested_response"
        # They must be different values
        assert ids.local_correlation_id != ids.grasshopper_request_id

    def test_missing_inner_id_gives_unknown_provenance(self):
        upload = parse_upload_response(UPLOAD_RESPONSE_NO_REQUEST_ID)
        ids = build_identifier_set(
            local_correlation_id="local-123",
            upload_result=upload,
        )
        assert ids.grasshopper_request_id == ""
        assert ids.identifier_provenance == "executor_success_no_inner_id"


# ── Test: Profile Metadata ────────────────────────────────────────────────────

class TestProfileMetadata:

    def test_profile_metadata_is_propagated_to_plan(self):
        meta = ProfileMetadata(
            issue_profile="atv_reboot_instability",
            grasshopper_profile="atv_core",
            upload_mode="balanced",
            max_files_per_type=15,
            max_total_files=120,
            core_log_types=("procmgr", "android_main", "android_system", "launcher"),
            supplemental_log_types=("qt_gui", "reactuijava", "sddp_log", "invidiDebugLog", "updater"),
        )
        result = parse_plan_response(PLAN_RESPONSE_POSITIVE_FILES, profile_metadata=meta)
        assert result.profile_metadata is not None
        assert result.profile_metadata.issue_profile == "atv_reboot_instability"
        assert result.profile_metadata.grasshopper_profile == "atv_core"

    def test_nightly_cap_clamps_profile_max_total_to_50(self):
        meta = ProfileMetadata(
            issue_profile="atv_reboot_instability",
            grasshopper_profile="atv_core",
            max_total_files=120,
        )
        assert meta.max_total_files == 120
        assert meta.effective_max_total_files == 50
        assert NIGHTLY_MAX_UPLOAD_FILES == 50


# ── Test: Fake Client Contract ────────────────────────────────────────────────

class TestFakeClientContract:

    def test_fake_client_uses_real_nested_grasshopper_contract(self):
        """Verify that the fake client plan response can be parsed
        by the typed adapter, proving contract alignment."""
        # This is the shape the existing fake returns:
        old_fake_response = {"file_count": 10, "date": "2026-07-21", "files": ["f0", "f1"]}
        # When wrapped by executor:
        executor_result = {"status": "OK", "response": old_fake_response}
        result = parse_plan_response(executor_result)
        # The OLD fake uses top-level file_count, which the adapter supports as fallback
        assert result.selected_file_count == 10
        assert result.is_executor_success is True
        # But the REAL contract uses nested plan.selected_file_count
        real_result = parse_plan_response(PLAN_RESPONSE_POSITIVE_FILES)
        assert real_result.selected_file_count == 10
        assert real_result.is_inner_success is True
        assert real_result.expanded_log_types == ("procmgr", "android_main", "qt_gui")


class TestEndToEndPhase5Integration:
    """End-to-end regression: real nested plan fixture → Phase 5 upload path → correct state."""

    def test_nested_plan_fixture_through_parse_plan_response(self):
        """Simulates what Phase 5 sees: executor wraps the real Grasshopper contract."""
        # This is what SerialExecutor.call returns for grasshopper_plan_profile_upload
        executor_result = {
            "status": "OK",
            "response": {
                "status": "success",
                "profile": "atv_core",
                "receiver_id": "R123456",
                "plan": {
                    "selected_file_count": 10,
                    "selected_file_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "expanded_log_types": ["procmgr", "sg_server", "qt_gui", "reactuijava"],
                    "missing_profile_log_types": [],
                    "deferred_profile_log_types_by_upload_mode": [],
                    "files_available_by_type": {"procmgr": 3, "sg_server": 3, "qt_gui": 2, "reactuijava": 2},
                },
            },
            "error": None,
        }
        plan = parse_plan_response(executor_result)
        assert plan.is_executor_success
        assert plan.is_inner_success
        assert plan.selected_file_count == 10
        assert plan.has_valid_file_count
        assert not plan.file_count_is_zero
        assert plan.file_count_is_positive
        assert plan.profile == "atv_core"
        assert plan.receiver_id == "R123456"
        assert "procmgr" in plan.expanded_log_types
        assert plan.files_available_by_type.get("procmgr") == 3

    def test_nested_upload_fixture_through_parse_upload_response(self):
        """Simulates what Phase 5 sees for grasshopper_upload_profile_logs."""
        executor_result = {
            "status": "OK",
            "response": {
                "status": "success",
                "uploaded_file_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "destination": "s3://grasshopper-uploads/atv_core/R123456/",
                "response": {
                    "status": "success",
                    "request_id": "gh-req-abc123",
                },
            },
            "error": None,
        }
        upload = parse_upload_response(executor_result)
        assert upload.is_executor_success
        assert upload.is_inner_success
        assert upload.inner_request_id == "gh-req-abc123"
        assert upload.has_proven_request_id

    def test_full_plan_upload_state_machine_accepted(self):
        """Full path: plan(OK, 10 files) + upload(accepted) → UPLOAD_ACCEPTED."""
        plan_result = {
            "status": "OK",
            "response": {
                "status": "success",
                "profile": "atv_core",
                "receiver_id": "R123456",
                "plan": {"selected_file_count": 10, "selected_file_ids": list(range(1, 11))},
            },
            "error": None,
        }
        upload_result = {
            "status": "OK",
            "response": {
                "status": "success",
                "uploaded_file_ids": list(range(1, 11)),
                "destination": "s3://test/",
                "response": {"status": "success", "request_id": "gh-req-xyz"},
            },
            "error": None,
        }
        plan = parse_plan_response(plan_result)
        upload = parse_upload_response(upload_result)
        state = determine_acquisition_state(plan, upload, dry_run=False)
        assert state == AcquisitionState.UPLOAD_ACCEPTED

    def test_full_plan_upload_state_machine_blocked_batch_limit(self):
        """Full path: plan(OK, 51 files) → UPLOAD_BLOCKED_BATCH_LIMIT (50 cap)."""
        plan_result = {
            "status": "OK",
            "response": {
                "status": "success",
                "profile": "atv_core",
                "receiver_id": "R999999",
                "plan": {"selected_file_count": 51, "selected_file_ids": list(range(1, 52))},
            },
            "error": None,
        }
        plan = parse_plan_response(plan_result)
        state = determine_acquisition_state(plan, upload=None, dry_run=True, max_upload_files=50)
        assert state == AcquisitionState.UPLOAD_BLOCKED_BATCH_LIMIT

    def test_identifier_provenance_separation(self):
        """Proven Grasshopper request ID is separate from local correlation ID."""
        upload_result = {
            "status": "OK",
            "response": {
                "status": "success",
                "uploaded_file_ids": [1],
                "destination": "s3://test/",
                "response": {"status": "success", "request_id": "gh-proven-id"},
            },
            "error": None,
        }
        upload = parse_upload_response(upload_result)
        ids = build_identifier_set(
            local_correlation_id="nightly-rca-phase5-cluster1-R123-run1",
            upload_result=upload,
            tracker_id="tracker-001",
            origin_run_id="run-20260806",
        )
        assert ids.local_correlation_id == "nightly-rca-phase5-cluster1-R123-run1"
        assert ids.grasshopper_request_id == "gh-proven-id"
        assert ids.tracker_id == "tracker-001"
        assert ids.identifier_provenance == "proven_from_nested_response"
        # They are distinct — never conflated
        assert ids.local_correlation_id != ids.grasshopper_request_id

    def test_zero_files_distinct_from_unknown_count(self):
        """Zero files and unknown count produce different acquisition states."""
        # Zero files
        zero_plan = parse_plan_response({
            "status": "OK",
            "response": {"status": "success", "profile": "atv_core", "plan": {"selected_file_count": 0}},
            "error": None,
        })
        assert determine_acquisition_state(zero_plan, None, dry_run=True) == AcquisitionState.UPLOAD_SKIPPED_NO_FILES

        # Unknown count (missing selected_file_count)
        unknown_plan = parse_plan_response({
            "status": "OK",
            "response": {"status": "success", "profile": "atv_core", "plan": {}},
            "error": None,
        })
        assert determine_acquisition_state(unknown_plan, None, dry_run=True) == AcquisitionState.UPLOAD_BLOCKED_UNKNOWN_BATCH_SIZE

        # They are DIFFERENT states
        assert AcquisitionState.UPLOAD_SKIPPED_NO_FILES != AcquisitionState.UPLOAD_BLOCKED_UNKNOWN_BATCH_SIZE
