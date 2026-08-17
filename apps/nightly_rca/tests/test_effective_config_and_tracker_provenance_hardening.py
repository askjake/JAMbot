from __future__ import annotations

import json
from pathlib import Path

from apps.nightly_rca.config import Settings
from apps.nightly_rca.state import RunStore


def test_effective_mode_provenance_exposes_source_not_secret_values(tmp_path: Path):
    from apps.nightly_rca.config import build_effective_configuration_provenance

    settings = Settings(output_dir=tmp_path, commit=True, webhook_url="https://secret.invalid/hook")
    result = build_effective_configuration_provenance(
        settings,
        process_env={
            "NIGHTLY_RCA_COMMIT": "false",
            "NIGHTLY_RCA_WRITE_AUTHORIZED": "true",
            "NIGHTLY_RCA_WEBHOOK": "https://secret.invalid/hook",
        },
        cli_mode_source="CLI_COMMIT",
        env_file_path=tmp_path / "nightly.env",
        bootstrapped_keys={"NIGHTLY_RCA_COMMIT", "NIGHTLY_RCA_WEBHOOK"},
    )
    assert result["effective_mode"] == "commit"
    assert result["effective_mode_source"] == "CLI_COMMIT"
    assert result["write_authorization_present"] is True
    assert result["webhook_configured"] is True
    rendered = json.dumps(result, sort_keys=True)
    assert "secret.invalid" not in rendered
    assert "https://" not in rendered
    assert result["configuration_sources"]["commit"] == "CLI_COMMIT"


def test_cron_timezone_is_unknown_without_direct_cron_tz_or_tz(tmp_path: Path):
    from apps.nightly_rca.config import build_effective_configuration_provenance

    result = build_effective_configuration_provenance(
        Settings(output_dir=tmp_path),
        process_env={},
        env_file_path=tmp_path / "nightly.env",
    )
    assert result["cron_timezone"] == "UNKNOWN"
    assert result["cron_timezone_evidence"] == "NOT_DIRECTLY_VERIFIED"
    assert result["cron_timezone_confidence"] == "UNKNOWN"


def test_direct_cron_tz_is_reported_as_verified(tmp_path: Path):
    from apps.nightly_rca.config import build_effective_configuration_provenance

    result = build_effective_configuration_provenance(
        Settings(output_dir=tmp_path),
        process_env={"CRON_TZ": "America/Denver"},
    )
    assert result["cron_timezone"] == "America/Denver"
    assert result["cron_timezone_evidence"] == "CRON_TZ_ENV"
    assert result["cron_timezone_confidence"] == "DIRECT"


def test_pending_identifiers_are_typed_and_historical_state_is_explicit(tmp_path: Path):
    from apps.nightly_rca.state import normalize_pending_identifier_provenance

    current = normalize_pending_identifier_provenance(
        {
            "tracker_id": "trk-1",
            "request_id": "local-1",
            "grasshopper_request_id": "gh-1",
            "origin_run_id": "older-run",
        },
        current_run_id="current-run",
    )
    assert current["local_correlation_id"] == "local-1"
    assert current["grasshopper_request_id"] == "gh-1"
    assert current["identifier_provenance"] == "TYPED_TRACKER_LOCAL_AND_EXTERNAL"
    assert current["historical_carry_forward"] is True
    assert current["record_scope"] == "PHYSICAL_TRACKER_ROW"


def test_legacy_request_id_is_not_claimed_as_external_grasshopper_id(tmp_path: Path):
    from apps.nightly_rca.state import normalize_pending_identifier_provenance

    row = normalize_pending_identifier_provenance(
        {"request_id": "legacy-opaque", "receiver_id": "R1955706171"},
        current_run_id="run-now",
    )
    assert row["grasshopper_request_id"] == ""
    assert row["local_correlation_id"] == ""
    assert row["legacy_untyped_request_id"] == "legacy-opaque"
    assert row["identifier_provenance"] == "LEGACY_UNTYPED_REQUEST_ID"
    assert row["historical_carry_forward"] is True


def test_pending_store_deduplicates_by_typed_physical_identity(tmp_path: Path):
    store = RunStore(tmp_path, "run-now")
    store.save_pending_investigations(
        [
            {"tracker_id": "trk-1", "request_id": "local-a", "grasshopper_request_id": "gh-a"},
            {"tracker_id": "trk-1", "request_id": "local-b", "grasshopper_request_id": "gh-b"},
            {"request_id": "legacy-1", "receiver_id": "R1955706171"},
        ]
    )
    payload = json.loads(store.pending_path.read_text())
    assert payload["item_count"] == 2
    assert payload["count_scope"] == {
        "collection": "pending_investigations",
        "record_scope": "PHYSICAL_TRACKER_ROW",
        "deduplication_key": "typed_physical_identity",
    }
    rows = store.load_pending_investigations()
    tracker = next(row for row in rows if row.get("tracker_id") == "trk-1")
    assert tracker["local_correlation_id"] == "local-b"
    legacy = next(row for row in rows if row.get("legacy_untyped_request_id") == "legacy-1")
    assert legacy["identifier_provenance"] == "LEGACY_UNTYPED_REQUEST_ID"


def test_launcher_records_env_source_without_emitting_values():
    script = (Path(__file__).parents[1] / "run_nightly.sh").read_text()
    assert 'export NIGHTLY_RCA_LAUNCHER_ENV_SOURCE="$ENV_FILE"' in script
    assert "cat $ENV_FILE" not in script


def test_commit_enablement_does_not_claim_cron_is_utc():
    script = (Path(__file__).parents[1] / "enable_nightly_rca_commit.sh").read_text()
    assert "0 2 * * * UTC" not in script
    assert "cron daemon timezone" in script
    assert "CRON_TZ or TZ" in script


def test_report_surfaces_effective_config_and_typed_tracker_scope_without_secret(tmp_path: Path):
    from apps.nightly_rca.report import render_report
    from apps.nightly_rca.state import RunState

    state = RunState(
        schema_version=1, run_id="run-now", mode="commit", role="operator",
        started_at="2026-08-06T00:00:00Z", completed_at="2026-08-06T00:01:00Z",
        status="COMPLETE",
        data={
            "effective_configuration": {
                "effective_mode": "commit",
                "effective_mode_source": "CLI_COMMIT",
                "role": "operator",
                "write_authorization_present": True,
                "notify_enabled": True,
                "webhook_configured": True,
                "output_dir": str(tmp_path),
                "max_upload_files": 10,
                "pending_batch_size": 25,
                "configured_server_families": ["grasshopper_mcp"],
                "env_file_path": str(tmp_path / "nightly.env"),
                "env_file_exists": False,
                "cron_timezone": "UNKNOWN",
                "cron_timezone_evidence": "NOT_DIRECTLY_VERIFIED",
                "cron_timezone_confidence": "UNKNOWN",
                "configuration_sources": {"commit": "CLI_COMMIT"},
            },
            "pending_investigations": [{
                "currentness": "HISTORICAL_CARRY_FORWARD",
                "profile_id": "atv_reboot_instability",
                "receiver_id": "R1955706171",
                "tracker_id": "trk-1",
                "local_correlation_id": "local-1",
                "grasshopper_request_id": "gh-1",
                "identifier_provenance": "TYPED_TRACKER_LOCAL_AND_EXTERNAL",
                "workflow_status": "WAITING_FOR_LOGS",
            }],
            "final": {"executive_verdict": "PARTIAL", "final_numbers": {}, "remaining_blockers": []},
        },
    )
    text = render_report(state)
    assert "Effective configuration provenance" in text
    assert "HISTORICAL_CARRY_FORWARD" in text
    assert "TYPED_TRACKER_LOCAL_AND_EXTERNAL" in text
    assert "secret.invalid" not in text
    assert "https://" not in text


def test_notification_labels_pending_currentness_and_timezone_evidence(tmp_path: Path):
    from apps.nightly_rca.notify import summarize
    from apps.nightly_rca.state import RunState, RunStore

    state = RunState(
        schema_version=1, run_id="run-now", mode="commit", role="operator",
        started_at="2026-08-06T00:00:00Z", completed_at="2026-08-06T00:01:00Z",
        status="COMPLETE", completed_phases=[],
        data={
            "effective_configuration": {
                "effective_mode": "commit",
                "effective_mode_source": "CLI_COMMIT",
                "cron_timezone": "UNKNOWN",
                "cron_timezone_evidence": "NOT_DIRECTLY_VERIFIED",
            },
            "pending_investigations": [
                {"historical_carry_forward": True, "currentness": "HISTORICAL_CARRY_FORWARD"},
                {"historical_carry_forward": False, "currentness": "CURRENT_RUN"},
            ],
            "final": {
                "executive_verdict": "DEFERRED", "final_numbers": {},
                "remaining_blockers": [], "learning_recommendations": [],
            },
        },
    )
    text = summarize(state, RunStore(tmp_path, state.run_id))
    assert "cron TZ=UNKNOWN [NOT_DIRECTLY_VERIFIED]" in text
    assert "historical_carry_forward=1" in text
    assert "current=1" in text
