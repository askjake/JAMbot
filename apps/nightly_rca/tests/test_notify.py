from __future__ import annotations

from pathlib import Path

from nightly_rca.notify import summarize
from nightly_rca.state import RunState, RunStore


def test_notification_reports_real_phase_count_receiver_and_pending_work(tmp_path: Path):
    state = RunState(
        schema_version=1,
        run_id="20260723T080002Z",
        mode="commit",
        role="operator",
        started_at="2026-07-23T08:00:02+00:00",
        completed_at="2026-07-23T08:25:21+00:00",
        completed_phases=["0", "1", "D1", "2", "2b", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"],
        active_date_window=["2026-06-01", "2026-05-29", "2026-05-28"],
    )
    state.data.update({
        "candidate_clusters": [{
            "candidate_id": "cand_001",
            "suspected_profile": "plc_error_drm_server_not_reachable",
            "alert_name": "PLC_ERROR_DRM_SERVER_NOT_REACHABLE",
            "alert_count": 2545,
            "receivers_seen": ["R1887512426"],
            "receiver_samples": [{"t_receiver_id": "R1887512426", "n_alert_count": 81}],
        }],
        "profile_validations": [{
            "profile_id": "auto_plc_error_drm_server_not_reachable",
            "best_receiver": "R1887512426",
            "required_logs_present": False,
            "classification": "AWAITING_LOGS",
        }],
        "pending_investigations": [{
            "tracker_id": "nightly-rca-phase5-cand_001-R1887512426-20260723T080002Z",
            "receiver_id": "R1887512426",
            "workflow_status": "WAITING_FOR_LOGS",
        }],
        "cases": [{"status": "CASE_BLOCKED_DATA_COLLECTION", "real_case": False}],
        "canaries": {"canaries": [
            {"canary": "A_heavy_auth", "pass": True, "skipped": False},
            {"canary": "E_expected_no_logs", "pass": True, "skipped": False},
        ]},
        "final": {
            "executive_verdict": "DEFERRED — pending investigations were persisted and will be reconciled automatically on the next cron run.",
            "final_numbers": {},
            "remaining_blockers": ["auto_plc_error_drm_server_not_reachable: AWAITING_LOGS"],
            "learning_recommendations": [],
        },
    })
    store = RunStore(tmp_path, state.run_id)
    store.save_phase("2", {
        "anomalies": {
            "summary": [{
                "alert_name": "PLC_ERROR_DRM_SERVER_NOT_REACHABLE",
                "max_actual": 2545,
                "max_typical": 108,
            }],
            "records": [{
                "alert_name": "PLC_ERROR_DRM_SERVER_NOT_REACHABLE",
                "actual": 2545,
                "percentage_change": 1942,
            }],
        },
        "candidate_clusters": state.data["candidate_clusters"],
    })

    state.metrics["expected_negative_canary_pass"] = 1
    message = summarize(state, store)

    assert "Phases:* 18/18" in message
    assert "`R1887512426`" in message
    assert "`cand_001` —" not in message
    assert "2,545 actual" in message
    assert "Automatic receipt reconciliation" in message
    assert "No confirmed problems" in message
    assert "🔵 *Verdict:* DEFERRED" in message
    assert "expected_negative_canary_pass=1" in message
    assert "E_expected_no_logs" in message
