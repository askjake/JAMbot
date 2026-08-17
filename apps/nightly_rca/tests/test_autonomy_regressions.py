from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from nightly_rca.config import Settings
from nightly_rca.logic import discover_clusters
from nightly_rca.pipeline import NightlyPipeline
from nightly_rca.state import RunState, RunStore
from nightly_rca.tests.fake_client import FakeToolClient


def _state(name: str, *, mode: str = "commit") -> RunState:
    return RunState(
        schema_version=1,
        run_id=f"regression-{name}",
        mode=mode,
        role="operator",
        started_at="2026-07-23T08:00:02+00:00",
    )


class StrictWorkspaceClient(FakeToolClient):
    """Emulates the actual preview/commit and immutable queue contracts."""

    def __init__(self) -> None:
        super().__init__()
        self.queues: dict[str, dict[str, object]] = {}

    def _response(self, tool, args):
        if tool == "create_human_review_queue":
            queue_id = args["queue_id"]
            queue_hash = f"qh-{queue_id}"
            if args.get("dry_run"):
                assert args.get("case_ids"), "the nightly queue must be scoped to this run's eligible cases"
                items = [{"case_id": x} for x in args["case_ids"].split(",")]
                return {"ok": True, "result": "HUMAN_REVIEW_QUEUE_PREVIEW", "queue_hash": queue_hash, "item_count": len(items), "items": items, "write_performed": False}
            assert args.get("confirm_create") == "CREATE_HUMAN_REVIEW_QUEUE"
            assert args.get("expected_queue_hash") == queue_hash
            prior = self.queues.get(queue_id)
            if prior and prior.get("queue_hash") == queue_hash:
                return {"ok": True, "result": "HUMAN_REVIEW_QUEUE_ALREADY_EXISTS", "queue_id": queue_id, "queue_hash": queue_hash, "item_count": len(prior.get("items", [])), "write_performed": False}
            assert prior is None, "immutable queue id was reused for changed content"
            items = [{"case_id": x} for x in args["case_ids"].split(",")]
            self.queues[queue_id] = {"queue_hash": queue_hash, "items": items}
            return {"ok": True, "result": "HUMAN_REVIEW_QUEUE_CREATED", "queue_id": queue_id, "queue_hash": queue_hash, "item_count": len(items), "write_performed": True}
        if tool == "build_human_evidence_bundle":
            queue_id = args["queue_id"]
            assert queue_id in self.queues, "bundle attempted before its authoritative queue existed"
            bundle_hash = f"bh-{args['case_id']}"
            if args.get("dry_run"):
                return {"ok": True, "result": "HUMAN_EVIDENCE_BUNDLE_PREVIEW", "bundle_hash": bundle_hash, "blocking_codes": [], "write_performed": False}
            assert args.get("confirm_build") == "BUILD_HUMAN_EVIDENCE_BUNDLE"
            assert args.get("expected_bundle_hash") == bundle_hash
            return {"ok": True, "result": "HUMAN_EVIDENCE_BUNDLE_BUILT", "bundle_id": f"bundle-{args['case_id']}", "bundle_hash": bundle_hash, "write_performed": True}
        if tool == "export_human_adjudication_packet":
            packet_hash = f"ph-{args['case_id']}"
            if args.get("dry_run"):
                return {"ok": True, "result": "HUMAN_ADJUDICATION_PACKET_PREVIEW", "packet_hash": packet_hash, "blocking_codes": [], "write_performed": False}
            assert args.get("confirm_export") == "EXPORT_HUMAN_ADJUDICATION_PACKET"
            assert args.get("expected_packet_hash") == packet_hash
            return {"ok": True, "result": "HUMAN_ADJUDICATION_PACKET_EXPORTED", "packet_id": f"packet-{args['case_id']}", "packet_hash": packet_hash, "write_performed": True}
        return super()._response(tool, args)


@pytest.mark.asyncio
async def test_workspace_writes_follow_real_hash_bound_contract_and_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NIGHTLY_RCA_WRITE_AUTHORIZED", "true")
    settings = replace(Settings(), output_dir=tmp_path, commit=True, notify=False)
    client = StrictWorkspaceClient()
    state = _state("workspace")

    rc = await NightlyPipeline(settings=settings, client=client, state=state).run()

    assert rc == 0
    assert state.status == "COMPLETE"
    queue = state.data["queue"]
    assert queue["persisted"] is True
    assert queue["queue_id"] != settings.queue_id
    assert queue["queue_id"].endswith(state.run_id)
    assert all(row["persisted"] for row in state.data["bundles"])
    assert all(row["packet_persisted"] for row in state.data["packets"])
    assert state.metrics["persist_failures"] == 0


def test_anomaly_aliases_and_receiver_samples_create_actionable_cluster():
    clusters = discover_clusters(
        {},
        {
            "summary": [{"alert_name": "PLC_ERROR_DRM_SERVER_NOT_REACHABLE", "max_actual": 2545}],
            "records": [{
                "alert_name": "PLC_ERROR_DRM_SERVER_NOT_REACHABLE",
                "actual": 2545,
                "record_score": 99,
                "receiver_samples": [{"t_receiver_id": "R1887512426", "n_alert_count": 81}],
            }],
        },
        ["2026-06-01"],
    )
    assert len(clusters) == 1
    assert clusters[0]["alert_count"] == 2545
    assert clusters[0]["alert_name"] == "PLC_ERROR_DRM_SERVER_NOT_REACHABLE"
    assert clusters[0]["receivers_seen"] == ["R1887512426"]
    assert clusters[0]["receiver_samples"][0]["n_alert_count"] == 81


def test_pending_investigations_are_durable_across_run_directories(tmp_path: Path):
    first = RunStore(tmp_path, "run-one")
    pending = [{
        "tracker_id": "upload_auto_2004_R1891933896_req",
        "receiver_id": "R1891933896",
        "profile_id": "auto_2004",
        "candidate_id": "cand_002",
        "workflow_status": "WAITING_FOR_LOGS",
    }]
    first.save_pending_investigations(pending)

    second = RunStore(tmp_path, "run-two")
    loaded = second.load_pending_investigations()
    assert len(loaded) == 1
    # Core fields survive the round-trip unchanged
    for key in ("tracker_id", "receiver_id", "profile_id", "candidate_id", "workflow_status"):
        assert loaded[0][key] == pending[0][key]
    # save_pending_investigations adds created_at for priority ordering
    assert "created_at" in loaded[0]


@pytest.mark.asyncio
async def test_missing_logs_are_deferred_without_serial_sleep_or_false_upload_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class UploadFailureClient(FakeToolClient):
        def _response(self, tool, args):
            if tool == "verify_receiver_log_coverage":
                receivers = [x for x in str(args.get("receiver_ids", "")).split(",") if x]
                return {"coverage_matrix": [{"receiver_id": rx, "coverage_status": "no_s3_data", "required_logs_present": False} for rx in receivers]}
            if tool == "grasshopper_upload_profile_logs":
                return {"ok": False, "result": "UPLOAD_REJECTED", "write_performed": False}
            if tool == "count_alerts" and args.get("group_by") == "receiver":
                return {"buckets": []}
            return super()._response(tool, args)

    async def forbidden_sleep(seconds: float):
        if seconds:
            raise AssertionError("cron must not block waiting for an external log upload")
        return None

    monkeypatch.setenv("NIGHTLY_RCA_WRITE_AUTHORIZED", "true")
    monkeypatch.setattr("asyncio.sleep", forbidden_sleep)
    settings = replace(
        Settings(), output_dir=tmp_path, commit=True, notify=False,
        log_poll_max_attempts=1, log_poll_interval_seconds=300,
    )
    client = UploadFailureClient()
    state = _state("upload-failure")

    rc = await NightlyPipeline(settings=settings, client=client, state=state).run(stop_after="5")

    assert rc == 0
    affected = [v for v in state.data["profile_validations"] if v.get("required_logs_present") is False]
    assert affected
    assert all(v["classification"] != "AWAITING_LOGS" for v in affected)
    assert RunStore(tmp_path, state.run_id).load_pending_investigations() == []


@pytest.mark.asyncio
async def test_pending_upload_resumes_next_run_without_anomaly_or_duplicate_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A durable tracker advances on a later cron even when discovery is quiet."""

    class CrossRunClient(FakeToolClient):
        def __init__(self) -> None:
            super().__init__()
            self.second_run = False
            self.logs_ready = False

        def _response(self, tool, args):
            if tool == "list_anomalies" and self.second_run:
                return {"anomalies": []}
            if tool == "count_alerts":
                if args.get("group_by") == "receiver":
                    return {"buckets": []}
                if self.second_run:
                    return {"buckets": []}
            if tool == "verify_receiver_log_coverage":
                receivers = [x for x in str(args.get("receiver_ids", "")).split(",") if x]
                if self.logs_ready:
                    return {"coverage_matrix": [{
                        "receiver_id": rx,
                        "coverage_status": "ready_for_analysis",
                        "required_logs_present": True,
                        "available_log_types": ["procmgr", "android_main"],
                    } for rx in receivers]}
                return {"coverage_matrix": [{
                    "receiver_id": rx,
                    "coverage_status": "no_s3_data",
                    "required_logs_present": False,
                } for rx in receivers]}
            if tool == "update_upload_tracker_from_s3":
                response = super()._response(tool, args)
                if str(response.get("workflow_status", "")).upper() == "READY_FOR_ANALYSIS":
                    self.logs_ready = True
                return response
            return super()._response(tool, args)

    monkeypatch.setenv("NIGHTLY_RCA_WRITE_AUTHORIZED", "true")
    settings = replace(
        Settings(), output_dir=tmp_path, commit=True, notify=False,
        log_poll_max_attempts=1, log_poll_interval_seconds=0,
    )
    client = CrossRunClient()

    first = _state("cross-run-one")
    assert await NightlyPipeline(settings=settings, client=client, state=first).run(stop_after="5") == 0
    pending_after_first = RunStore(tmp_path, first.run_id).load_pending_investigations()
    assert pending_after_first
    tracker_id = pending_after_first[0]["tracker_id"]
    upload_count_after_first = client.counts["grasshopper_upload_profile_logs"]
    assert upload_count_after_first >= 1

    client.second_run = True
    second = _state("cross-run-two")
    assert await NightlyPipeline(settings=settings, client=client, state=second).run(stop_after="5") == 0

    assert second.data["candidate_clusters"] == []
    resumed = [row for row in second.data["profile_validations"] if row.get("tracker_id") == tracker_id]
    assert resumed and resumed[0]["resumed"] is True
    assert resumed[0]["classification"] == "PROFILE_VALIDATION_PASS"
    assert client.counts["grasshopper_upload_profile_logs"] == upload_count_after_first, "resumption submitted a duplicate upload"
    assert RunStore(tmp_path, second.run_id).load_pending_investigations() == []

def test_pending_created_at_is_preserved_across_repeated_saves(tmp_path: Path):
    """Repeated ledger saves must not replace an existing creation timestamp."""
    original_created_at = "2026-07-01T12:00:00+00:00"

    pending = [{
        "tracker_id": "upload_test_R123_request",
        "receiver_id": "R123",
        "profile_id": "test_profile",
        "candidate_id": "cand_test",
        "workflow_status": "WAITING_FOR_LOGS",
        "created_at": original_created_at,
    }]

    first = RunStore(tmp_path, "run-one")
    first.save_pending_investigations(pending)

    loaded = first.load_pending_investigations()
    assert len(loaded) == 1
    assert loaded[0]["created_at"] == original_created_at

    second = RunStore(tmp_path, "run-two")
    second.save_pending_investigations(loaded)

    reloaded = second.load_pending_investigations()
    assert len(reloaded) == 1
    assert reloaded[0]["created_at"] == original_created_at

