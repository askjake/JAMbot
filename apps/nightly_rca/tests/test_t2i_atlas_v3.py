from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from nightly_rca.config import Settings, build_effective_configuration_provenance
from nightly_rca.pipeline import NightlyPipeline
from nightly_rca.report import render_report
from nightly_rca.state import RunState
from nightly_rca.t2i_atlas import FONT_BOLD, FONT_REG, T2IAtlasError, generate_t2i_v3_atlas
from nightly_rca.tests.fake_client import FakeToolClient


LOCAL_FONT_SHA256 = hashlib.sha256(FONT_REG.read_bytes()).hexdigest()
LOCAL_FONT_BOLD_SHA256 = hashlib.sha256(FONT_BOLD.read_bytes()).hexdigest()


def _write_events(path: Path) -> bytes:
    rows = [
        {
            "phase": "1",
            "step": "1.1",
            "server": "s3_stb_logs",
            "tool": "get_tool_info",
            "arguments": {},
            "status": "OK",
            "elapsed_ms": 11,
            "response": {"ok": True, "schema_version": "1"},
            "error": None,
        },
        {
            "phase": "1",
            "step": "1.2",
            "server": "s3_stb_logs",
            "tool": "list_dates",
            "arguments": {"receiver_id": "R1234567890", "limit": 5},
            "status": "OK",
            "elapsed_ms": 22,
            "response": {"receiver_id": "R1234567890", "result": "DATES_READY"},
            "error": None,
        },
        {
            "phase": "6",
            "step": "6.1",
            "server": "grasshopper_mcp",
            "tool": "request_receiver_logs",
            "arguments": {"receiver_id": "R1234567890", "profile": "atv_core"},
            "status": "OK",
            "elapsed_ms": 33,
            "response": {
                "ok": False,
                "receiver_id": "R1234567890",
                "result": "UPLOAD_TRACKER_WAITING_PREVIEW",
                "receipt_status": "pending",
                "workflow_status": "WAITING_FOR_LOGS",
            },
            "error": None,
        },
        {
            "phase": "14",
            "step": "14.1",
            "server": "s3_stb_logs",
            "tool": "list_dates",
            "arguments": {"receiver_id": "R0000000001", "limit": 5},
            "status": "STEP_FAILED",
            "elapsed_ms": 44,
            "response": None,
            "error": "ToolTransportError: No logs found for receiver R0000000001",
        },
    ]
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text)
    return text.encode()


def test_t2i_atlas_is_disabled_by_default_and_profile_is_fail_closed():
    settings = Settings()
    assert settings.t2i_atlas_enabled is False

    enabled_bad = replace(
        settings,
        t2i_atlas_enabled=True,
        t2i_atlas_profile="candidate_b",
    )
    assert any("must be 'candidate_a'" in x for x in enabled_bad.validate())


def test_effective_configuration_reports_t2i_value_and_source_without_secrets(tmp_path: Path):
    settings = Settings(
        output_dir=tmp_path,
        t2i_atlas_enabled=True,
        t2i_atlas_profile="candidate_a",
        t2i_atlas_font_sha256=LOCAL_FONT_SHA256,
        t2i_atlas_font_bold_sha256=LOCAL_FONT_BOLD_SHA256,
    )
    result = build_effective_configuration_provenance(
        settings,
        process_env={
            "NIGHTLY_RCA_T2I_ATLAS_ENABLED": "true",
            "NIGHTLY_RCA_T2I_ATLAS_PROFILE": "candidate_a",
        },
    )
    assert result["t2i_atlas_enabled"] is True
    assert result["t2i_atlas_profile"] == "candidate_a"
    assert result["configuration_sources"]["t2i_atlas_enabled"] == "PROCESS_ENV"
    assert result["configuration_sources"]["t2i_atlas_profile"] == "PROCESS_ENV"


def test_t2i_renderer_is_deterministic_traceable_and_source_read_only(tmp_path: Path):
    source = tmp_path / "events.jsonl"
    source_bytes = _write_events(source)
    source_before = hashlib.sha256(source_bytes).hexdigest()

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    result_a = generate_t2i_v3_atlas(source, out_a)
    result_b = generate_t2i_v3_atlas(source, out_b)

    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_before
    assert result_a["source_sha256"] == source_before
    assert result_a["source_records"] == 4
    assert result_a["page_count"] >= 1
    assert result_a["profile"] == "candidate_a"

    atlas_a = out_a / "t2i_atlas_v3_a"
    atlas_b = out_b / "t2i_atlas_v3_a"
    manifest_a = json.loads((atlas_a / "t2i_log_atlas_v3_a_manifest.json").read_text())
    manifest_b = json.loads((atlas_b / "t2i_log_atlas_v3_a_manifest.json").read_text())

    # Manifest and bytes are deterministic for the same source/font identity.
    assert manifest_a == manifest_b
    assert (atlas_a / "t2i_log_atlas_v3_a_records.jsonl").read_bytes() == (
        atlas_b / "t2i_log_atlas_v3_a_records.jsonl"
    ).read_bytes()
    for page in manifest_a["pages"]:
        assert (atlas_a / page["name"]).read_bytes() == (atlas_b / page["name"]).read_bytes()

    # Re-running into the same run directory is idempotent and does not
    # overwrite a conflicting representation.
    rerun = generate_t2i_v3_atlas(source, out_a)
    assert rerun["idempotent_existing_match"] is True

    assert manifest_a["integrity"] == {
        "all_event_pages_assigned": True,
        "no_page_exceeds_hard_capacity": True,
        "source_events_indexed_exactly_once": True,
    }
    assert manifest_a["event_index_count"] == 4
    assert [row["label"] for row in manifest_a["event_index"]] == [
        "L0001", "L0002", "L0003", "L0004"
    ]
    assert all(len(row["raw_sha256"]) == 64 for row in manifest_a["event_index"])


def test_t2i_renderer_refuses_font_identity_drift_before_publish(tmp_path: Path):
    source = tmp_path / "events.jsonl"
    _write_events(source)
    with pytest.raises(T2IAtlasError, match="regular font SHA256 drift"):
        generate_t2i_v3_atlas(
            source,
            tmp_path / "out",
            expected_font_sha256="0" * 64,
            expected_font_bold_sha256=LOCAL_FONT_BOLD_SHA256,
        )
    assert not (tmp_path / "out" / "t2i_atlas_v3_a").exists()


def test_t2i_renderer_refuses_malformed_source_without_silent_event_loss(tmp_path: Path):
    source = tmp_path / "events.jsonl"
    source.write_text('{"status":"OK"}\nnot-json\n')
    with pytest.raises(T2IAtlasError, match="malformed source records=1"):
        generate_t2i_v3_atlas(source, tmp_path / "out")
    assert not (tmp_path / "out" / "t2i_atlas_v3_a").exists()




@pytest.mark.asyncio
async def test_pipeline_default_disabled_does_not_generate_atlas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NIGHTLY_RCA_WRITE_AUTHORIZED", "true")
    settings = replace(
        Settings(),
        output_dir=tmp_path,
        commit=True,
        notify=False,
        t2i_atlas_enabled=False,
    )
    state = RunState(
        schema_version=1,
        run_id="t2i-disabled",
        mode="commit",
        role="operator",
        started_at="2026-08-12T00:00:00+00:00",
    )
    pipeline = NightlyPipeline(
        settings=settings,
        client=FakeToolClient(),
        state=state,
    )

    rc = await pipeline.run()

    assert rc == 0
    assert "t2i_atlas" not in state.data
    assert not (pipeline.store.run_dir / "t2i_atlas_v3_a").exists()


@pytest.mark.asyncio
async def test_pipeline_generates_opt_in_atlas_and_report_surfaces_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NIGHTLY_RCA_WRITE_AUTHORIZED", "true")
    settings = replace(
        Settings(),
        output_dir=tmp_path,
        commit=True,
        notify=False,
        t2i_atlas_enabled=True,
        t2i_atlas_profile="candidate_a",
        t2i_atlas_font_sha256=LOCAL_FONT_SHA256,
        t2i_atlas_font_bold_sha256=LOCAL_FONT_BOLD_SHA256,
    )
    state = RunState(
        schema_version=1,
        run_id="t2i-enabled",
        mode="commit",
        role="operator",
        started_at="2026-08-12T00:00:00+00:00",
    )
    pipeline = NightlyPipeline(
        settings=settings,
        client=FakeToolClient(),
        state=state,
    )

    rc = await pipeline.run()

    assert rc == 0
    atlas = state.data["t2i_atlas"]
    assert atlas["status"] == "GENERATED"
    assert atlas["profile"] == "candidate_a"
    assert atlas["source_records"] == len(state.steps)
    assert atlas["page_count"] >= 1
    assert Path(atlas["manifest_path"]).is_file()
    assert all(Path(path).is_file() for path in atlas["page_paths"])

    report = (pipeline.store.run_dir / "FINAL_REPORT.md").read_text()
    assert "## T2I Log Atlas" in report
    assert "candidate_a" in report
    assert atlas["source_sha256"] in report


@pytest.mark.asyncio
async def test_pipeline_atlas_generation_error_is_explicit_and_nonfatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("NIGHTLY_RCA_WRITE_AUTHORIZED", "true")
    import nightly_rca.t2i_atlas as atlas_module

    def fail(*args, **kwargs):
        raise T2IAtlasError("synthetic atlas failure")

    monkeypatch.setattr(atlas_module, "generate_t2i_v3_atlas", fail)

    settings = replace(
        Settings(),
        output_dir=tmp_path,
        commit=True,
        notify=False,
        t2i_atlas_enabled=True,
    )
    state = RunState(
        schema_version=1,
        run_id="t2i-failure",
        mode="commit",
        role="operator",
        started_at="2026-08-12T00:00:00+00:00",
    )
    pipeline = NightlyPipeline(
        settings=settings,
        client=FakeToolClient(),
        state=state,
    )

    rc = await pipeline.run()

    assert rc == 0
    assert state.status == "COMPLETE"
    assert state.data["t2i_atlas"]["status"] == "GENERATION_ERROR"
    assert state.data["t2i_atlas"]["error_class"] == "T2IAtlasError"
    assert "synthetic atlas failure" in state.data["t2i_atlas"]["error"]

    report = (pipeline.store.run_dir / "FINAL_REPORT.md").read_text()
    assert "## T2I Log Atlas" in report
    assert "GENERATION_ERROR" in report
