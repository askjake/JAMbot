import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from apps.nightly_rca import t2i_atlas as base
from apps.nightly_rca.t2i_selective import (
    T2ISelectiveError,
    materialize_event_micro_atlas,
    materialize_global_summary_atlas,
)


def _write_events(path: Path, count: int = 6) -> None:
    rows = []
    for index in range(1, count + 1):
        rows.append(
            {
                "server": "s3_stb_logs",
                "tool": "search_logs" if index % 2 else "read_log",
                "arguments": {
                    "receiver_id": f"R{index:010d}",
                    "date": "2026-08-13",
                },
                "status": "OK",
                "elapsed_ms": 100 + index,
                "response": {
                    "receiver_id": f"R{index:010d}",
                    "result": f"RESULT_{index}",
                    "ok": True,
                    "returned_item_count": index,
                },
            }
        )
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _font_shas() -> tuple[str, str]:
    return base._sha256_path(base.FONT_REG), base._sha256_path(base.FONT_BOLD)


def test_event_micro_atlas_is_one_page_source_immutable_and_idempotent(tmp_path: Path):
    events = tmp_path / "events.jsonl"
    _write_events(events)
    before = hashlib.sha256(events.read_bytes()).hexdigest()
    font, bold = _font_shas()

    first = materialize_event_micro_atlas(
        events,
        tmp_path / "out",
        event_labels=["L0005", "L0001", "L0003"],
        request_id="request-one",
        expected_font_sha256=font,
        expected_font_bold_sha256=bold,
    )
    second = materialize_event_micro_atlas(
        events,
        tmp_path / "out",
        event_labels=["L0005", "L0001", "L0003"],
        request_id="request-two",
        expected_font_sha256=font,
        expected_font_bold_sha256=bold,
    )

    assert hashlib.sha256(events.read_bytes()).hexdigest() == before
    assert first["page_count"] == 1
    assert first["selection"]["labels"] == ["L0005", "L0001", "L0003"]
    assert first["image"]["width"] == 2048
    assert first["image"]["height"] == 2048
    assert first["image"]["font_px"] == 18
    assert second["idempotent_reuse"] is True
    assert first["artifact_key"] == second["artifact_key"]
    assert first["image"]["sha256"] == second["image"]["sha256"]
    manifest = json.loads(Path(first["manifest_path"]).read_text())
    assert "request_id" not in json.dumps(manifest)
    assert [row["label"] for row in manifest["selection"]["event_index"]] == [
        "L0005",
        "L0001",
        "L0003",
    ]

    with Image.open(first["image_path"]) as image:
        assert image.size == (2048, 2048)
        assert image.format == "PNG"


def test_event_micro_atlas_rejects_bad_bounds_and_labels(tmp_path: Path):
    events = tmp_path / "events.jsonl"
    _write_events(events)
    font, bold = _font_shas()

    common = dict(
        events_path=events,
        out_dir=tmp_path / "out",
        request_id="test",
        expected_font_sha256=font,
        expected_font_bold_sha256=bold,
    )
    with pytest.raises(T2ISelectiveError):
        materialize_event_micro_atlas(event_labels=[], **common)
    with pytest.raises(T2ISelectiveError):
        materialize_event_micro_atlas(
            event_labels=[f"L{i:04d}" for i in range(1, 7)], **common
        )
    with pytest.raises(T2ISelectiveError):
        materialize_event_micro_atlas(event_labels=["../state.json"], **common)
    with pytest.raises(T2ISelectiveError):
        materialize_event_micro_atlas(event_labels=["L0001", "L0001"], **common)
    with pytest.raises(T2ISelectiveError):
        materialize_event_micro_atlas(event_labels=["L9999"], **common)


def test_global_summary_is_one_candidate_a_page(tmp_path: Path):
    events = tmp_path / "events.jsonl"
    _write_events(events)
    font, bold = _font_shas()

    result = materialize_global_summary_atlas(
        events,
        tmp_path / "out",
        request_id="global",
        expected_font_sha256=font,
        expected_font_bold_sha256=bold,
    )
    assert result["page_count"] == 1
    assert result["selection"]["mode"] == "global"
    assert result["selection"]["row_count"] <= base.HARD_MAX_ROWS_PER_PAGE
    assert result["image"]["width"] == 2048
    assert result["image"]["height"] == 2048


def test_one_page_bound_fails_closed(monkeypatch, tmp_path: Path):
    events = tmp_path / "events.jsonl"
    _write_events(events)
    font, bold = _font_shas()
    monkeypatch.setattr(base, "HARD_MAX_ROWS_PER_PAGE", 1)
    with pytest.raises(T2ISelectiveError, match="one-page capacity"):
        materialize_event_micro_atlas(
            events,
            tmp_path / "out",
            event_labels=["L0001"],
            request_id="overflow",
            expected_font_sha256=font,
            expected_font_bold_sha256=bold,
        )
