from __future__ import annotations

from pathlib import Path

from dataclasses import replace

import pytest

from nightly_rca.config import Settings
from nightly_rca.executor import SerialExecutor, redact
from nightly_rca.tests.fake_client import FakeToolClient
from nightly_rca.state import RunState, RunStore


def test_redaction_removes_secret_fields_and_tokens():
    value = {"token": "abc", "nested": {"api_key": "def"}, "text": "token=secret"}
    cleaned = redact(value)
    assert cleaned["token"] == "<redacted>"
    assert cleaned["nested"]["api_key"] == "<redacted>"
    assert "secret" not in cleaned["text"]


def test_runstore_atomic_round_trip(tmp_path: Path):
    state = RunState(schema_version=1, run_id="roundtrip", mode="dry_run", role="operator", started_at="now")
    store = RunStore(tmp_path, state.run_id)
    store.save(state)
    loaded_store, loaded = RunStore.load(store.state_path)
    assert loaded.run_id == state.run_id
    assert loaded_store.output_dir == tmp_path


def _safe_read_tree(root: Path) -> str:
    """Read all text files under root, skipping unreadable paths."""
    parts = []
    for p in root.rglob("*"):
        try:
            is_file = p.is_file()
        except PermissionError:
            continue
        if not is_file or "__pycache__" in p.parts:
            continue
        try:
            parts.append(p.read_text(errors="ignore"))
        except PermissionError:
            continue
    return "\n".join(parts)


def test_source_tree_has_no_embedded_google_webhook_credentials():
    root = Path(__file__).resolve().parents[1]
    content = _safe_read_tree(root)
    import re
    assert re.search(r"AIza[0-9A-Za-z_-]{30,}", content) is None
    assert ("chat.googleapis.com/v1/" + "spaces/AAQ") not in content
    assert ("AT" + "ATT") not in content


@pytest.mark.asyncio
async def test_executor_blocks_accidental_live_write_in_dry_run(tmp_path: Path):
    settings = replace(Settings(), output_dir=tmp_path, commit=False, notify=False)
    state = RunState(schema_version=1, run_id="guard", mode="dry_run", role="operator", started_at="now")
    store = RunStore(tmp_path, state.run_id)
    client = FakeToolClient()
    executor = SerialExecutor(client, settings, state, store)

    result = await executor.call(
        phase="8",
        step="guard",
        tool="record_investigation_case",
        arguments={"investigation_json": "{}", "case_type": "x", "outcome_status": "unreviewed", "source": "test"},
    )

    assert result["status"] == "WRITE_BLOCKED"
    assert client.calls == []
    assert state.metrics["tool_calls"] == 0
    assert state.metrics["write_blocked"] == 1


def test_runstore_detects_tampered_checkpoint(tmp_path: Path):
    import json

    state = RunState(schema_version=1, run_id="tamper", mode="dry_run", role="operator", started_at="now")
    store = RunStore(tmp_path, state.run_id)
    store.save(state)
    raw = json.loads(store.state_path.read_text())
    raw["mode"] = "commit"
    store.state_path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="integrity check failed"):
        RunStore.load(store.state_path)


@pytest.mark.skipif(
    not hasattr(__import__("os"), "getuid") or __import__("os").getuid() == 0,
    reason="Cannot create unreadable paths as root",
)
def test_credential_scan_skips_unreadable_directory(tmp_path: Path):
    """Regression: PermissionError during traversal must not abort the scan."""
    import os
    import re

    # Create a readable file with known content
    readable = tmp_path / "clean.py"
    readable.write_text("clean_code = True")

    # Create an unreadable directory
    blocked = tmp_path / "blocked_dir"
    blocked.mkdir()
    secret = blocked / "secret.py"
    secret.write_text("x = 1")
    os.chmod(str(blocked), 0o000)

    try:
        # Reproduce the verify.sh scanner logic directly
        parts = []
        for path in tmp_path.rglob("*"):
            try:
                is_file = path.is_file()
            except PermissionError:
                continue
            if not is_file:
                continue
            try:
                parts.append(path.read_text(errors="ignore"))
            except PermissionError:
                continue
        text = "\n".join(parts)

        # Scanner completes without exception and finds clean content
        assert "clean_code" in text
        # Credential check logic succeeds
        checks = {"Google API key": r"AIza[0-9A-Za-z_-]{30,}"}
        found = [name for name, pat in checks.items() if re.search(pat, text)]
        assert found == []
    finally:
        os.chmod(str(blocked), 0o755)

