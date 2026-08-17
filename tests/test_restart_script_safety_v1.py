"""Safety tests for restart-dishchat.sh.

These tests enforce the non-negotiable operational guarantees of the restart
script. They are deliberately static plus one fail-fast behavioural check; the
destructive recovery code paths are asserted ABSENT rather than exercised.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "restart-dishchat.sh"

# Commands that must never appear: they destroy or recreate database state.
FORBIDDEN_PATTERNS = [
    r"pg_resetwal",
    r"docker\s+volume\s+rm",
    r"\bdropdb\b",
    r"\bcreatedb\b",
    r"DROP\s+DATABASE",
    r"docker\s+compose\s+up",
]


@pytest.fixture(scope="module")
def script_text() -> str:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    return SCRIPT.read_text(encoding="utf-8")


def test_script_is_syntactically_valid() -> None:
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_strict_mode_enabled(script_text: str) -> None:
    assert re.search(r"^set -Eeuo pipefail$", script_text, re.M), \
        "restart script must enable strict mode (set -Eeuo pipefail)"


def test_no_destructive_database_commands(script_text: str) -> None:
    """Destructive PostgreSQL recovery must not exist in the script at all."""
    offenders = []
    for lineno, line in enumerate(script_text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # comments may document that we never do these things
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, stripped, re.I):
                offenders.append((lineno, pattern))
    assert not offenders, f"destructive command(s) present: {offenders}"


def test_install_dir_is_overridable_and_guarded(script_text: str) -> None:
    assert re.search(r'INSTALL_DIR="\$\{INSTALL_DIR:-', script_text), \
        "INSTALL_DIR must be environment-overridable for testability"
    assert re.search(r'\[\[ -d "\$\{INSTALL_DIR\}" \]\]', script_text), \
        "INSTALL_DIR must be validated before use"


def test_backend_does_not_use_reload(script_text: str) -> None:
    """--reload hot-reloads mid-deploy and must not be used for the service."""
    for line in script_text.splitlines():
        if "uvicorn app.main:app" in line:
            assert "--reload" not in line, f"backend must not use --reload: {line.strip()}"


def test_check_only_mode_exists(script_text: str) -> None:
    assert "--check" in script_text and "CHECK_ONLY" in script_text


def test_missing_install_dir_fails_fast_without_destructive_action() -> None:
    """A missing install directory must abort immediately, not 'recover'."""
    with tempfile.TemporaryDirectory() as tmp:
        missing = os.path.join(tmp, "definitely-absent")
        env = dict(os.environ)
        env["INSTALL_DIR"] = missing
        env["START_OPTIONAL_SERVICES"] = "0"
        proc = subprocess.run(
            ["bash", str(SCRIPT), "--check"],
            capture_output=True, text=True, env=env, timeout=120,
        )
        assert proc.returncode != 0, "missing install dir must not succeed"
        combined = (proc.stdout + proc.stderr).lower()
        assert "missing" in combined or "does not exist" in combined
        for pattern in ("pg_resetwal", "volume rm"):
            assert pattern not in combined, f"destructive action attempted: {pattern}"
        assert not os.path.isdir(missing), "must not create the install directory"


def test_lock_file_is_overridable(script_text: str) -> None:
    assert re.search(r'LOCK_FILE="\$\{LOCK_FILE:-', script_text), \
        "LOCK_FILE must be environment-overridable"


def test_background_children_do_not_inherit_lock_fd(script_text: str) -> None:
    """fd 9 holds the flock; children must not inherit it.

    A long-lived background child that inherits fd 9 keeps the advisory lock
    held after the parent exits, which permanently blocks every later run.
    """
    lines = script_text.splitlines()
    spawn_lines = [i for i, ln in enumerate(lines) if "nohup setsid" in ln]
    assert spawn_lines, "expected background spawn sites"
    offenders = []
    for i in spawn_lines:
        # a spawn may be line-continued; collect until the line ending in '&'
        block = []
        j = i
        while j < len(lines):
            block.append(lines[j])
            if lines[j].rstrip().endswith("&") and not lines[j].rstrip().endswith("\\"):
                break
            j += 1
        text = "\n".join(block)
        if "9>&-" not in text:
            offenders.append(i + 1)
    assert not offenders, f"background spawns inherit lock fd 9 at lines {offenders}"
