"""Regression tests: the restart lock (fd 9) must never be inherited by daemons.

Background
----------
restart-dishchat.sh serialises runs with an advisory ``flock`` on fd 9.
Any process that inherits fd 9 and outlives the restart shell keeps that lock
held, which permanently blocks every later run with
"Another restart-dishchat.sh run is already active".

A previous fix closed fd 9 on the five direct ``nohup setsid`` spawns, but the
DVA gateway was launched through a *helper script* that daemonises its own
child.  The helper inherited fd 9 and passed it to the daemon, so the leak
survived.  These tests cover that indirect path.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "restart-dishchat.sh"
LOCK_CLOSE = "9>&-"


@pytest.fixture(scope="module")
def script_text() -> str:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    return SCRIPT.read_text(encoding="utf-8")


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Join backslash-continued lines into single logical lines."""
    out: list[tuple[int, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        start = i + 1
        buf = [lines[i]]
        while lines[i].rstrip().endswith("\\") and i + 1 < len(lines):
            i += 1
            buf.append(lines[i])
        out.append((start, "\n".join(buf)))
        i += 1
    return out


# Command forms that can outlive the restart shell or hand fds to a daemon.
# Matched at *command position* so that help text, variable assignments and
# synchronous inspection commands (e.g. "timeout 25 kubectl get svc") are not
# mistaken for long-lived launches.
LAUNCH_AT_COMMAND_POSITION = re.compile(
    r"^(?:"
    r"nohup\b"
    r"|setsid\b"
    r"|ssh\b"
    r"|(?:timeout\s+\d+\s+)?(?:bash|sh)\s+\S*\.sh\b"
    r"|(?:timeout\s+\d+\s+)?(?:bash|sh)\s+\"?\$\{"
    r"|timeout\s+\d+\s+(?:ssh|python3?)\b"
    r")"
)

# Leading shell keywords that may precede the real command.
LEADING_KEYWORDS = re.compile(r"^\s*(?:if|elif|while|until|then|do|!)\s+")

ASSIGNMENT = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*=")

# "for cmd in bash curl ... nohup ..." is a dependency word list, not a launch.
FOR_LIST = re.compile(r"^\s*for\s+\w+\s+in\b")

SPAWN_WORD = re.compile(r"\b(?:nohup|setsid)\b")


def _command_text(logical: str) -> str:
    """Strip leading shell keywords so the command word comes first."""
    text = logical.strip()
    while True:
        stripped = LEADING_KEYWORDS.sub("", text)
        if stripped == text:
            return stripped
        text = stripped


def _long_lived_sites(text: str) -> list[tuple[int, str]]:
    """Launch sites whose process (or a descendant) may outlive the script."""
    sites = []
    for lineno, logical in _logical_lines(text):
        stripped = logical.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if FOR_LIST.match(logical):
            continue
        command = _command_text(logical)
        # Checked before ASSIGNMENT so "VAR=x nohup cmd &" is still a launch.
        if SPAWN_WORD.search(command):
            sites.append((lineno, logical))
            continue
        if ASSIGNMENT.match(command):
            continue
        if LAUNCH_AT_COMMAND_POSITION.match(command):
            sites.append((lineno, logical))
    return sites


def test_every_long_lived_launch_site_closes_lock_fd(script_text: str) -> None:
    """Direct spawns AND daemonising helpers must close fd 9."""
    offenders = []
    for lineno, logical in _long_lived_sites(script_text):
        if LOCK_CLOSE not in logical:
            offenders.append((lineno, " ".join(logical.split())[:110]))
    assert not offenders, (
        "launch site(s) may leak restart lock fd 9; add '9>&-':\n"
        + "\n".join(f"  line {n}: {t}" for n, t in offenders)
    )


def test_dva_gateway_helper_invocation_closes_lock_fd(script_text: str) -> None:
    """Explicit guard for the exact previously-leaking invocation."""
    match = re.search(
        r"start_dva_services\(\)\s*\{(.*?)\n\}", script_text, re.S
    )
    assert match, "start_dva_services function not found"
    body = match.group(1)
    assert "timeout 60 bash" in body, "DVA gateway helper launch not found"
    assert LOCK_CLOSE in body, (
        "DVA gateway helper must close fd 9 so its daemonised child "
        "cannot hold the restart lock"
    )


def test_helper_paths_are_overridable(script_text: str) -> None:
    """Overridable paths let this regression be tested without touching prod."""
    assert re.search(r'DVA_GATEWAY_START="\$\{DVA_GATEWAY_START:-', script_text)
    assert re.search(r'DVA_GATEWAY_LOG="\$\{DVA_GATEWAY_LOG:-', script_text)


def _extract_function(text: str, name: str) -> str:
    match = re.search(rf"{name}\(\)\s*\{{.*?\n\}}", text, re.S)
    assert match, f"{name} not found in {SCRIPT}"
    return match.group(0)


def _second_acquisition_succeeds(lock: Path) -> bool:
    """True if a fresh process can take the flock (i.e. nobody holds it)."""
    proc = subprocess.run(
        ["bash", "-c", f'exec 9>"{lock}"; flock -n 9'],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.returncode == 0


def _fds_pointing_at(pid: int, target: Path) -> list[str]:
    resolved = os.path.realpath(target)
    found = []
    fd_dir = Path("/proc") / str(pid) / "fd"
    if not fd_dir.is_dir():
        return found
    for fd in fd_dir.iterdir():
        try:
            if os.path.realpath(fd) == resolved:
                found.append(fd.name)
        except OSError:
            continue
    return found


def _run_daemonizing_helper_harness(tmp_path: Path, *, close_fd: bool):
    """Drive the REAL start_dva_services against a fake daemonising helper.

    Returns (daemon_pid, lock_path). The caller must terminate the daemon.
    """
    lock = tmp_path / "restart.lock"
    dva_log = tmp_path / "dva.log"
    pid_file = tmp_path / "daemon.pid"

    # A real daemon process, written as its own file to keep quoting simple.
    daemon_py = tmp_path / "fake_daemon.py"
    daemon_py.write_text(
        "import os, sys, time\n"
        "open(sys.argv[1], \"w\").write(str(os.getpid()))\n"
        "time.sleep(120)\n"
    )

    # Fake helper mirrors the real one: it daemonises a child and exits 0.
    helper = tmp_path / "start-dva-gateway.sh"
    helper.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "nohup python3 " + str(daemon_py) + " " + str(pid_file)
        + " >>" + str(dva_log) + " 2>&1 &\n"
        "exit 0\n"
    )
    helper.chmod(0o755)

    # Hermetic stub so the harness never performs a real remote ssh call.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "ssh").chmod(0o755)

    function_text = _extract_function(
        SCRIPT.read_text(encoding="utf-8"), "start_dva_services"
    )
    if not close_fd:
        # Negative control: reproduce the pre-fix (leaking) form.
        function_text = function_text.replace(" " + LOCK_CLOSE, "")

    harness = tmp_path / "harness.sh"
    harness.write_text(
        "set -Eeuo pipefail\n"
        "log_info() { :; }\nlog_warn() { :; }\nsection() { :; }\n"
        f'DVA_GATEWAY_START="{helper}"\n'
        f'DVA_GATEWAY_LOG="{dva_log}"\n'
        f'exec 9>"{lock}"\n'
        "flock -n 9 || exit 90\n"
        f"{function_text}\n"
        "start_dva_services\n"
    )

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    proc = subprocess.run(
        ["bash", str(harness)],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert proc.returncode == 0, f"harness failed: {proc.stdout}{proc.stderr}"

    deadline = time.time() + 30
    while time.time() < deadline and not pid_file.is_file():
        time.sleep(0.1)
    assert pid_file.is_file(), "fake helper did not daemonise a child"
    daemon_pid = int(pid_file.read_text().strip())

    # The daemon must survive its parent helper, like the real gateway does.
    os.kill(daemon_pid, 0)
    return daemon_pid, lock


def test_daemonized_helper_child_does_not_inherit_restart_lock(tmp_path) -> None:
    """The fixed script: daemon stays alive but holds no lock."""
    daemon_pid, lock = _run_daemonizing_helper_harness(tmp_path, close_fd=True)
    try:
        os.kill(daemon_pid, 0)  # still running
        held = _fds_pointing_at(daemon_pid, lock)
        assert not held, (
            f"daemonised child {daemon_pid} inherited the restart lock "
            f"on fd(s) {held}"
        )
        assert _second_acquisition_succeeds(lock), (
            "restart lock could not be reacquired while the daemon runs"
        )
    finally:
        try:
            os.kill(daemon_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def test_negative_control_prefix_form_would_leak_the_lock(tmp_path) -> None:
    """Proves the test is meaningful: the pre-fix form really does leak."""
    daemon_pid, lock = _run_daemonizing_helper_harness(tmp_path, close_fd=False)
    try:
        held = _fds_pointing_at(daemon_pid, lock)
        assert held, (
            "negative control did not reproduce the leak; the test would "
            "not detect a regression"
        )
        assert not _second_acquisition_succeeds(lock), (
            "negative control should block reacquisition"
        )
    finally:
        try:
            os.kill(daemon_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
