"""Determinism tests for frontend Node/pnpm selection in restart-dishchat.sh.

Context
-------
Under a clean non-interactive environment (cron, systemd, `ssh host ./restart`)
the first `node` on PATH is /usr/bin/node v12, and the previously hardcoded
Volta pnpm shim also resolves to Node 12. Node 12 cannot parse pnpm 10 (optional
chaining) nor run Next.js 16. The old preflight only performed an existence
check (`[[ -x "$PNPM" ]]`), so the failure surfaced *after* services had already
been stopped.

These tests assert that:
  * the runtime pair is resolved and *executed* during preflight,
  * resolution happens before anything is stopped,
  * an unsupported Node or a broken pnpm is rejected,
  * inherited PATH differences cannot change the selected pair,
  * the selected pair survives helper shells / timeout / background launch,
  * no secret material is printed,
  * existing restart-lock safety is preserved.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "restart-dishchat.sh"

MINIMAL_PATH = "/usr/local/bin:/usr/bin:/bin"

# A PATH deliberately poisoned with Volta and an old NVM Node, mimicking an
# interactive login shell that would otherwise win.
POISONED_PATH = (
    "/home/montjac/.volta/bin:"
    "/home/montjac/.nvm/versions/node/v14.21.3/bin:"
    "/usr/local/bin:/usr/bin:/bin"
)


@pytest.fixture(scope="module")
def script_text() -> str:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"
    return SCRIPT.read_text(encoding="utf-8")


def _minimal_env(**overrides: str) -> dict:
    env = {
        "HOME": os.environ.get("HOME", "/home/montjac"),
        "USER": os.environ.get("USER", "montjac"),
        "PATH": MINIMAL_PATH,
    }
    env.update(overrides)
    return env


def _source_and_run(snippet: str, env: dict, timeout: int = 180):
    """Source the restart script in a clean shell and run a snippet."""
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", f"source '{SCRIPT}'\n{snippet}"],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Static structure
# ---------------------------------------------------------------------------

def test_script_is_syntactically_valid() -> None:
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_no_hardcoded_volta_runtime_remains(script_text: str) -> None:
    assert "/home/montjac/.volta" not in script_text, (
        "a hardcoded Volta path reappeared; Volta pins Node 12 here"
    )
    assert 'PNPM="/home/montjac/.volta/bin/pnpm"' not in script_text


def test_existence_only_pnpm_check_is_gone(script_text: str) -> None:
    assert '[[ -x "${PNPM}" ]]' not in script_text, (
        "existence-only pnpm check must be replaced by real execution"
    )


def test_resolver_is_invoked_during_preflight(script_text: str) -> None:
    assert "resolve_frontend_runtime || {" in script_text


def test_resolution_happens_before_anything_is_stopped(script_text: str) -> None:
    """The pair must be validated while all services are still running."""
    resolve_at = script_text.index("resolve_frontend_runtime || {")
    for stopper in ("stop_legacy_service\n", "stop_core_services\n"):
        # locate the call site inside main(), not the function definition
        call_idx = script_text.index("    " + stopper)
        assert resolve_at < call_idx, (
            f"resolve_frontend_runtime must run before {stopper.strip()}"
        )


def test_check_mode_exits_before_stopping_services(script_text: str) -> None:
    check_at = script_text.index('if [[ "${CHECK_ONLY}" -eq 1 ]]; then')
    stop_at = script_text.index("    stop_legacy_service\n")
    assert check_at < stop_at


def test_frontend_launch_uses_explicit_verified_executables(script_text: str) -> None:
    assert '"${NODE_BIN}" "${PNPM_ENTRY}" exec next dev' in script_text, (
        "frontend must be launched with the exact validated node+pnpm pair"
    )
    assert 'PATH="$(frontend_runtime_path "${NODE_BIN}")"' in script_text, (
        "the launch must pin PATH to the verified node directory"
    )


def test_frontend_launch_still_closes_restart_lock_fd(script_text: str) -> None:
    """Regression guard: the Phase-earlier lock-leak fix must survive."""
    launch = script_text[script_text.index("start_frontend() {"):]
    launch = launch[: launch.index("\n}\n")]
    assert "9>&-" in launch, "frontend launch must not inherit the restart lock fd"


def test_start_frontend_refuses_to_run_unresolved(script_text: str) -> None:
    assert "Frontend runtime not resolved" in script_text


def test_new_command_dependencies_are_declared(script_text: str) -> None:
    require_line = next(
        line for line in script_text.splitlines() if "for cmd in" in line
    )
    for cmd in ("awk", "env", "sort", "id"):
        assert f" {cmd} " in require_line, f"{cmd} must be declared as a dependency"


def test_script_is_sourceable_without_running_main(script_text: str) -> None:
    assert 'if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then' in script_text


# ---------------------------------------------------------------------------
# Requirement derivation
# ---------------------------------------------------------------------------

def test_required_node_major_is_derived_from_frontend_package() -> None:
    proc = _source_and_run(
        'frontend_required_node_major "${FRONTEND_PKG_ROOT}"', _minimal_env()
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().isdigit()
    assert int(proc.stdout.strip()) >= 20


def test_required_node_major_falls_back_when_metadata_missing(tmp_path) -> None:
    proc = _source_and_run(
        f'frontend_required_node_major "{tmp_path}/nope.json"', _minimal_env()
    )
    assert proc.stdout.strip() == "20"


def test_required_node_major_reads_a_synthetic_engines_block(tmp_path) -> None:
    pkg = tmp_path / "package.json"
    pkg.write_text('{\n "engines": {\n  "node": ">=22"\n }\n}\n', encoding="utf-8")
    proc = _source_and_run(f'frontend_required_node_major "{pkg}"', _minimal_env())
    assert proc.stdout.strip() == "22"


# ---------------------------------------------------------------------------
# Functional resolution
# ---------------------------------------------------------------------------

def _assert_failed_without_destructive_action(proc) -> None:
    # A resolver failure must abort with status 2 and touch no database.
    #
    # restart-dishchat.sh runs with `set -Eeuo pipefail` plus an ERR trap, so a
    # non-zero return from resolve_frontend_runtime terminates the shell. That is
    # the desired behaviour: it happens during preflight, before any service is
    # stopped.
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"expected exit 2, got {proc.returncode}: {combined}"
    assert "PostgreSQL was not reset, recreated, removed, or stopped." in combined, (
        "fast failure must confirm no destructive database action"
    )



def test_minimal_noninteractive_environment_selects_supported_node() -> None:
    """The core regression: a clean shell must NOT end up on Node 12."""
    proc = _source_and_run(
        'resolve_frontend_runtime >/dev/null || exit 1\n'
        'printf "%s|%s|%s|%s\\n" "$NODE_BIN" "$NODE_VERSION" "$PNPM_ENTRY" "$PNPM_VERSION"',
        _minimal_env(),
    )
    assert proc.returncode == 0, proc.stderr
    node_bin, node_version, pnpm_entry, pnpm_version = proc.stdout.strip().split("|")

    assert node_bin.startswith("/"), "node must be an absolute path"
    assert pnpm_entry.startswith("/"), "pnpm must be an absolute path"
    assert "/.volta/" not in node_bin
    assert "/.volta/" not in pnpm_entry

    major = int(re.match(r"v(\d+)\.", node_version).group(1))
    assert major >= 20, f"selected unsupported Node {node_version}"
    assert re.match(r"^\d+\.\d+\.\d+", pnpm_version)


def test_selected_node_actually_runs_the_reported_version() -> None:
    proc = _source_and_run(
        'resolve_frontend_runtime >/dev/null || exit 1\n'
        '"$NODE_BIN" --version',
        _minimal_env(),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().startswith("v")


def test_inherited_path_differences_do_not_change_the_selected_pair() -> None:
    snippet = (
        'resolve_frontend_runtime >/dev/null || exit 1\n'
        'printf "%s|%s|%s\\n" "$NODE_BIN" "$NODE_VERSION" "$PNPM_VERSION"'
    )
    clean = _source_and_run(snippet, _minimal_env())
    poisoned = _source_and_run(snippet, _minimal_env(PATH=POISONED_PATH))

    assert clean.returncode == 0, clean.stderr
    assert poisoned.returncode == 0, poisoned.stderr
    assert clean.stdout.strip() == poisoned.stdout.strip(), (
        "a Volta/old-Node PATH changed the selected runtime"
    )


def test_node_below_required_major_is_rejected() -> None:
    legacy = "/usr/bin/node"
    if not os.access(legacy, os.X_OK):
        pytest.skip("no system node available to act as an unsupported version")
    ver = subprocess.run([legacy, "--version"], capture_output=True, text=True)
    if ver.returncode != 0 or int(re.match(r"v(\d+)\.", ver.stdout).group(1)) >= 20:
        pytest.skip("system node is not an unsupported version on this host")

    proc = _source_and_run(
        "resolve_frontend_runtime",
        _minimal_env(DISHCHAT_NODE_BIN=legacy),
    )
    _assert_failed_without_destructive_action(proc)
    combined = proc.stdout + proc.stderr
    assert "required major" in combined


def test_volta_managed_node_is_rejected_even_when_requested() -> None:
    volta_node = "/home/montjac/.volta/bin/node"
    if not os.path.exists(volta_node):
        pytest.skip("Volta is not installed on this host")
    proc = _source_and_run(
        "resolve_frontend_runtime",
        _minimal_env(DISHCHAT_NODE_BIN=volta_node),
    )
    _assert_failed_without_destructive_action(proc)
    assert "Volta-managed" in (proc.stdout + proc.stderr)


def test_explicit_valid_node_override_is_honoured() -> None:
    discover = _source_and_run(
        'resolve_frontend_runtime >/dev/null || exit 1; printf "%s" "$NODE_BIN"',
        _minimal_env(),
    )
    assert discover.returncode == 0, discover.stderr
    chosen = discover.stdout.strip()

    proc = _source_and_run(
        'resolve_frontend_runtime >/dev/null || exit 1; printf "%s" "$NODE_BIN"',
        _minimal_env(DISHCHAT_NODE_BIN=chosen),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == chosen


def test_broken_pnpm_pair_is_rejected(tmp_path) -> None:
    broken = tmp_path / "broken_pnpm.cjs"
    broken.write_text(
        '#!/usr/bin/env node\nthrow new Error("broken pnpm");\n', encoding="utf-8"
    )
    proc = _source_and_run(
        "resolve_frontend_runtime",
        _minimal_env(DISHCHAT_PNPM_BIN=str(broken)),
    )
    _assert_failed_without_destructive_action(proc)
    assert "DISHCHAT_PNPM_BIN" in (proc.stdout + proc.stderr)


def test_nonexistent_pnpm_override_is_rejected(tmp_path) -> None:
    proc = _source_and_run(
        "resolve_frontend_runtime",
        _minimal_env(DISHCHAT_PNPM_BIN=str(tmp_path / "absent-pnpm")),
    )
    _assert_failed_without_destructive_action(proc)


def test_validated_pair_survives_timeout_and_nested_helper_shell() -> None:
    """The launch path goes through setsid/nohup/env and a subshell."""
    if shutil.which("timeout") is None:
        pytest.skip("timeout(1) unavailable")
    proc = _source_and_run(
        'resolve_frontend_runtime >/dev/null || exit 1\n'
        'timeout 90 env PATH="$(frontend_runtime_path "$NODE_BIN")" '
        'bash --noprofile --norc -c '
        '"cd \\"$FRONTEND_DIR\\" && \\"$NODE_BIN\\" \\"$PNPM_ENTRY\\" --version"',
        _minimal_env(),
    )
    assert proc.returncode == 0, proc.stderr
    assert re.match(r"^\d+\.\d+\.\d+", proc.stdout.strip()), proc.stdout


def test_pair_is_consistent_between_rich_and_minimal_environments() -> None:
    """resolve_frontend_runtime validates in both env shapes; versions must agree."""
    proc = _source_and_run(
        'resolve_frontend_runtime >/dev/null || exit 1\n'
        'a="$(validate_frontend_runtime_pair "$NODE_BIN" "$PNPM_ENTRY")"\n'
        'b="$(validate_frontend_runtime_pair_minimal_env "$NODE_BIN" "$PNPM_ENTRY")"\n'
        'printf "%s|%s\\n" "$a" "$b"',
        _minimal_env(),
    )
    assert proc.returncode == 0, proc.stderr
    rich, minimal = proc.stdout.strip().split("|")
    assert rich == minimal != ""


# ---------------------------------------------------------------------------
# Secret hygiene
# ---------------------------------------------------------------------------

def test_resolver_output_contains_no_secret_material() -> None:
    proc = _source_and_run(
        "resolve_frontend_runtime",
        _minimal_env(
            FAKE_WEBHOOK_URL="https://example.invalid/hook/SHOULD-NOT-APPEAR",
            FAKE_TOKEN="SHOULD-NOT-APPEAR-TOKEN",
        ),
    )
    combined = proc.stdout + proc.stderr
    assert "SHOULD-NOT-APPEAR" not in combined
    for pattern in (
        r"postgres(ql)?://",
        r"AKIA[0-9A-Z]{8}",
        r"hooks\.google",
        r"chat\.googleapis",
        r"Bearer\s+[A-Za-z0-9._-]{12,}",
    ):
        assert not re.search(pattern, combined, re.IGNORECASE), pattern


def test_resolver_does_not_dump_the_environment() -> None:
    proc = _source_and_run(
        "resolve_frontend_runtime",
        _minimal_env(UNIQUE_ENV_CANARY="CANARY-VALUE-1234"),
    )
    assert "CANARY-VALUE-1234" not in (proc.stdout + proc.stderr)
