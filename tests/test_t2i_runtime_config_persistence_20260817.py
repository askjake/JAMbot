"""D3R regression tests for persistent guarded T2I runtime config."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from app.agent.runtime_config import (
    CONFIG_PATH_ENV,
    T2I_RUNTIME_KEYS,
    T2IRuntimeConfigError,
    bootstrap_t2i_runtime_env,
    expected_runs_root,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENT_INIT = REPO_ROOT / "app" / "agent" / "__init__.py"
APP_MAIN = REPO_ROOT / "app" / "main.py"

QUALIFIED = {
    "NIGHTLY_RCA_T2I_ATLAS_ENABLED": "false",
    "NIGHTLY_RCA_T2I_SELECTIVE_CONTEXT_ENABLED": "true",
    "NIGHTLY_RCA_T2I_SELECTIVE_ALLOWED_EMAILS": "test.test@dish.com",
    "NIGHTLY_RCA_T2I_SELECTIVE_MAX_EVENTS": "5",
    "NIGHTLY_RCA_T2I_RUNS_ROOT": str(expected_runs_root()),
}


def _clear_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (*T2I_RUNTIME_KEYS, CONFIG_PATH_ENV):
        monkeypatch.delenv(key, raising=False)


def _write_config(
    tmp_path: Path,
    values: dict[str, str] | None = None,
    *,
    mode: int = 0o600,
    extra: str = "",
) -> Path:
    payload = dict(QUALIFIED if values is None else values)
    path = tmp_path / "t2i.env"
    text = "\n".join(f"{key}={value}" for key, value in payload.items()) + "\n"
    if extra:
        text += extra.rstrip("\n") + "\n"
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


def test_missing_file_preserves_safe_disabled_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_env(monkeypatch)
    result = bootstrap_t2i_runtime_env(tmp_path / "missing.env")
    assert result.file_loaded is False
    assert result.selective_enabled is False
    assert result.allowlist_count == 0
    assert result.max_events == 5
    for key in T2I_RUNTIME_KEYS:
        assert key not in os.environ


def test_qualified_file_bootstraps_exact_guarded_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_env(monkeypatch)
    path = _write_config(tmp_path)

    result = bootstrap_t2i_runtime_env(path)

    assert result.file_loaded is True
    assert result.selective_enabled is True
    assert result.allowlist_count == 1
    assert result.max_events == 5
    assert set(result.applied_keys) == set(T2I_RUNTIME_KEYS)
    assert result.preserved_process_keys == ()
    for key, value in QUALIFIED.items():
        assert os.environ[key] == value


def test_explicit_process_env_wins_but_effective_values_still_validate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_env(monkeypatch)
    path = _write_config(tmp_path)
    monkeypatch.setenv("NIGHTLY_RCA_T2I_SELECTIVE_CONTEXT_ENABLED", "false")

    result = bootstrap_t2i_runtime_env(path)

    assert result.selective_enabled is False
    assert "NIGHTLY_RCA_T2I_SELECTIVE_CONTEXT_ENABLED" in result.preserved_process_keys
    assert os.environ["NIGHTLY_RCA_T2I_SELECTIVE_CONTEXT_ENABLED"] == "false"


def test_process_env_cannot_enable_legacy_full_atlas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_env(monkeypatch)
    path = _write_config(tmp_path)
    monkeypatch.setenv("NIGHTLY_RCA_T2I_ATLAS_ENABLED", "true")

    with pytest.raises(T2IRuntimeConfigError, match="must remain false"):
        bootstrap_t2i_runtime_env(path)


def test_config_requires_private_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_env(monkeypatch)
    path = _write_config(tmp_path, mode=0o640)

    with pytest.raises(T2IRuntimeConfigError, match="group/other"):
        bootstrap_t2i_runtime_env(path)


def test_unknown_key_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_env(monkeypatch)
    path = _write_config(tmp_path, extra="NOT_A_T2I_SETTING=value")

    with pytest.raises(T2IRuntimeConfigError, match="unsupported"):
        bootstrap_t2i_runtime_env(path)


def test_shell_syntax_is_never_evaluated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_env(monkeypatch)
    sentinel = tmp_path / "MUST_NOT_EXIST"
    values = dict(QUALIFIED)
    values["NIGHTLY_RCA_T2I_SELECTIVE_ALLOWED_EMAILS"] = (
        f"$(touch{sentinel})@dish.com"
    )
    path = _write_config(tmp_path, values)

    with pytest.raises(T2IRuntimeConfigError, match="unsafe characters"):
        bootstrap_t2i_runtime_env(path)

    assert not sentinel.exists()


@pytest.mark.parametrize("bad_value", ["0", "6", "100", "five"])
def test_max_events_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_value: str
) -> None:
    _clear_runtime_env(monkeypatch)
    values = dict(QUALIFIED)
    values["NIGHTLY_RCA_T2I_SELECTIVE_MAX_EVENTS"] = bad_value
    path = _write_config(tmp_path, values)

    with pytest.raises(T2IRuntimeConfigError, match="within 1..5"):
        bootstrap_t2i_runtime_env(path)


def test_guarded_mode_rejects_multiple_emails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_env(monkeypatch)
    values = dict(QUALIFIED)
    values["NIGHTLY_RCA_T2I_SELECTIVE_ALLOWED_EMAILS"] = (
        "test.test@dish.com,second@dish.com"
    )
    path = _write_config(tmp_path, values)

    with pytest.raises(T2IRuntimeConfigError, match="exactly one"):
        bootstrap_t2i_runtime_env(path)


def test_guarded_mode_rejects_empty_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_env(monkeypatch)
    values = dict(QUALIFIED)
    values["NIGHTLY_RCA_T2I_SELECTIVE_ALLOWED_EMAILS"] = ""
    path = _write_config(tmp_path, values)

    with pytest.raises(T2IRuntimeConfigError, match="exactly one"):
        bootstrap_t2i_runtime_env(path)


def test_runs_root_cannot_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_runtime_env(monkeypatch)
    values = dict(QUALIFIED)
    values["NIGHTLY_RCA_T2I_RUNS_ROOT"] = str(tmp_path / "other-runs")
    path = _write_config(tmp_path, values)

    with pytest.raises(T2IRuntimeConfigError, match="qualified runs directory"):
        bootstrap_t2i_runtime_env(path)


def test_agent_package_bootstraps_before_app_settings_instantiation() -> None:
    init_text = AGENT_INIT.read_text(encoding="utf-8")
    main_text = APP_MAIN.read_text(encoding="utf-8")

    assert "bootstrap_t2i_runtime_env()" in init_text
    first_agent_import = main_text.index("from app.agent.")
    settings_materialization = main_text.index("settings = get_settings()")
    assert first_agent_import < settings_materialization


def test_non_backend_import_does_not_mutate_process_env(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    env = dict(os.environ)
    for key in T2I_RUNTIME_KEYS:
        env.pop(key, None)
    env[CONFIG_PATH_ENV] = str(path)
    proc = subprocess.run(
        [
            os.sys.executable,
            "-c",
            (
                "import os, app.agent; "
                "print(int(app.agent.T2I_RUNTIME_BOOTSTRAP is None), "
                "os.environ.get('NIGHTLY_RCA_T2I_SELECTIVE_CONTEXT_ENABLED', 'UNSET'))"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "1 UNSET"

def test_real_python_import_loads_config_without_shell_sourcing(tmp_path: Path) -> None:
    path = _write_config(tmp_path)
    env = dict(os.environ)
    for key in T2I_RUNTIME_KEYS:
        env.pop(key, None)
    env[CONFIG_PATH_ENV] = str(path)
    proc = subprocess.run(
        [
            os.sys.executable,
            "-c",
            (
                "import os, sys; "
                "sys.argv=['uvicorn', 'app.main:app']; "
                "import app.agent; "
                "r=app.agent.T2I_RUNTIME_BOOTSTRAP; "
                "print(int(r.file_loaded), int(r.selective_enabled), "
                "r.allowlist_count, r.max_events, "
                "os.environ['NIGHTLY_RCA_T2I_SELECTIVE_CONTEXT_ENABLED'])"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "1 1 1 5 true"
