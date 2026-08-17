"""D3S pre-restart tests for bounded T2I runtime persistence evidence."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from app.agent.runtime_config import CONFIG_PATH_ENV, T2I_RUNTIME_KEYS, expected_runs_root

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTER = REPO_ROOT / "app" / "agent" / "routers.py"

QUALIFIED = {
    "NIGHTLY_RCA_T2I_ATLAS_ENABLED": "false",
    "NIGHTLY_RCA_T2I_SELECTIVE_CONTEXT_ENABLED": "true",
    "NIGHTLY_RCA_T2I_SELECTIVE_ALLOWED_EMAILS": "test.test@dish.com",
    "NIGHTLY_RCA_T2I_SELECTIVE_MAX_EVENTS": "5",
    "NIGHTLY_RCA_T2I_RUNS_ROOT": str(expected_runs_root()),
}


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "t2i-runtime.env"
    path.write_text(
        "\n".join(f"{key}={value}" for key, value in QUALIFIED.items()) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _subprocess_env(config: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in (*T2I_RUNTIME_KEYS, CONFIG_PATH_ENV):
        env.pop(key, None)
    env[CONFIG_PATH_ENV] = str(config)
    return env


def test_backend_status_proves_all_values_came_from_persistence_file(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    proc = subprocess.run(
        [
            os.sys.executable,
            "-c",
            (
                "import json, sys; "
                "sys.argv=['uvicorn','app.main:app','--host','0.0.0.0','--port','8002']; "
                "from app.agent.t2i_runtime_status import t2i_runtime_persistence_payload; "
                "print(json.dumps(t2i_runtime_persistence_payload(), sort_keys=True))"
            ),
        ],
        cwd=REPO_ROOT,
        env=_subprocess_env(config),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip())

    assert payload == {
        "allowlist_count": 1,
        "applied_key_count": 5,
        "bootstrap_active": True,
        "bootstrap_consistent": True,
        "file_loaded": True,
        "legacy_full_atlas_enabled": False,
        "max_events": 5,
        "preserved_process_key_count": 0,
        "runs_root_qualified": True,
        "schema": "t2i-runtime-persistence/1.0",
        "selective_enabled": True,
        "status": "pass",
    }

    rendered = json.dumps(payload, sort_keys=True)
    assert "test.test@dish.com" not in rendered
    assert str(config) not in rendered
    assert str(expected_runs_root()) not in rendered
    assert "config_path" not in payload


def test_non_backend_import_is_inactive_and_does_not_load_file(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    proc = subprocess.run(
        [
            os.sys.executable,
            "-c",
            (
                "import json, os; "
                "from app.agent.t2i_runtime_status import t2i_runtime_persistence_payload; "
                "print(json.dumps(t2i_runtime_persistence_payload(), sort_keys=True)); "
                "print(os.environ.get('NIGHTLY_RCA_T2I_SELECTIVE_CONTEXT_ENABLED','UNSET'))"
            ),
        ],
        cwd=REPO_ROOT,
        env=_subprocess_env(config),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert json.loads(lines[0]) == {
        "bootstrap_active": False,
        "schema": "t2i-runtime-persistence/1.0",
        "status": "inactive",
    }
    assert lines[1] == "UNSET"


def test_agent_router_exposes_read_only_persistence_status() -> None:
    text = ROUTER.read_text(encoding="utf-8")
    assert '@router.get("/t2i-runtime-persistence")' in text
    assert "t2i_runtime_persistence_payload" in text
    assert "@router.post" not in text
