"""Bounded read-only status for guarded T2I runtime persistence.

The payload deliberately reports only booleans/counts. It never exposes the
allowlisted email address, configuration path, run path, or arbitrary process
environment values.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import app.agent as agent_package
from app.agent.runtime_config import T2I_RUNTIME_KEYS, expected_runs_root

SCHEMA = "t2i-runtime-persistence/1.0"


def _env_true(name: str) -> bool:
    return os.getenv(name, "").strip().lower() == "true"


def _allowlist_count() -> int:
    return len(
        {
            item.strip().lower()
            for item in os.getenv(
                "NIGHTLY_RCA_T2I_SELECTIVE_ALLOWED_EMAILS", ""
            ).split(",")
            if item.strip()
        }
    )


def _max_events() -> int | None:
    raw = os.getenv("NIGHTLY_RCA_T2I_SELECTIVE_MAX_EVENTS", "").strip()
    try:
        return int(raw)
    except ValueError:
        return None


def _runs_root_qualified() -> bool:
    raw = os.getenv("NIGHTLY_RCA_T2I_RUNS_ROOT", "").strip()
    if not raw:
        return False
    return (
        Path(raw).resolve(strict=False)
        == expected_runs_root().resolve(strict=False)
    )


def t2i_runtime_persistence_payload() -> dict[str, Any]:
    """Return non-secret evidence that the running backend used local persistence.

    A ``pass`` requires all five qualified keys to have been applied by the
    bootstrap from the dedicated file, with zero keys preserved from the
    process environment. This distinguishes a real persistence restart from a
    process that merely inherited the earlier D3O runtime variables.
    """
    result = getattr(agent_package, "T2I_RUNTIME_BOOTSTRAP", None)
    if result is None:
        return {
            "status": "inactive",
            "schema": SCHEMA,
            "bootstrap_active": False,
        }

    try:
        file_loaded = bool(result.file_loaded)
        applied_key_count = len(result.applied_keys)
        preserved_process_key_count = len(result.preserved_process_keys)
        legacy_full_atlas_enabled = _env_true(
            "NIGHTLY_RCA_T2I_ATLAS_ENABLED"
        )
        selective_enabled = _env_true(
            "NIGHTLY_RCA_T2I_SELECTIVE_CONTEXT_ENABLED"
        )
        allowlist_count = _allowlist_count()
        max_events = _max_events()
        runs_root_qualified = _runs_root_qualified()

        bootstrap_consistent = (
            bool(result.selective_enabled) == selective_enabled
            and int(result.allowlist_count) == allowlist_count
            and int(result.max_events) == max_events
        )
        passed = (
            file_loaded
            and applied_key_count == len(T2I_RUNTIME_KEYS)
            and preserved_process_key_count == 0
            and not legacy_full_atlas_enabled
            and selective_enabled
            and allowlist_count == 1
            and max_events == 5
            and runs_root_qualified
            and bootstrap_consistent
        )

        return {
            "status": "pass" if passed else "fail",
            "schema": SCHEMA,
            "bootstrap_active": True,
            "file_loaded": file_loaded,
            "applied_key_count": applied_key_count,
            "preserved_process_key_count": preserved_process_key_count,
            "legacy_full_atlas_enabled": legacy_full_atlas_enabled,
            "selective_enabled": selective_enabled,
            "allowlist_count": allowlist_count,
            "max_events": max_events,
            "runs_root_qualified": runs_root_qualified,
            "bootstrap_consistent": bootstrap_consistent,
        }
    except Exception as exc:  # fail closed without leaking values/paths
        return {
            "status": "fail",
            "schema": SCHEMA,
            "bootstrap_active": True,
            "error_type": type(exc).__name__,
        }
