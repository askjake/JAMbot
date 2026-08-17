"""Fail-closed bootstrap for guarded T2I runtime configuration.

The production backend is intentionally started without shell-sourcing an env
file. This module provides a narrow persistence bridge for the five qualified
Nightly-RCA T2I settings. It never evaluates shell syntax and never imports
arbitrary keys from the local configuration file.
"""
from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

CONFIG_PATH_ENV = "DISHCHAT_T2I_RUNTIME_CONFIG_FILE"
DEFAULT_CONFIG_NAME = ".t2i-runtime.env"
MAX_CONFIG_BYTES = 16 * 1024

T2I_RUNTIME_KEYS = (
    "NIGHTLY_RCA_T2I_ATLAS_ENABLED",
    "NIGHTLY_RCA_T2I_SELECTIVE_CONTEXT_ENABLED",
    "NIGHTLY_RCA_T2I_SELECTIVE_ALLOWED_EMAILS",
    "NIGHTLY_RCA_T2I_SELECTIVE_MAX_EVENTS",
    "NIGHTLY_RCA_T2I_RUNS_ROOT",
)
_ALLOWED_KEY_SET = frozenset(T2I_RUNTIME_KEYS)
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9_@%+.,:/-]*$")


class T2IRuntimeConfigError(RuntimeError):
    """Persistent T2I configuration is invalid or unsafe."""


@dataclass(frozen=True)
class T2IRuntimeBootstrapResult:
    config_path: str
    file_loaded: bool
    applied_keys: tuple[str, ...]
    preserved_process_keys: tuple[str, ...]
    selective_enabled: bool
    allowlist_count: int
    max_events: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return _repo_root() / DEFAULT_CONFIG_NAME


def expected_runs_root() -> Path:
    return _repo_root() / "var" / "nightly_rca_v6" / "runs"


def _bool_value(name: str, raw: str) -> bool:
    if raw not in {"true", "false"}:
        raise T2IRuntimeConfigError(f"{name} must be exactly 'true' or 'false'")
    return raw == "true"


def _validate_complete_values(values: Mapping[str, str]) -> tuple[bool, int, int]:
    missing = [name for name in T2I_RUNTIME_KEYS if name not in values]
    if missing:
        raise T2IRuntimeConfigError(
            "missing required T2I runtime key(s): " + ", ".join(missing)
        )

    atlas_enabled = _bool_value(
        "NIGHTLY_RCA_T2I_ATLAS_ENABLED",
        values["NIGHTLY_RCA_T2I_ATLAS_ENABLED"],
    )
    if atlas_enabled:
        raise T2IRuntimeConfigError(
            "NIGHTLY_RCA_T2I_ATLAS_ENABLED must remain false for guarded selective mode"
        )

    selective_enabled = _bool_value(
        "NIGHTLY_RCA_T2I_SELECTIVE_CONTEXT_ENABLED",
        values["NIGHTLY_RCA_T2I_SELECTIVE_CONTEXT_ENABLED"],
    )

    max_raw = values["NIGHTLY_RCA_T2I_SELECTIVE_MAX_EVENTS"]
    if not re.fullmatch(r"[1-5]", max_raw):
        raise T2IRuntimeConfigError(
            "NIGHTLY_RCA_T2I_SELECTIVE_MAX_EVENTS must be an integer within 1..5"
        )
    max_events = int(max_raw)

    runs_raw = values["NIGHTLY_RCA_T2I_RUNS_ROOT"]
    runs_path = Path(runs_raw)
    if not runs_path.is_absolute():
        raise T2IRuntimeConfigError("NIGHTLY_RCA_T2I_RUNS_ROOT must be absolute")
    if runs_path.resolve(strict=False) != expected_runs_root().resolve(strict=False):
        raise T2IRuntimeConfigError(
            "NIGHTLY_RCA_T2I_RUNS_ROOT must remain the local qualified runs directory"
        )

    raw_allowlist = values["NIGHTLY_RCA_T2I_SELECTIVE_ALLOWED_EMAILS"]
    emails = tuple(
        item.strip().lower()
        for item in raw_allowlist.split(",")
        if item.strip()
    )
    if selective_enabled and len(emails) != 1:
        raise T2IRuntimeConfigError(
            "guarded selective mode requires exactly one allowlisted email"
        )
    if not selective_enabled and len(emails) > 1:
        raise T2IRuntimeConfigError(
            "guarded selective mode permits at most one allowlisted email"
        )
    for email in emails:
        if not _EMAIL_RE.fullmatch(email):
            raise T2IRuntimeConfigError(
                "NIGHTLY_RCA_T2I_SELECTIVE_ALLOWED_EMAILS contains an invalid email"
            )

    return selective_enabled, len(emails), max_events


def _read_config_file(path: Path) -> dict[str, str]:
    if path.is_symlink():
        raise T2IRuntimeConfigError("T2I runtime config must not be a symlink")
    try:
        st = path.stat()
    except FileNotFoundError:
        return {}
    if not stat.S_ISREG(st.st_mode):
        raise T2IRuntimeConfigError("T2I runtime config must be a regular file")
    if st.st_size > MAX_CONFIG_BYTES:
        raise T2IRuntimeConfigError(
            f"T2I runtime config exceeds {MAX_CONFIG_BYTES} bytes"
        )
    if stat.S_IMODE(st.st_mode) & 0o077:
        raise T2IRuntimeConfigError(
            "T2I runtime config permissions must not grant group/other access"
        )

    values: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise T2IRuntimeConfigError(
                f"invalid T2I runtime config line {lineno}: expected KEY=VALUE"
            )
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in _ALLOWED_KEY_SET:
            raise T2IRuntimeConfigError(
                f"unsupported T2I runtime config key at line {lineno}: {key!r}"
            )
        if key in values:
            raise T2IRuntimeConfigError(
                f"duplicate T2I runtime config key at line {lineno}: {key}"
            )
        if not _SAFE_VALUE_RE.fullmatch(value):
            raise T2IRuntimeConfigError(
                f"unsafe characters in T2I runtime config value at line {lineno}"
            )
        values[key] = value

    _validate_complete_values(values)
    return values


def _effective_values(file_values: Mapping[str, str]) -> dict[str, str]:
    defaults = {
        "NIGHTLY_RCA_T2I_ATLAS_ENABLED": "false",
        "NIGHTLY_RCA_T2I_SELECTIVE_CONTEXT_ENABLED": "false",
        "NIGHTLY_RCA_T2I_SELECTIVE_ALLOWED_EMAILS": "",
        "NIGHTLY_RCA_T2I_SELECTIVE_MAX_EVENTS": "5",
        "NIGHTLY_RCA_T2I_RUNS_ROOT": str(expected_runs_root()),
    }
    effective = dict(defaults)
    effective.update(file_values)
    for key in T2I_RUNTIME_KEYS:
        if key in os.environ:
            effective[key] = os.environ[key].strip()
    return effective


def bootstrap_t2i_runtime_env(
    config_file: str | os.PathLike[str] | None = None,
) -> T2IRuntimeBootstrapResult:
    """Load guarded local T2I config into missing process environment keys.

    Explicit process environment values always win, preserving the qualified
    D3N/D3O override behavior. The final effective configuration is validated
    fail-closed regardless of whether each value came from the process, file,
    or safe default.
    """
    raw_path = (
        os.fspath(config_file)
        if config_file is not None
        else os.getenv(CONFIG_PATH_ENV, "").strip()
    )
    path = Path(raw_path) if raw_path else default_config_path()

    file_exists = path.exists() or path.is_symlink()
    file_values = _read_config_file(path) if file_exists else {}

    applied: list[str] = []
    preserved: list[str] = []
    if file_values:
        for key in T2I_RUNTIME_KEYS:
            if key in os.environ:
                preserved.append(key)
                continue
            os.environ[key] = file_values[key]
            applied.append(key)

    effective = _effective_values(file_values)
    selective_enabled, allowlist_count, max_events = _validate_complete_values(effective)

    return T2IRuntimeBootstrapResult(
        config_path=str(path),
        file_loaded=bool(file_values),
        applied_keys=tuple(applied),
        preserved_process_keys=tuple(preserved),
        selective_enabled=selective_enabled,
        allowlist_count=allowlist_count,
        max_events=max_events,
    )
