"""Agent package bootstrap."""

from __future__ import annotations

import sys

from app.agent.runtime_config import (
    T2IRuntimeBootstrapResult,
    T2IRuntimeConfigError,
    bootstrap_t2i_runtime_env,
)


def _backend_entrypoint_requested(argv: list[str] | None = None) -> bool:
    """True only for the qualified uvicorn backend entrypoint."""
    args = sys.argv if argv is None else argv
    return "app.main:app" in args


# Avoid changing generic CLI/pytest process environments merely because they
# import app.agent. The canonical production backend command is:
#   python -m uvicorn app.main:app --host ... --port 8002
# and app.main imports app.agent modules before materializing its Settings.
T2I_RUNTIME_BOOTSTRAP: T2IRuntimeBootstrapResult | None = None
if _backend_entrypoint_requested():
    T2I_RUNTIME_BOOTSTRAP = bootstrap_t2i_runtime_env()


__all__ = [
    "T2IRuntimeBootstrapResult",
    "T2IRuntimeConfigError",
    "T2I_RUNTIME_BOOTSTRAP",
    "bootstrap_t2i_runtime_env",
]
