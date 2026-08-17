"""Agent package bootstrap."""

from app.agent.runtime_config import (
    T2IRuntimeBootstrapResult,
    T2IRuntimeConfigError,
    bootstrap_t2i_runtime_env,
)

# The canonical backend imports app.agent modules before app.main materializes
# its Settings object. Load only the dedicated, allowlisted T2I runtime keys
# here so guarded selective-context enablement survives a normal backend restart.
T2I_RUNTIME_BOOTSTRAP: T2IRuntimeBootstrapResult = bootstrap_t2i_runtime_env()

__all__ = [
    "T2IRuntimeBootstrapResult",
    "T2IRuntimeConfigError",
    "T2I_RUNTIME_BOOTSTRAP",
    "bootstrap_t2i_runtime_env",
]
