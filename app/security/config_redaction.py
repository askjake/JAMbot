"""Build an allowlisted, credential-free settings summary for runtime logs.

The application settings model contains database URLs, API tokens, AWS
credentials, MCP headers, and authentication objects.  Serializing the full
model is therefore unsafe even at DEBUG level.  This module intentionally
avoids generic redaction-after-serialization: only explicitly approved fields
are copied into the log summary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_SAFE_SCALAR_FIELDS: tuple[str, ...] = (
    "NAME",
    "VERSION",
    "API_PREFIX",
    "DEBUG",
    "LOCAL",
    "AUTH_DISABLED",
    "FASTAPI_HOST",
    "FASTAPI_PORT",
    "PLLM_PROVIDER",
    "PLLM_MODEL",
    "PLLM_TOOL_MODEL",
    "PLLM_CTX_LEN",
    "ELLM_PROVIDER",
    "ELLM_MODEL",
    "ELLM_TOOL_MODEL",
    "ELLM_CTX_LEN",
    "EMBED_PROVIDER",
    "EMBED_MODEL",
    "AUTO_MODEL_ROUTING_ENABLED",
    "MCOP_ENABLED",
    "MCOP_MAX_CHILDREN",
    "MCOP_PARALLEL_LIMIT",
    "ENABLE_LOG_ASSIST_MCP",
    "ENABLE_INTERNAL_TOOLS_MCP",
)

# Configuration names whose values must never be serialized into log output.
# This is informational metadata only; the returned summary reports whether a
# field is configured, not its value.
_SECRET_FIELD_NAMES: tuple[str, ...] = (
    "MASTER_KEY",
    "POSTGRES_PWD",
    "POSTGRES_URL",
    "POSTGRES_SQLALCHEMY_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "COVERITY_ASSIST_TOKEN",
    "SENTRY_TOKEN",
    "SENTRY_API_KEY",
    "SENTRY_AUTH_TOKEN",
    "CONFLUENCE_API_TOKEN",
    "GITLAB_TOKEN",
    "JIRA_API_TOKEN",
    "GRASSHOPPER_AUTH_KEY",
    "GRASSHOPPER_OAUTH_CLIENT_SECRET",
)


def _configured(value: Any) -> bool:
    """Return whether a setting is materially configured without exposing it."""

    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return bool(value)
    return True


def _mcp_toolset_names(settings: Any) -> list[str]:
    """Return only configured MCP family names, never URLs, headers, or auth."""

    names: set[str] = set()
    for field_name in dir(settings):
        if not field_name.endswith("_MCP_CONFIG"):
            continue
        try:
            value = getattr(settings, field_name)
        except Exception:
            continue
        if not isinstance(value, Mapping):
            continue
        names.update(str(name) for name in value.keys() if str(name).strip())
    return sorted(names)


def build_settings_log_summary(settings: Any) -> dict[str, Any]:
    """Return a deterministic allowlisted summary safe for application logs.

    The function never calls ``model_dump`` and never walks arbitrary nested
    configuration.  Secret-bearing fields are represented only by boolean
    configured-state flags.
    """

    runtime: dict[str, Any] = {}
    for field_name in _SAFE_SCALAR_FIELDS:
        if not hasattr(settings, field_name):
            continue
        value = getattr(settings, field_name)
        if isinstance(value, (str, int, float, bool)) or value is None:
            runtime[field_name] = value
        else:
            runtime[field_name] = str(value)

    configured_secrets = {
        field_name: _configured(getattr(settings, field_name, None))
        for field_name in _SECRET_FIELD_NAMES
    }

    return {
        "schema": "settings_log_summary.v1",
        "runtime": runtime,
        "mcp_toolsets": _mcp_toolset_names(settings),
        "configured_secrets": configured_secrets,
        "redaction_policy": "allowlist_only",
    }
