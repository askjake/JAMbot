from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.security.config_redaction import build_settings_log_summary


SECRET_CANARIES = {
    "MASTER_KEY": "master-secret-canary",
    "POSTGRES_PWD": "postgres-secret-canary",
    "POSTGRES_URL": "postgresql://user:postgres-secret-canary@db/example",
    "POSTGRES_SQLALCHEMY_URL": "postgresql+asyncpg://user:postgres-secret-canary@db/example",
    "AWS_ACCESS_KEY_ID": "AKIASECRETTEST000001",
    "AWS_SECRET_ACCESS_KEY": "aws-secret-canary",
    "AWS_SESSION_TOKEN": "aws-session-secret-canary",
    "COVERITY_ASSIST_TOKEN": "coverity-secret-canary",
    "SENTRY_TOKEN": "sentry-secret-canary",
    "SENTRY_API_KEY": "sentry-api-secret-canary",
    "SENTRY_AUTH_TOKEN": "sentry-auth-secret-canary",
    "CONFLUENCE_API_TOKEN": "confluence-secret-canary",
    "GITLAB_TOKEN": "gitlab-secret-canary",
    "JIRA_API_TOKEN": "jira-secret-canary",
    "GRASSHOPPER_AUTH_KEY": "grasshopper-secret-canary",
    "GRASSHOPPER_OAUTH_CLIENT_SECRET": "grasshopper-oauth-secret-canary",
}


def _settings_fixture() -> SimpleNamespace:
    values = {
        "NAME": "Dish-Chat",
        "VERSION": "2.1.0",
        "API_PREFIX": "/rest/api/v1",
        "DEBUG": True,
        "LOCAL": True,
        "AUTH_DISABLED": False,
        "FASTAPI_HOST": "127.0.0.1",
        "FASTAPI_PORT": 8001,
        "PLLM_PROVIDER": "ollama",
        "PLLM_MODEL": "model-a",
        "PLLM_TOOL_MODEL": "model-b",
        "PLLM_CTX_LEN": 65536,
        "ELLM_PROVIDER": "ollama",
        "ELLM_MODEL": "model-c",
        "ELLM_TOOL_MODEL": "model-d",
        "ELLM_CTX_LEN": 32768,
        "EMBED_PROVIDER": "ollama",
        "EMBED_MODEL": "embed-a",
        "AUTO_MODEL_ROUTING_ENABLED": True,
        "MCOP_ENABLED": True,
        "MCOP_MAX_CHILDREN": 10,
        "MCOP_PARALLEL_LIMIT": 4,
        "ENABLE_LOG_ASSIST_MCP": False,
        "ENABLE_INTERNAL_TOOLS_MCP": True,
        "S3_STB_LOGS_MCP_CONFIG": {
            "s3_stb_logs": {
                "url": "https://endpoint.invalid/mcp",
                "headers": {"Authorization": "Bearer nested-secret-canary"},
            }
        },
    }
    values.update(SECRET_CANARIES)
    return SimpleNamespace(**values)


def test_settings_log_summary_is_allowlist_only_and_deterministic():
    summary = build_settings_log_summary(_settings_fixture())
    rendered = json.dumps(summary, sort_keys=True)

    assert summary["schema"] == "settings_log_summary.v1"
    assert summary["redaction_policy"] == "allowlist_only"
    assert summary["mcp_toolsets"] == ["s3_stb_logs"]
    assert summary["runtime"]["FASTAPI_PORT"] == 8001
    assert build_settings_log_summary(_settings_fixture()) == summary

    for value in SECRET_CANARIES.values():
        assert value not in rendered
    assert "nested-secret-canary" not in rendered
    assert "endpoint.invalid" not in rendered
    assert all(summary["configured_secrets"].values())


def test_main_does_not_serialize_complete_settings_model():
    source = (Path(__file__).parents[1] / "app" / "main.py").read_text()
    assert "build_settings_log_summary(settings)" in source
    assert "settings.model_dump()" not in source
    assert "Settings loaded" in source
