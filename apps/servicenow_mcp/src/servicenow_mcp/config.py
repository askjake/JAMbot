from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceNowSettings(BaseSettings):
    """
    ServiceNow MCP server settings.
    All values can be overridden via environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ServiceNow instance
    snow_instance_url: str = "https://dish.service-now.com"

    # Auth mode: "oauth" or "basic"
    snow_auth_mode: str = "oauth"

    # OAuth 2.0 Client Credentials
    snow_client_id: str = ""
    snow_client_secret: str = ""

    # Basic Auth
    snow_username: str = ""
    snow_password: str = ""

    # Server bind
    snow_mcp_host: str = "127.0.0.1"
    snow_mcp_port: int = 8095


@lru_cache(maxsize=1)
def get_settings() -> ServiceNowSettings:
    return ServiceNowSettings()
