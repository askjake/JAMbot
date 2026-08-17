"""Typed operational identity and endpoint attribution.

Self-location is established only by an explicit user-pinned target together
with a successful runtime identity observation on that target.  Remote service,
HTTP, browser, gateway, source-address, MCP, and repository-origin endpoints are
attribution fields only; none may establish or override self-location.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping
from urllib.parse import urlsplit

_HOST_FIELDS = (
    "agent_execution_host",
    "tool_executor_host",
    "remote_service_host",
    "http_target",
    "source_client_address",
    "mcp_server_host",
    "repository_checkout_host",
    "repository_path",
    "repository_origin_host",
    "self_location_basis",
)
_ENDPOINT_FIELDS = {
    "tool_executor_host",
    "remote_service_host",
    "http_target",
    "source_client_address",
    "mcp_server_host",
    "repository_origin_host",
}
_SAFE_TEXT_RE = re.compile(r"^[A-Za-z0-9_.:/@+-]{0,256}$")


def _host_only(value: Any) -> str:
    text = str(value or "").strip().replace("\n", " ").replace("\r", " ")
    if not text:
        return ""
    candidate = text
    if "://" not in candidate and ("/" in candidate or "?" in candidate or "#" in candidate):
        candidate = "//" + candidate
    try:
        parsed = urlsplit(candidate)
        if parsed.hostname:
            host = parsed.hostname
            if parsed.port is not None:
                host += f":{parsed.port}"
            return host[:256]
    except (ValueError, TypeError):
        pass
    # SCP-style Git origin: user@host:path
    if "@" in text:
        text = text.rsplit("@", 1)[1]
    text = text.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
    if ":" in text and text.count(":") == 1:
        host, suffix = text.split(":", 1)
        if suffix.isdigit():
            return f"{host}:{suffix}"[:256]
        return host[:256]
    return text[:256]


def _safe_path(value: Any) -> str:
    text = str(value or "").strip().replace("\n", " ").replace("\r", " ")
    text = text.split("?", 1)[0].split("#", 1)[0]
    if not text.startswith("/") or "\x00" in text:
        return ""
    return text[:256]


def typed_host_attribution(values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    source = values or {}
    out: dict[str, Any] = {}
    for field_name in _HOST_FIELDS:
        value = source.get(field_name)
        if field_name == "repository_path":
            out[field_name] = _safe_path(value)
        elif field_name == "self_location_basis":
            text = str(value or "")[:128]
            out[field_name] = text if _SAFE_TEXT_RE.fullmatch(text) else ""
        else:
            out[field_name] = _host_only(value)
    # Endpoint observations are attribution only by contract.
    out["endpoint_can_establish_self_location"] = False
    return out


@dataclass(frozen=True)
class OperationalIdentity:
    user_pinned_target: str = ""
    runtime_hostname: str = ""
    repository_checkout_host: str = ""
    repository_path: str = ""
    repository_origin_host: str = ""
    repository_branch: str = ""
    repository_head: str = ""
    endpoint_observations: Mapping[str, str] = field(default_factory=dict)
    endpoint_can_establish_self_location: bool = False

    @property
    def authoritative_target(self) -> str:
        return _host_only(self.user_pinned_target)

    @property
    def self_location_established(self) -> bool:
        return bool(self.authoritative_target and _host_only(self.runtime_hostname))

    @property
    def self_location_basis(self) -> str:
        if self.self_location_established:
            return "USER_PINNED_TARGET_PLUS_RUNTIME_HOSTNAME"
        if self.authoritative_target:
            return "PINNED_TARGET_RUNTIME_IDENTITY_PENDING"
        return "SELF_LOCATION_UNPROVEN"

    @property
    def repository_identity_established(self) -> bool:
        return bool(
            self.self_location_established
            and _host_only(self.repository_checkout_host)
            and _safe_path(self.repository_path)
            and _host_only(self.repository_origin_host)
            and str(self.repository_branch or "")
            and re.fullmatch(r"[0-9a-fA-F]{7,64}", str(self.repository_head or ""))
        )

    def observe_endpoint(self, endpoint_type: str, value: Any) -> "OperationalIdentity":
        endpoint_type = str(endpoint_type or "")
        if endpoint_type not in _ENDPOINT_FIELDS:
            raise ValueError(f"unsupported endpoint type: {endpoint_type}")
        observations = dict(self.endpoint_observations)
        observations[endpoint_type] = _host_only(value)
        return replace(self, endpoint_observations=observations)

    def to_host_attribution(self) -> dict[str, Any]:
        return typed_host_attribution(
            {
                "agent_execution_host": self.runtime_hostname,
                "repository_checkout_host": self.repository_checkout_host,
                "repository_path": self.repository_path,
                "repository_origin_host": self.repository_origin_host,
                "self_location_basis": self.self_location_basis,
                **dict(self.endpoint_observations),
            }
        )
