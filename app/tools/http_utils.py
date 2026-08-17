"""Shared httpx client factories for MCP connections."""
from typing import Any

import httpx


def make_noverify_http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """httpx client factory that disables TLS certificate verification.

    Used for MCP servers that present self-signed certificates (e.g. the
    qodo-ssh-proxy ClusterIP reached via kubectl port-forward).  All other
    parameters mirror create_mcp_http_client from mcp.shared._httpx_utils.
    """
    kwargs: dict[str, Any] = {
        "follow_redirects": True,
        "verify": False,  # accept self-signed cert from qodo-ssh-proxy
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if headers is not None:
        kwargs["headers"] = headers
    if auth is not None:
        kwargs["auth"] = auth
    return httpx.AsyncClient(**kwargs)
