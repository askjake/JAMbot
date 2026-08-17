"""MCP transport with lazy project imports so local unit tests remain isolated."""
from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any


class ToolTransportError(RuntimeError):
    pass


class McpToolClient:
    def __init__(self, server_urls: dict[str, str], aws_region: str = "us-west-2"):
        self.server_urls = server_urls
        self.aws_region = aws_region
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, Any] = {}

    async def __aenter__(self) -> "McpToolClient":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(exc_type, exc, tb)
        self._stack = None
        self._sessions.clear()

    async def _session(self, server: str):
        if server in self._sessions:
            return self._sessions[server]
        url = self.server_urls.get(server, "")
        if not url:
            raise ToolTransportError(f"MCP URL not configured for {server!r} — tool call will be skipped")
        if self._stack is None:
            raise ToolTransportError("McpToolClient must be used as an async context manager")

        try:
            import httpx
            from app.sigv4_auth import AWSSigV4Auth
            from app.tools.http_utils import make_noverify_http_client
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:
            raise ToolTransportError(f"project MCP dependencies unavailable: {exc}") from exc

        http_client = make_noverify_http_client(
            timeout=httpx.Timeout(120, read=600),
            auth=AWSSigV4Auth(service="lambda", region=self.aws_region),
        )
        await self._stack.enter_async_context(http_client)
        streams = await self._stack.enter_async_context(
            streamable_http_client(url, http_client=http_client)
        )
        read_stream, write_stream, _ = streams
        session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        self._sessions[server] = session
        return session

    async def call(self, server: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        session = await self._session(server)
        result = await session.call_tool(tool_name, arguments=arguments)
        if getattr(result, "isError", False):
            text = "\n".join(getattr(block, "text", "") for block in result.content)
            raise ToolTransportError(text or f"tool returned isError: {tool_name}")
        structured = getattr(result, "structuredContent", None)
        if structured is None:
            structured = getattr(result, "structured_content", None)
        if structured is not None:
            return structured

        text_parts = [
            getattr(block, "text", "") for block in result.content if hasattr(block, "text")
        ]
        text = "\n".join(text_parts)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            parsed_parts: list[Any] = []
            for part in text_parts:
                try:
                    parsed_parts.append(json.loads(part))
                except (json.JSONDecodeError, ValueError):
                    parsed_parts = []
                    break
            if parsed_parts:
                return parsed_parts[0] if len(parsed_parts) == 1 else parsed_parts
            return text
