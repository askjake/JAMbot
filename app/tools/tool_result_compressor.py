"""
Web-browse tool result compression compatibility layer.

``progressive_tool_memory`` historically imports ``compress_web_browse_result``
from this module.  Keep the public API small and delegate the implementation to
the content-aware ToolMessage compressor.
"""

from __future__ import annotations

from app.message.tool_message_compressor import compress_web_tool_output

WEB_BROWSE_TOOLS: set[str] = {
    "web_browse",
    "web_browse_interact",
    "web_browse_api",
    "local_web_browse_manual_login",
    "local_web_browse_clear_session",
    "headless_browser_mcp",
}


def compress_web_browse_result(tool_name: str, raw_result: str, token_budget: int = 3000) -> str:
    """
    Compress browser output into readable text within an approximate token budget.

    Args:
        tool_name: Name of the web/browser tool.
        raw_result: Raw browser output.
        token_budget: Approximate output token budget.  The implementation uses
            a conservative 4 characters/token conversion because this hot-path
            helper should not depend on tokenizer availability.
    """

    max_chars = max(1_500, int(token_budget or 3000) * 4)
    return compress_web_tool_output(
        content=raw_result,
        tool_name=tool_name,
        max_chars=max_chars,
    )
