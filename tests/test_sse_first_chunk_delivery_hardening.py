from __future__ import annotations

import ast
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Awaitable, Callable, Coroutine, Optional
import uuid


class FakeChunk:
    def __init__(self, content, response_metadata=None):
        self.content = content
        self.response_metadata = response_metadata or {}


def _load_transformer_namespace():
    source_path = Path(__file__).parents[1] / "app" / "message" / "utils.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {"_format_sse_event", "_format_content_delta", "sse_transformer_for_langgraph_astream"}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {
        "Any": Any,
        "AsyncGenerator": AsyncGenerator,
        "Awaitable": Awaitable,
        "Callable": Callable,
        "Coroutine": Coroutine,
        "Optional": Optional,
        "AIMessageChunk": FakeChunk,
        "CONTENT_TYPE_MAPPING": {"text": "text"},
        "json": json,
        "logging": logging,
        "logger": logging.getLogger("sse-test"),
        "uuid": uuid,
    }
    exec(compile(module, str(source_path), "exec"), ns)
    return ns


def test_single_direct_ai_message_chunk_is_not_discarded_by_sse_transformer():
    ns = _load_transformer_namespace()
    transformer = ns["sse_transformer_for_langgraph_astream"]

    async def graph_stream():
        yield (
            FakeChunk(
                content=[{"type": "text", "text": "BLOCKED_MISSING_CAPABILITY", "index": 0}],
                response_metadata={},
            ),
            {},
        )

    async def stream_coro():
        return graph_stream()

    async def error_callback(_):
        return "stream-error"

    async def save_message_cb(**_):
        return None

    async def collect():
        return [
            event
            async for event in transformer(
                stream_coro(),
                error_callback=error_callback,
                save_message_cb=save_message_cb,
                save_message_kwargs={},
            )
        ]

    events = asyncio.run(collect())
    joined = "".join(events)
    assert "content_block_delta" in joined
    assert "BLOCKED_MISSING_CAPABILITY" in joined
