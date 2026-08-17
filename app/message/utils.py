from collections import defaultdict
from collections.abc import AsyncIterator
import json
from typing import AsyncGenerator, Optional, Callable, Any, Awaitable, Coroutine
import logging
import uuid

from langchain_core.messages import AIMessageChunk

from app.core.constants import CHAT_NS_MAP
from .models import MessageMD
from .constants import CONTENT_TYPE_MAPPING

logger = logging.getLogger(__name__)


def get_chat_agent_type(namespace: str) -> str:
    return CHAT_NS_MAP.get(namespace, CHAT_NS_MAP["generic"]).chat_agent


def get_chat_agent_param(namespace: str) -> str | None:
    return CHAT_NS_MAP.get(namespace, CHAT_NS_MAP["generic"]).agent_params


def get_last_checkpoint_of_branch(
    all_messages_for_chat: list[MessageMD], start_checkpoint_id: str
) -> Optional[str]:
    """
    Finds the checkpoint_id of the last message in the "latest" branch
    that originates from the specified start_checkpoint_id.
    "Latest" is determined by the highest checkpoint_id among the branch's leaf nodes.

    Args:
        all_messages_for_chat: All Message model instances for the current chat_id.
        start_checkpoint_id: The checkpoint_id of the message to branch from.

    Returns:
        The checkpoint_id string of the last message in the identified latest branch,
        or None if no such branch or message is found.
    """
    if not all_messages_for_chat:
        return None

    start_node: Optional[MessageMD] = None
    message_map_by_cpid_local = {
        m.checkpoint_id: m for m in all_messages_for_chat
    }  # For quick lookup

    start_node = message_map_by_cpid_local.get(start_checkpoint_id)

    if not start_node:
        return None  # Start message (by checkpoint_id) not found

    children_map = defaultdict(list)
    for msg in all_messages_for_chat:
        if msg.parent_checkpoint_id:  # Check if parent_checkpoint_id is not None
            children_map[msg.parent_checkpoint_id].append(msg)

    dfs_stack: list[MessageMD] = [start_node]
    visited_in_dfs = set()
    latest_leaf_id = None

    while dfs_stack:
        current_node = dfs_stack.pop()

        if current_node.checkpoint_id in visited_in_dfs:
            continue
        visited_in_dfs.add(current_node.checkpoint_id)

        children = children_map.get(current_node.checkpoint_id, [])
        if children:
            dfs_stack.extend(children)
        else:
            if latest_leaf_id is None or current_node.checkpoint_id > latest_leaf_id:
                latest_leaf_id = current_node.checkpoint_id

    return latest_leaf_id


def _format_sse_event(event_name: str, data: dict) -> str:
    """Helper to format a string for Server-Sent Events."""
    json_data = json.dumps(data)
    return f"event: {event_name}\ndata: {json_data}\n\n"


def _format_content_delta(type: str, data: Any) -> dict:
    """Helper function to format LangChain streaming
    response content data into content delta block
    """
    mapped_type = CONTENT_TYPE_MAPPING.get(type, type)

    # Define special handling of LangChain resp structure
    if type == "reasoning_content":
        data = data.get("text")

    return {"type": f"{mapped_type}_delta", mapped_type: data}


async def sse_transformer_for_langgraph_astream(
    resp_stream_coro: Coroutine[
        None, None, AsyncIterator[tuple[AIMessageChunk, dict[str, Any]], None]
    ],
    error_callback: Callable[..., Awaitable[None]],
    save_message_cb: Callable[..., Awaitable[None]],
    save_message_kwargs: dict[str, Any],
    input_message_id: uuid.UUID | None = None,
    input_version_index: int = 0,
    on_stream_end_callbacks: list[Callable[..., Awaitable[None]]] = [],
    callback_kwargs: list[Optional[dict[str, Any]]] = [],
) -> AsyncGenerator[str, None]:
    """Transform LangGraph message streaming into the Chat SSE contract.

    Every LangGraph message tuple is processed.  In particular, do not consume
    the first tuple as metadata-only: graph nodes may return a single synthetic
    AIMessage (for example a bounded terminal/no-progress result), and dropping
    that tuple produces an empty assistant bubble in the GUI.
    """

    in_content_block: bool = False
    content_index: int = 0
    final_stop_reason: Optional[str] = None
    err_msg: Optional[str] = None
    model_name: str = "unknown_model"
    resolved_callback_kwargs = [kargs or {} for kargs in callback_kwargs]
    message_ended_gracefully: bool = False

    user_logical_message_id = input_message_id or uuid.uuid4()
    ai_logical_message_id = uuid.uuid4()

    start_message_data = {
        "type": "message_start",
        "message": {
            "input_message_id": str(user_logical_message_id),
            "input_version_index": input_version_index,
            "response_message_id": str(ai_logical_message_id),
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": None,
            "stop_reason": None,
        },
    }
    yield _format_sse_event("message_start", start_message_data)

    async def emit_text(text: str):
        nonlocal in_content_block
        if not text:
            return
        if not in_content_block:
            cb_start_data = {
                "type": "content_block_start",
                "index": content_index,
                "content_block": {"type": "text", "text": ""},
            }
            yield _format_sse_event("content_block_start", cb_start_data)
            in_content_block = True
        delta_event_data = {
            "type": "content_block_delta",
            "index": content_index,
            "delta": _format_content_delta("text", text),
        }
        yield _format_sse_event("content_block_delta", delta_event_data)

    astream_output = await resp_stream_coro
    try:
        async for chunk, metadata in astream_output:
            if metadata.get("ls_model_name"):
                model_name = metadata["ls_model_name"]

            response_metadata = getattr(chunk, "response_metadata", None) or {}
            stop_reason = response_metadata.get("stopReason")

            content = getattr(chunk, "content", None)
            if isinstance(content, str):
                async for event in emit_text(content):
                    yield event
            elif content:
                # Normal provider streaming uses a list of content-block deltas.
                for content_item in content:
                    if not isinstance(content_item, dict):
                        continue
                    if content_item.get("index") is None:
                        continue

                    item_type = content_item.get("type")
                    if item_type:
                        mapped_type = CONTENT_TYPE_MAPPING.get(item_type)
                        if mapped_type is None:
                            continue

                        if not in_content_block:
                            cb_start_data = {
                                "type": "content_block_start",
                                "index": content_index,
                                "content_block": {
                                    "type": mapped_type,
                                    mapped_type: "",
                                },
                            }
                            yield _format_sse_event("content_block_start", cb_start_data)
                            in_content_block = True

                        item_data = content_item.get(item_type)
                        if item_data:
                            delta_event_data = {
                                "type": "content_block_delta",
                                "index": content_index,
                                "delta": _format_content_delta(item_type, item_data),
                            }
                            yield _format_sse_event("content_block_delta", delta_event_data)

                    elif in_content_block:
                        yield _format_sse_event(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": content_index},
                        )
                        content_index += 1
                        in_content_block = False

            if stop_reason and stop_reason != "tool_use":
                final_stop_reason = stop_reason
                break

        # If a synthetic/direct AIMessage used plain string content, no provider
        # block-stop marker exists. Close the block deterministically here.
        if in_content_block:
            yield _format_sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": content_index},
            )
            content_index += 1
            in_content_block = False

        # Drain after an explicit stop reason so callbacks/checkpoint streaming
        # can finish without creating duplicate content events.
        async for _chunk, _metadata in astream_output:
            pass

        message_delta_payload = {
            "stop_reason": final_stop_reason or "end_turn",
            "model": model_name,
        }
        yield _format_sse_event(
            "message_delta", {"type": "message_delta", "delta": message_delta_payload}
        )
        yield _format_sse_event("message_stop", {"type": "message_stop"})
        message_ended_gracefully = True

        await save_message_cb(
            human_mid=user_logical_message_id,
            ai_mid=ai_logical_message_id,
            **save_message_kwargs,
        )

        if on_stream_end_callbacks:
            for callback, kargs in zip(
                on_stream_end_callbacks, resolved_callback_kwargs
            ):
                await callback(**kargs)

    except Exception:
        logger.exception("Error during SSE streaming")
        err_msg = await error_callback(err_msg)

    finally:
        if not message_ended_gracefully:
            message_delta_payload = {
                "stop_reason": "error_or_unexpected_end",
                "error_msg": err_msg,
            }
            yield _format_sse_event(
                "message_delta",
                {"type": "message_delta", "delta": message_delta_payload},
            )
            yield _format_sse_event("message_stop", {"type": "message_stop"})
