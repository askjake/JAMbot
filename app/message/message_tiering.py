"""
Turn-based tiered conversation memory.

This runs before LangChain's last-N token trimming.  It keeps the current turns
at full fidelity, compresses recent assistant/tool-heavy turns, and replaces
older turns with one compact HumanMessage summary.  The implementation is fully
rule-based and preserves AIMessage.tool_calls / ToolMessage.tool_call_id pairs by
never splitting a turn between tiers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.message.ai_message_compressor import compress_ai_message
from app.message.compression import count_message_tokens
from app.message.conversation_summarizer import summarize_turns
from app.tools.progressive_tool_memory import generate_progressive_levels

logger = logging.getLogger(__name__)

DEFAULT_CURRENT_TURNS = 3
DEFAULT_RECENT_TURNS = 7
DEFAULT_SUMMARY_TOKEN_BUDGET = 3000


@dataclass
class TieringStats:
    before_messages: int
    after_messages: int
    before_tokens: int
    after_tokens: int
    tier1_messages: int
    tier1_tokens: int
    tier1_turns: int
    tier2_messages: int
    tier2_tokens: int
    tier2_original_tokens: int
    tier2_turns: int
    tier3_messages: int
    tier3_tokens: int
    tier3_turns: int


@dataclass
class MessageTurn:
    index: int
    messages: list[BaseMessage]

    def token_count(self) -> int:
        return count_message_tokens(self.messages)



def apply_tiered_compression(
    messages: list[BaseMessage],
    target_tokens: int = 100_000,
    current_turns: int = DEFAULT_CURRENT_TURNS,
    recent_turns: int = DEFAULT_RECENT_TURNS,
    summary_token_budget: int = DEFAULT_SUMMARY_TOKEN_BUDGET,
) -> list[BaseMessage]:
    """
    Apply 3-zone memory compression to a message list.

    Tier 1: last ``current_turns`` user turns, unchanged.
    Tier 2: previous ``recent_turns`` user turns, AI/tool messages compressed.
    Tier 3: older turns summarized into one HumanMessage.
    """
    if not messages:
        return messages

    before_tokens = count_message_tokens(messages)
    turns, leading_messages, existing_summary = split_message_turns(messages)
    if len(turns) <= current_turns:
        return messages

    tier1 = turns[-current_turns:]
    recent_start = max(0, len(turns) - current_turns - recent_turns)
    tier2 = turns[recent_start : len(turns) - current_turns]
    tier3 = turns[:recent_start]

    current_turn_number = turns[-1].index if turns else 0
    compressed_tier2 = [
        MessageTurn(
            index=turn.index,
            messages=compress_recent_turn(turn.messages, current_turn=current_turn_number, turn_created=turn.index),
        )
        for turn in tier2
    ]

    result: list[BaseMessage] = []
    # Drop pre-first-human fragments.  They are usually orphaned AI/tool messages
    # introduced by count/window trimming, and keeping them before the synthetic
    # summary would violate Bedrock's required user-first conversation shape.
    if leading_messages:
        logger.debug("Tiered compression dropped %s pre-first-human message fragments", len(leading_messages))

    summary_message = _build_summary_message(
        tier3,
        start_index=tier3[0].index if tier3 else 1,
        existing_summary=existing_summary,
        summary_token_budget=summary_token_budget,
    )
    if summary_message is not None:
        result.append(summary_message)

    for turn in compressed_tier2:
        result.extend(turn.messages)
    for turn in tier1:
        result.extend(turn.messages)

    result = _fit_to_target_by_dropping_old_recent_turns(
        result=result,
        compressed_recent_turns=compressed_tier2,
        tier1=tier1,
        summary_message=summary_message,
        target_tokens=target_tokens,
    )

    after_tokens = count_message_tokens(result)
    stats = TieringStats(
        before_messages=len(messages),
        after_messages=len(result),
        before_tokens=before_tokens,
        after_tokens=after_tokens,
        tier1_messages=sum(len(turn.messages) for turn in tier1),
        tier1_tokens=count_message_tokens(_flatten(turn.messages for turn in tier1)),
        tier1_turns=len(tier1),
        tier2_messages=sum(len(turn.messages) for turn in compressed_tier2),
        tier2_tokens=count_message_tokens(_flatten(turn.messages for turn in compressed_tier2)),
        tier2_original_tokens=count_message_tokens(_flatten(turn.messages for turn in tier2)),
        tier2_turns=len(compressed_tier2),
        tier3_messages=1 if summary_message is not None else 0,
        tier3_tokens=count_message_tokens([summary_message]) if summary_message is not None else 0,
        tier3_turns=len(tier3),
    )
    _log_tiering_stats(stats)
    return result


def split_message_turns(messages: list[BaseMessage]) -> tuple[list[MessageTurn], list[BaseMessage], str | None]:
    """
    Split messages into user turns.

    Returns ``(turns, leading_messages, existing_summary)``.  An existing
    synthetic summary is detected and merged into the new summary instead of
    being treated as a normal user turn.
    """
    turns: list[MessageTurn] = []
    leading: list[BaseMessage] = []
    current: list[BaseMessage] = []
    existing_summary: str | None = None
    turn_index = 0

    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = msg.content if isinstance(msg.content, str) else ""
            if text.startswith("[CONVERSATION CONTEXT - Turns"):
                existing_summary = text
                continue

            if current:
                turns.append(MessageTurn(index=turn_index, messages=current))
            turn_index += 1
            current = [msg]
        else:
            if turn_index == 0:
                leading.append(msg)
            else:
                current.append(msg)

    if current:
        turns.append(MessageTurn(index=turn_index, messages=current))

    return turns, leading, existing_summary


def compress_recent_turn(messages: list[BaseMessage], current_turn: int, turn_created: int) -> list[BaseMessage]:
    """Compress AI and aged tool result messages inside one recent turn."""
    tool_call_names: dict[str, str] = {}
    compressed: list[BaseMessage] = []

    for msg in messages:
        if isinstance(msg, AIMessage):
            for tool_call in getattr(msg, "tool_calls", None) or []:
                if not isinstance(tool_call, dict):
                    continue
                tool_call_id = tool_call.get("id")
                tool_name = tool_call.get("name")
                if tool_call_id and tool_name:
                    tool_call_names[str(tool_call_id)] = str(tool_name)
            compressed.append(compress_ai_message(msg, target_tokens=400))
        elif isinstance(msg, ToolMessage):
            tool_call_id = getattr(msg, "tool_call_id", None) or ""
            tool_name = getattr(msg, "name", None) or tool_call_names.get(tool_call_id, "")
            if isinstance(msg.content, str) and tool_name:
                progressive = generate_progressive_levels(
                    tool_name=tool_name,
                    raw_result=msg.content,
                    tool_call_id=tool_call_id,
                    turn_created=turn_created,
                )
                content = progressive.get_content_for_turn(current_turn)
                compressed.append(_copy_tool_message_with_content(msg, content))
            else:
                compressed.append(msg)
        else:
            compressed.append(msg)

    return compressed


def _build_summary_message(
    tier3: list[MessageTurn],
    start_index: int,
    existing_summary: str | None,
    summary_token_budget: int,
) -> HumanMessage | None:
    if not tier3 and not existing_summary:
        return None

    summary = summarize_turns(
        [turn.messages for turn in tier3],
        start_index=start_index,
        max_tokens=summary_token_budget,
        existing_summary=existing_summary,
    )
    if not summary.strip():
        return None
    return HumanMessage(content=summary)


def _fit_to_target_by_dropping_old_recent_turns(
    result: list[BaseMessage],
    compressed_recent_turns: list[MessageTurn],
    tier1: list[MessageTurn],
    summary_message: HumanMessage | None,
    target_tokens: int,
) -> list[BaseMessage]:
    """Drop oldest Tier-2 turns if compression still exceeds target."""
    if count_message_tokens(result) <= target_tokens:
        return result

    tier1_messages = _flatten(turn.messages for turn in tier1)
    summary_messages = [summary_message] if summary_message is not None else []

    retained_recent = list(compressed_recent_turns)
    while retained_recent:
        candidate = summary_messages + _flatten(turn.messages for turn in retained_recent) + tier1_messages
        if count_message_tokens(candidate) <= target_tokens:
            return candidate
        retained_recent.pop(0)

    candidate = summary_messages + tier1_messages
    if count_message_tokens(candidate) <= target_tokens:
        return candidate

    # Last resort: keep the summary and current turns.  The downstream large
    # message and LangChain trim guards remain the final safety net for a huge
    # current turn.
    logger.warning(
        "Tiered compression result still exceeds target after dropping Tier 2: %s > %s",
        count_message_tokens(candidate),
        target_tokens,
    )
    return candidate


def _copy_tool_message_with_content(message: ToolMessage, content: str) -> ToolMessage:
    try:
        return message.model_copy(update={"content": content})
    except Exception:
        pass
    try:
        return message.copy(update={"content": content})
    except Exception:
        pass

    kwargs = {"content": content, "tool_call_id": getattr(message, "tool_call_id", None)}
    for attr in ("name", "id", "additional_kwargs", "response_metadata", "status", "artifact"):
        value = getattr(message, attr, None)
        if value not in (None, [], {}):
            kwargs[attr] = value
    try:
        return ToolMessage(**kwargs)
    except TypeError:
        kwargs.pop("artifact", None)
        kwargs.pop("status", None)
        return ToolMessage(**kwargs)


def _flatten(groups: Iterable[Iterable[BaseMessage]]) -> list[BaseMessage]:
    return [msg for group in groups for msg in group]


def _log_tiering_stats(stats: TieringStats) -> None:
    logger.info(
        "Tiered compression: %s messages (%s tokens) → %s messages (%s tokens)",
        stats.before_messages,
        stats.before_tokens,
        stats.after_messages,
        stats.after_tokens,
    )
    logger.info(
        "  Tier 1 (current, %s turns): %s messages, %s tokens [kept full]",
        stats.tier1_turns,
        stats.tier1_messages,
        stats.tier1_tokens,
    )
    logger.info(
        "  Tier 2 (recent, %s turns): %s messages, %s tokens [compressed from %s]",
        stats.tier2_turns,
        stats.tier2_messages,
        stats.tier2_tokens,
        stats.tier2_original_tokens,
    )
    logger.info(
        "  Tier 3 (summary): %s message, %s tokens [summarizing %s evicted turns]",
        stats.tier3_messages,
        stats.tier3_tokens,
        stats.tier3_turns,
    )
