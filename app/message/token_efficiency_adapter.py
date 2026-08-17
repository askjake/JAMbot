"""
Bridge adapter between the Token Efficiency Layer and the LangChain message pipeline.

Applies codebook encoding and prompt compression to conversation history messages
AFTER tiered compression has already run, providing an additional ~20-40% token
reduction on the remaining message content.

Integration point: called from truncate_messages() in agentic_rag.py after
apply_tiered_compression() and before truncate_large_messages().

Guards:
- System messages are NEVER compressed or encoded
- The LAST HumanMessage (current user turn) is NEVER compressed or encoded
- AIMessage.tool_calls metadata is always preserved
- ToolMessage content is NEVER codebook-encoded or prompt-compressed
- ToolMessage.tool_call_id is always preserved
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

from app.message.compression import count_tokens, count_message_tokens

logger = logging.getLogger(__name__)

# Add the token_efficiency_project to path for imports
_TEL_PATH = Path(__file__).resolve().parent.parent.parent / "token_efficiency_project"
if str(_TEL_PATH) not in sys.path:
    sys.path.insert(0, str(_TEL_PATH))

try:
    from src.codebook_manager import CodebookManager
    from src.prompt_compressor import PromptCompressor
    _TEL_AVAILABLE = True
except ImportError:
    _TEL_AVAILABLE = False
    logger.warning("Token efficiency layer not available; skipping codebook/compression integration")


# Module-level singleton (reused across calls within a session)
_codebook: Optional["CodebookManager"] = None
_compressor: Optional["PromptCompressor"] = None


def _get_codebook() -> "CodebookManager":
    global _codebook
    if _codebook is None:
        _codebook = CodebookManager(
            max_entries=15,
            sigil_prefix="\u03a3",  # Sigma
            min_occurrences_to_promote=3,
            min_phrase_tokens=6,
            expiry_turns=12,
            match_mode="exact",
        )
    return _codebook


def _get_compressor() -> "PromptCompressor":
    global _compressor
    if _compressor is None:
        _compressor = PromptCompressor(
            target_ratio=0.70,
            protected_patterns=[
                r"R\d{7,}",            # Receiver IDs
                r"ATVDI-\d+",          # JIRA tickets
                r"s3://[^\s]+",        # S3 paths
                r"https?://[^\s]+",    # URLs
                r"/[\w/._-]{5,}",     # File paths
                r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",  # IPs
                r"tool_call_id:\s*\S+",  # tool call IDs
                r"exit code: \d+",     # Exit codes
            ],
            stop_phrases=[
                "Let me ",
                "I\'ll now ",
                "Here\'s what ",
                "Now let me ",
                "I need to ",
                "Let me check ",
                "I can see that ",
            ],
            method="heuristic",
        )
    return _compressor


def _get_message_text(msg: BaseMessage) -> str:
    """Extract text content from a message."""
    if isinstance(msg.content, str):
        return msg.content
    elif isinstance(msg.content, list):
        parts = []
        for item in msg.content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return " ".join(parts)
    return str(msg.content) if msg.content else ""


def _is_current_user_message(msg: BaseMessage, messages: list[BaseMessage]) -> bool:
    """Check if this is the last HumanMessage (current turn - never compress)."""
    if not isinstance(msg, HumanMessage):
        return False
    # Walk backwards to find the last HumanMessage
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m is msg
    return False


def apply_token_efficiency_layer(
    messages: list[BaseMessage],
    min_messages_to_activate: int = 15,
    min_tokens_to_activate: int = 40000,
) -> list[BaseMessage]:
    """
    Apply codebook encoding and prompt compression to message history.

    This is a supplementary compression layer that runs AFTER tiered compression
    but BEFORE the final trim_messages() safety net. It provides additional
    savings through:
    1. Codebook encoding: repeated phrases → compact sigils
    2. Prompt compression: remove low-importance tokens from old messages

    Args:
        messages: List of LangChain messages (post-tiered-compression)
        min_messages_to_activate: Skip if fewer messages than this
        min_tokens_to_activate: Skip if total tokens below this threshold

    Returns:
        Possibly compressed list of messages (same types, metadata preserved)
    """
    if not _TEL_AVAILABLE:
        return messages

    if len(messages) < min_messages_to_activate:
        return messages

    total_tokens = count_message_tokens(messages)
    if total_tokens < min_tokens_to_activate:
        return messages

    # D3B3: this layer must be a pure function of its input.  A module-global
    # codebook that accumulates promoted phrases across calls makes the
    # model-visible prefix churn between otherwise identical turns (defeating
    # provider prompt caching) and couples unrelated conversations through
    # shared learned state.  Rebuild the codebook from the current window only.
    reset_session()
    codebook = _get_codebook()

    # Phase 1: Observe conversational text for codebook candidates.
    # ToolMessage payloads can be structured JSON/log data; encoding those can
    # corrupt keys, paths, or machine-readable values, so they are excluded.
    for msg in messages:
        if isinstance(msg, (SystemMessage, ToolMessage)):
            continue
        text = _get_message_text(msg)
        if text and len(text) > 20:
            codebook.observe(text)

    codebook.promote_candidates()
    codebook.garbage_collect()

    # Phase 2: Encode and compress eligible messages
    result: list[BaseMessage] = []
    # (result index, original message) for every message whose content was
    # replaced with codebook-encoded text.
    encoded_positions: list[tuple[int, BaseMessage]] = []
    codebook_preamble = codebook.render_codebook_prompt()

    for msg in messages:
        # GUARD: Never touch system messages or tool payloads.  ToolMessages
        # may contain JSON reports, raw logs, paths, or schema-bearing content;
        # the large-message compressor handles them with tool-name routing.
        if isinstance(msg, (SystemMessage, ToolMessage)):
            result.append(msg)
            continue

        # GUARD: Never touch the current user message
        if _is_current_user_message(msg, messages):
            result.append(msg)
            continue

        text = _get_message_text(msg)

        # Skip very short messages (not worth compressing)
        if not text or len(text) < 100:
            result.append(msg)
            continue

        # Apply codebook encoding.  Substitution is reversible through
        # codebook.decode() as long as the legend travels with the context.
        encoded_text = codebook.encode(text)

        # D3B3: the prompt compressor removes "low importance" tokens, which is
        # irreversible and demonstrably strips negations and qualifiers - for
        # example "no cross-child tool state was shared" collapsing to
        # "cross-child shared", and "the gap is ingestion lag or genuine
        # absence" collapsing to "gap ingestion genuine absence".  Inverting or
        # destroying a factual, evidence, or safety claim is never an acceptable
        # trade for a marginal token saving, so lossy token dropping is not
        # applied to narrative content.
        compressed_text = encoded_text

        # Only replace if we actually saved tokens
        original_tokens = count_tokens(text)
        new_tokens = count_tokens(compressed_text)

        if new_tokens >= original_tokens:
            result.append(msg)
            continue

        # Reconstruct message with compressed content, preserving all metadata
        if isinstance(msg, HumanMessage):
            new_msg = HumanMessage(content=compressed_text)
            new_msg.id = msg.id
        elif isinstance(msg, AIMessage):
            new_msg = AIMessage(
                content=compressed_text,
                tool_calls=getattr(msg, "tool_calls", None) or [],
            )
            new_msg.id = msg.id
        else:
            result.append(msg)
            continue

        encoded_positions.append((len(result), msg))
        result.append(new_msg)

    # Phase 3: Inject codebook preamble if there are active sigils
    sigil_prefix = str(getattr(codebook, "sigil_prefix", "\u03a3") or "\u03a3")
    encoded_present = any(
        0 <= pos < len(result) and sigil_prefix in _get_message_text(result[pos])
        for pos, _original in encoded_positions
    )

    if encoded_present and codebook_preamble and codebook_preamble.strip():
        # Find the first non-system message and inject codebook context before it
        insert_idx = 0
        for idx, msg in enumerate(result):
            if not isinstance(msg, SystemMessage):
                insert_idx = idx
                break

        # D3B3: the legend is injected whenever sigils are actually present.
        # It is not gated on estimated net savings, because emitting a sigil the
        # model cannot decode silently destroys meaning - a much larger cost
        # than the preamble tokens it was trying to avoid.
        codebook_msg = HumanMessage(
            content=f"[CODEBOOK - Use these abbreviations to decode messages:]\n{codebook_preamble}"
        )
        result.insert(insert_idx, codebook_msg)
    elif encoded_present:
        # No legend is available, so revert every substitution rather than ship
        # undecodable sigils to the model.
        for pos, original in encoded_positions:
            if 0 <= pos < len(result):
                result[pos] = original

    # Log results
    after_tokens = count_message_tokens(result)
    if after_tokens < total_tokens:
        savings_pct = ((total_tokens - after_tokens) / total_tokens) * 100
        logger.info(
            f"Token efficiency layer: {total_tokens:,} → {after_tokens:,} tokens "
            f"({savings_pct:.1f}% reduction, {len(messages)} → {len(result)} messages)"
        )
    else:
        logger.debug("Token efficiency layer: no additional savings achieved")

    return result


def reset_session():
    """Reset the codebook and compressor state for a new session."""
    global _codebook, _compressor
    _codebook = None
    _compressor = None
