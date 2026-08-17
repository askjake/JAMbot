"""
Rule-based compression for older AI messages.

This module deliberately does not call an LLM.  It keeps the parts of an
assistant response that are useful after the turn has aged: decisions,
conclusions, errors, code signatures, and a small amount of leading/trailing
context.  Tool-call metadata is preserved by cloning the original AIMessage and
only replacing ``content``.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage

from app.message.compression import count_tokens

logger = logging.getLogger(__name__)

_STATUS_PREAMBLE_RE = re.compile(
    r"^\s*(?:let me|i(?:'ll| will| am going to)|i’m going to|i can|i need to|now i(?:'ll| will)|next i(?:'ll| will))\b",
    flags=re.IGNORECASE,
)
_KEY_LINE_RE = re.compile(
    r"\b(accepted|blocked|blocker|bug|complete|completed|conclusion|decision|error|failed|failure|fixed|implemented|issue|next|pass|passed|recommend|recommended|regression|root cause|tested|updated|verified|warning)\b",
    flags=re.IGNORECASE,
)
_CODE_BLOCK_RE = re.compile(r"```([^\n`]*)\n(.*?)```", flags=re.DOTALL)


def compress_ai_message(
    message: AIMessage,
    target_tokens: int = 400,
    max_code_lines: int = 10,
) -> AIMessage:
    """
    Return a copy of ``message`` with a compact content string.

    The returned AIMessage keeps the original tool-call declarations and other
    metadata.  This is important for Bedrock/LangChain tool_call_id pairing.
    """
    content = getattr(message, "content", "")
    if not isinstance(content, str) or not content.strip():
        return message

    if count_tokens(content) <= target_tokens:
        return message

    compressed = compress_ai_content(
        content,
        target_tokens=target_tokens,
        max_code_lines=max_code_lines,
    )
    if compressed == content:
        return message

    return _copy_message_with_content(message, compressed)


def compress_ai_content(
    content: str,
    target_tokens: int = 400,
    max_code_lines: int = 10,
) -> str:
    """Compress raw AI content without relying on any model call."""
    if not content:
        return content

    code_summaries = _summarize_code_blocks(content, max_code_lines=max_code_lines)
    content_without_code = _CODE_BLOCK_RE.sub("\n[code block summarized below]\n", content)

    paragraphs = _paragraphs(content_without_code)
    filtered = _drop_display_only_paragraphs(paragraphs)

    first = filtered[0] if filtered else ""
    last = filtered[-1] if len(filtered) > 1 else ""
    key_lines = _extract_key_lines(content_without_code, limit=12)

    sections: list[str] = ["[COMPRESSED PRIOR AI RESPONSE]"]
    if first:
        sections.append(f"Initial answer: {_clean(first, 700)}")
    if key_lines:
        sections.append("Key retained points:\n" + "\n".join(f"- {_clean(line, 260)}" for line in key_lines))
    if code_summaries:
        sections.append("Code retained:\n" + "\n".join(code_summaries))
    if last and last != first:
        sections.append(f"Final answer: {_clean(last, 700)}")

    compressed = "\n\n".join(part for part in sections if part.strip())
    compressed = _dedupe_lines(compressed)
    return _fit_to_token_budget(compressed, target_tokens)


def _copy_message_with_content(message: AIMessage, content: str) -> AIMessage:
    """Clone a LangChain AIMessage while preserving tool calls and metadata."""
    try:
        return message.model_copy(update={"content": content})
    except Exception:
        pass

    try:
        return message.copy(update={"content": content})
    except Exception:
        pass

    kwargs: dict[str, Any] = {"content": content}
    for attr in (
        "tool_calls",
        "invalid_tool_calls",
        "additional_kwargs",
        "response_metadata",
        "id",
        "name",
        "usage_metadata",
    ):
        value = getattr(message, attr, None)
        if value not in (None, [], {}):
            kwargs[attr] = value
    try:
        return AIMessage(**kwargs)
    except TypeError:
        kwargs.pop("usage_metadata", None)
        return AIMessage(**kwargs)


def _paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n")
    parts = [p.strip() for p in re.split(r"\n\s*\n+", normalized) if p.strip()]
    if parts:
        return parts
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def _drop_display_only_paragraphs(paragraphs: list[str]) -> list[str]:
    kept: list[str] = []
    for idx, paragraph in enumerate(paragraphs):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue

        # Drop common status/preamble paragraphs except when they are the only
        # potentially useful context.
        if idx < 2 and all(_STATUS_PREAMBLE_RE.search(line) for line in lines[:2]):
            continue

        tableish = sum(1 for line in lines if "|" in line)
        separatorish = sum(1 for line in lines if re.fullmatch(r"[-:|\s]+", line))
        if tableish >= max(3, len(lines) // 2) or separatorish >= 1:
            # Tables are usually presentation artifacts in old AI turns.  Key
            # data rows that mention errors/decisions are recovered by
            # _extract_key_lines().
            continue

        kept.append(paragraph)
    return kept or paragraphs[:2]


def _extract_key_lines(text: str, limit: int = 12) -> list[str]:
    key_lines: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip(" -*\t")
        if not line or len(line) < 8:
            continue
        if _STATUS_PREAMBLE_RE.search(line):
            continue
        if not _KEY_LINE_RE.search(line):
            continue
        key = re.sub(r"\W+", " ", line).lower().strip()
        if key in seen:
            continue
        seen.add(key)
        key_lines.append(line)
        if len(key_lines) >= limit:
            break
    return key_lines


def _summarize_code_blocks(content: str, max_code_lines: int = 10) -> list[str]:
    summaries: list[str] = []
    for idx, match in enumerate(_CODE_BLOCK_RE.finditer(content), start=1):
        language = (match.group(1) or "text").strip() or "text"
        code = match.group(2).strip("\n")
        lines = code.splitlines()
        retained = lines[:max_code_lines]
        signature = _detect_code_signature(lines)
        header = f"- block {idx} ({language}, {len(lines)} lines"
        if signature:
            header += f", signature: {signature}"
        header += ")"
        rendered = [header]
        rendered.extend(f"  {line[:180]}" for line in retained)
        if len(lines) > max_code_lines:
            rendered.append(f"  [... code block truncated; {len(lines)} lines total ...]")
        summaries.append("\n".join(rendered))
        if len(summaries) >= 4:
            break
    return summaries


def _detect_code_signature(lines: list[str]) -> str:
    signature_patterns = (
        r"\bdef\s+\w+\s*\([^)]*\)",
        r"\bclass\s+\w+",
        r"\basync\s+def\s+\w+\s*\([^)]*\)",
        r"\bfunction\s+\w+\s*\([^)]*\)",
        r"\bconst\s+\w+\s*=",
        r"\bexport\s+(?:default\s+)?(?:function|class|const)\s+\w+",
    )
    for line in lines[:80]:
        stripped = line.strip()
        for pattern in signature_patterns:
            match = re.search(pattern, stripped)
            if match:
                return match.group(0)[:120]
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return ""


def _dedupe_lines(text: str) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        key = re.sub(r"\W+", " ", line).lower().strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(line)
    return "\n".join(out)


def _fit_to_token_budget(text: str, target_tokens: int) -> str:
    if count_tokens(text) <= target_tokens:
        return text

    # Character-ratio shrink is fast and deterministic.  Iterate a couple of
    # times because token density varies for code and paths.
    budget_chars = max(600, target_tokens * 4)
    trimmed = text[:budget_chars].rstrip()
    if len(text) > len(trimmed):
        trimmed += "\n[... compressed AI response truncated to target budget ...]"

    while count_tokens(trimmed) > target_tokens and len(trimmed) > 800:
        trimmed = trimmed[: int(len(trimmed) * 0.85)].rstrip()
        trimmed += "\n[... compressed AI response truncated to target budget ...]"
    return trimmed


def _clean(text: str, max_len: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."
