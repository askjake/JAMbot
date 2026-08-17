"""
Rule-based conversation summarization for aged turns.

The summary is intentionally structured and deterministic.  It is generated from
old turns immediately before they would otherwise be dropped from active context,
without an LLM call.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.message.compression import count_tokens

logger = logging.getLogger(__name__)

_PATH_RE = re.compile(r"(?:s3://[^\s'\"<>]+|https?://[^\s'\"<>]+|/[A-Za-z0-9._/\-]+)")
_ERROR_RE = re.compile(r"\b(error|exception|traceback|failed|failure|timeout|denied|unauthorized|not found|crashloop|segfault)\b", re.IGNORECASE)
_DECISION_RE = re.compile(r"\b(decision|decided|recommend|recommended|root cause|fixed|implemented|verified|accepted|rejected|blocker|next step|result)\b", re.IGNORECASE)


@dataclass
class TurnSummary:
    index: int
    user_goal: str = ""
    tools: list[str] | None = None
    outcome: str = ""
    artifacts: list[str] | None = None
    issues: list[str] | None = None

    def render(self) -> list[str]:
        lines: list[str] = []
        if self.user_goal:
            lines.append(f"- Turn {self.index} user goal: {self.user_goal}")
        if self.tools:
            lines.append(f"- Turn {self.index} tools: {', '.join(self.tools[:10])}")
        if self.outcome:
            lines.append(f"- Turn {self.index} outcome: {self.outcome}")
        if self.artifacts:
            lines.append(f"- Turn {self.index} artifacts: {', '.join(self.artifacts[:8])}")
        if self.issues:
            lines.append(f"- Turn {self.index} issues: {' | '.join(self.issues[:5])}")
        return lines


def summarize_turns(
    turns: Iterable[list[BaseMessage]],
    start_index: int = 1,
    max_tokens: int = 3000,
    existing_summary: str | None = None,
) -> str:
    """Build a compact structured summary for old turns."""
    turn_list = list(turns)
    if not turn_list and existing_summary:
        return _fit_summary(existing_summary, max_tokens)
    if not turn_list:
        return ""

    end_index = start_index + len(turn_list) - 1
    lines: list[str] = [f"[CONVERSATION CONTEXT - Turns {start_index}-{end_index}]"]

    if existing_summary:
        existing_lines = [line for line in existing_summary.splitlines() if line.strip()]
        if existing_lines:
            lines.append("- Prior compressed context retained below:")
            lines.extend(_prefix_if_needed(existing_lines[:40], prefix="  "))

    global_facts: list[str] = []
    global_artifacts: list[str] = []
    global_issues: list[str] = []

    for offset, turn in enumerate(turn_list):
        summary = summarize_turn(turn, index=start_index + offset)
        lines.extend(summary.render())
        _extend_unique(global_artifacts, summary.artifacts or [], limit=24)
        _extend_unique(global_issues, summary.issues or [], limit=20)

        facts = _extract_decision_lines(_messages_to_text(turn), limit=3)
        _extend_unique(global_facts, facts, limit=24)

    if global_facts:
        lines.append("- Established facts / decisions: " + " | ".join(global_facts[:12]))
    if global_artifacts:
        lines.append("- Referenced paths / URLs: " + ", ".join(global_artifacts[:12]))
    if global_issues:
        lines.append("- Recurrent issues / errors: " + " | ".join(global_issues[:10]))

    return _fit_summary("\n".join(lines), max_tokens)


def summarize_turn(turn: list[BaseMessage], index: int) -> TurnSummary:
    """Summarize one user turn plus following AI/tool messages."""
    user_goal = ""
    tools: list[str] = []
    outcome_candidates: list[str] = []
    artifacts: list[str] = []
    issues: list[str] = []

    for msg in turn:
        if isinstance(msg, HumanMessage) and not user_goal:
            user_goal = _clean(_content_to_text(msg.content), 220)
        elif isinstance(msg, AIMessage):
            for tool_call in getattr(msg, "tool_calls", None) or []:
                name = tool_call.get("name") if isinstance(tool_call, dict) else None
                if name and name not in tools:
                    tools.append(str(name))
            outcome_candidates.extend(_extract_decision_lines(_content_to_text(msg.content), limit=4))
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", None)
            if tool_name and tool_name not in tools:
                tools.append(str(tool_name))
            text = _content_to_text(msg.content)
            _extend_unique(artifacts, _extract_paths(text), limit=10)
            _extend_unique(issues, _extract_error_lines(text), limit=8)
            outcome = _summarize_tool_result(text)
            if outcome:
                outcome_candidates.append(outcome)

    outcome = _clean(" | ".join(_dedupe(outcome_candidates)[:5]), 420)
    return TurnSummary(
        index=index,
        user_goal=user_goal,
        tools=tools,
        outcome=outcome,
        artifacts=artifacts,
        issues=issues,
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, default=str))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, default=str)
    return str(content or "")


def _messages_to_text(messages: list[BaseMessage]) -> str:
    return "\n".join(_content_to_text(getattr(msg, "content", "")) for msg in messages)


def _extract_decision_lines(text: str, limit: int = 5) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip(" -*\t")
        if len(line) < 10:
            continue
        if not _DECISION_RE.search(line):
            continue
        lines.append(_clean(line, 220))
        if len(lines) >= limit:
            break
    return _dedupe(lines)


def _extract_paths(text: str, limit: int = 10) -> list[str]:
    out: list[str] = []
    for match in _PATH_RE.findall(text):
        cleaned = match.rstrip(".,);]")
        if cleaned not in out:
            out.append(cleaned)
        if len(out) >= limit:
            break
    return out


def _extract_error_lines(text: str, limit: int = 6) -> list[str]:
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if len(line) < 6:
            continue
        if _ERROR_RE.search(line):
            out.append(_clean(line, 220))
        if len(out) >= limit:
            break
    return _dedupe(out)


def _summarize_tool_result(text: str) -> str:
    if not text:
        return ""

    lowered = text.lower()
    line_count = len(text.splitlines())
    char_count = len(text)

    exit_match = re.search(r"\bexit(?:\s+code)?\s*[:=]?\s*(-?\d+)\b", text, flags=re.IGNORECASE)
    if exit_match:
        return f"tool result exit {exit_match.group(1)}, {line_count} lines, {char_count:,} chars"

    count_match = re.search(r"\b(?:found|total|count|items?|results?|rows?)\D{0,20}(\d{1,7})\b", text, flags=re.IGNORECASE)
    if count_match:
        return f"tool result reported count {count_match.group(1)}, {line_count} lines"

    if any(term in lowered for term in ("success", "passed", "complete", "created", "updated")):
        return f"tool result indicated success, {line_count} lines"
    if _ERROR_RE.search(text):
        return f"tool result included errors, {line_count} lines"
    return f"tool result returned {line_count} lines, {char_count:,} chars"


def _fit_summary(text: str, max_tokens: int) -> str:
    if count_tokens(text) <= max_tokens:
        return text

    lines = text.splitlines()
    if not lines:
        return text

    header = lines[0]
    retained = [header]
    for line in lines[1:]:
        candidate = "\n".join(retained + [line, "- Summary truncated to fit tier budget."])
        if count_tokens(candidate) > max_tokens:
            break
        retained.append(line)
    retained.append("- Summary truncated to fit tier budget.")
    return "\n".join(retained)


def _prefix_if_needed(lines: list[str], prefix: str) -> list[str]:
    return [line if line.startswith(prefix) else prefix + line for line in lines]


def _extend_unique(target: list[str], items: Iterable[str], limit: int) -> None:
    for item in items:
        if item and item not in target:
            target.append(item)
        if len(target) >= limit:
            return


def _dedupe(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = re.sub(r"\W+", " ", item).lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _clean(text: str, max_len: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 3].rstrip() + "..."
