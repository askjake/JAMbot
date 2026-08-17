"""
Progressive detail levels for aged ToolMessages.

Fresh tool results stay detailed.  As turns pass, older results degrade to key
facts and then one-line provenance so SSH/file/search/tool output does not keep
re-entering every model call at full size.  Web browse behavior remains
compatible with the previous implementation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.message.compression import count_tokens
from app.message.tool_message_compressor import (
    ToolContentType,
    classify_tool_content,
    compress_json_tool_output,
)
from app.tools.tool_result_compressor import compress_web_browse_result

try:  # Keep the module resilient during tests and partial imports.
    from app.tools.tool_result_compressor import WEB_BROWSE_TOOLS
except Exception:  # pragma: no cover - defensive only
    WEB_BROWSE_TOOLS = set()

logger = logging.getLogger(__name__)

SHELL_TOOL_HINTS = (
    "ssh",
    "shell",
    "terminal",
    "command",
    "run_command",
    "local_run",
    "execute",
    "bash",
    "powershell",
)
FILE_TOOL_HINTS = (
    "read_file",
    "write_file",
    "edit_file",
    "patch_file",
    "materialize",
    "file_read",
    "file_write",
    "cat_file",
)
SEARCH_TOOL_HINTS = (
    "search",
    "confluence",
    "jira",
    "github_search",
    "web_search",
    "list_",
    "find_",
)


@dataclass
class ProgressiveToolResult:
    tool_name: str
    tool_call_id: str
    turn_created: int
    one_liner: str
    key_facts: str
    full_content: str
    category: str = "generic"

    def get_content_for_turn(self, current_turn: int) -> str:
        age = current_turn - self.turn_created
        if age <= 0:
            return self.full_content

        if self.category == "browse":
            if age <= 2:
                return f"[Prior browse: {self.one_liner}]\n{self.key_facts}"
            return f"[Turn {self.turn_created}: {self.one_liner}]"

        if self.category == "shell":
            if age <= 2:
                return f"[Prior shell output: {self.one_liner}]\n{self.key_facts}"
            return f"[Turn {self.turn_created}: {self.one_liner}]"

        if self.category in {"file", "search", "json_report"}:
            if age <= 1:
                return self.full_content
            return f"[Turn {self.turn_created}: {self.one_liner}]\n{self.key_facts}"

        if age <= 2:
            return f"[Prior tool output: {self.one_liner}]\n{self.key_facts}"
        return f"[Turn {self.turn_created}: {self.one_liner}]"


def generate_progressive_levels(
    tool_name: str,
    raw_result: str,
    tool_call_id: str = "",
    turn_created: int = 0,
) -> ProgressiveToolResult:
    """Generate one-line, key-fact, and full levels for any tool result."""
    if not isinstance(raw_result, str):
        raw_result = str(raw_result)

    content_type = classify_tool_content(tool_name=tool_name, text=raw_result)
    if content_type is ToolContentType.JSON_REPORT:
        return _generate_json_report_levels(tool_name, raw_result, tool_call_id, turn_created)
    if content_type is ToolContentType.WEB:
        return _generate_browse_levels(tool_name, raw_result, tool_call_id, turn_created)

    category = classify_tool(tool_name)
    if category == "browse":
        return _generate_browse_levels(tool_name, raw_result, tool_call_id, turn_created)
    if category == "shell":
        return _generate_shell_levels(tool_name, raw_result, tool_call_id, turn_created)
    if category == "file":
        return _generate_file_levels(tool_name, raw_result, tool_call_id, turn_created)
    if category == "search":
        return _generate_search_levels(tool_name, raw_result, tool_call_id, turn_created)
    return _generate_generic_levels(tool_name, raw_result, tool_call_id, turn_created)


def classify_tool(tool_name: str) -> str:
    """Classify a tool by stable name hints."""
    name = (tool_name or "").lower()
    if tool_name in WEB_BROWSE_TOOLS or "browse" in name or "browser" in name:
        return "browse"
    if any(hint in name for hint in SHELL_TOOL_HINTS):
        return "shell"
    if any(hint in name for hint in FILE_TOOL_HINTS):
        return "file"
    if any(hint in name for hint in SEARCH_TOOL_HINTS):
        return "search"
    return "generic"


def _generate_browse_levels(tool_name: str, raw_result: str, tool_call_id: str, turn_created: int) -> ProgressiveToolResult:
    url = _extract_first(r"^URL:\s*(.+)$", raw_result) or _extract_first(r"^New URL:\s*(.+)$", raw_result) or "unknown URL"
    status = _extract_first(r"^Status:\s*(\d+|unknown).*$", raw_result) or _extract_status_from_json(raw_result) or "?"

    metric = _extract_count_metric(raw_result)
    one_liner = f"Browsed {_shorten_url(url)} → {status}"
    if metric:
        one_liner += f", {metric}"

    key_facts = _generate_key_facts(raw_result)
    full_content = compress_web_browse_result(tool_name, raw_result, token_budget=3000)

    return ProgressiveToolResult(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        turn_created=turn_created,
        one_liner=one_liner,
        key_facts=key_facts,
        full_content=full_content,
        category="browse",
    )


def _generate_shell_levels(tool_name: str, raw_result: str, tool_call_id: str, turn_created: int) -> ProgressiveToolResult:
    lines = raw_result.splitlines()
    command = _extract_command(raw_result) or tool_name or "shell command"
    exit_code = _extract_exit_code(raw_result)
    error_lines = _extract_issue_lines(raw_result, limit=12)
    one_liner = f"Ran `{_clean_line(command, 120)}` → exit {exit_code if exit_code is not None else '?'}"
    one_liner += f", {len(lines)} lines output"

    key_parts = [
        f"tool: {tool_name or 'unknown'}",
        f"command: {_clean_line(command, 240)}",
        f"exit_code: {exit_code if exit_code is not None else 'unknown'}",
        f"line_count: {len(lines)}",
    ]
    if lines:
        key_parts.append("first_20_lines:\n" + "\n".join(f"{idx:02d}: {_clean_line(line, 260)}" for idx, line in enumerate(lines[:20], start=1)))
    if error_lines:
        key_parts.append("error_lines:\n" + "\n".join(f"- {_clean_line(line, 260)}" for line in error_lines))
    if len(lines) > 20:
        key_parts.append("last_5_lines:\n" + "\n".join(f"- {_clean_line(line, 260)}" for line in lines[-5:]))

    return ProgressiveToolResult(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        turn_created=turn_created,
        one_liner=one_liner,
        key_facts=_fit_text("\n".join(key_parts), 900),
        full_content=raw_result,
        category="shell",
    )


def _generate_file_levels(tool_name: str, raw_result: str, tool_call_id: str, turn_created: int) -> ProgressiveToolResult:
    filename = _extract_file_path(raw_result) or "unknown file"
    line_count = len(raw_result.splitlines())
    action = "wrote" if re.search(r"\b(wrote|written|created|updated|patched|saved)\b", raw_result, re.I) else "read"
    one_liner = f"{action.capitalize()} `{_clean_line(filename, 120)}` ({line_count} lines, {len(raw_result):,} chars)"

    issue_lines = _extract_issue_lines(raw_result, limit=8)
    key_parts = [
        f"tool: {tool_name or 'unknown'}",
        f"file: {filename}",
        f"action: {action}",
        f"line_count: {line_count}",
        f"char_count: {len(raw_result)}",
    ]
    if issue_lines:
        key_parts.append("notable_lines:\n" + "\n".join(f"- {_clean_line(line, 220)}" for line in issue_lines))

    return ProgressiveToolResult(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        turn_created=turn_created,
        one_liner=one_liner,
        key_facts="\n".join(key_parts),
        full_content=raw_result,
        category="file",
    )


def _generate_search_levels(tool_name: str, raw_result: str, tool_call_id: str, turn_created: int) -> ProgressiveToolResult:
    titles = _extract_titles(raw_result, limit=10)
    count = _extract_result_count(raw_result) or len(titles)
    one_liner = f"Searched with {tool_name or 'search tool'}, found {count if count is not None else '?'} results"
    if titles:
        one_liner += f": {', '.join(_clean_line(title, 50) for title in titles[:3])}"

    key_parts = [
        f"tool: {tool_name or 'unknown'}",
        f"result_count: {count if count is not None else 'unknown'}",
    ]
    if titles:
        key_parts.append("top_results:\n" + "\n".join(f"- {_clean_line(title, 180)}" for title in titles))
    issues = _extract_issue_lines(raw_result, limit=5)
    if issues:
        key_parts.append("issues:\n" + "\n".join(f"- {_clean_line(line, 180)}" for line in issues))

    return ProgressiveToolResult(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        turn_created=turn_created,
        one_liner=one_liner,
        key_facts=_fit_text("\n".join(key_parts), 500),
        full_content=raw_result,
        category="search",
    )


def _generate_json_report_levels(tool_name: str, raw_result: str, tool_call_id: str, turn_created: int) -> ProgressiveToolResult:
    """Generate progressive levels for structured metadata/report tools."""
    count = _extract_result_count(raw_result)
    one_liner = f"Ran {tool_name or 'JSON report tool'} → structured JSON report, {len(raw_result):,} chars"
    if count is not None:
        one_liner += f", count={count}"

    compact = compress_json_tool_output(
        content=raw_result,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        max_chars=1800,
    )
    return ProgressiveToolResult(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        turn_created=turn_created,
        one_liner=one_liner,
        key_facts=_fit_text(compact, 900),
        full_content=raw_result,
        category="json_report",
    )


def _generate_generic_levels(tool_name: str, raw_result: str, tool_call_id: str, turn_created: int) -> ProgressiveToolResult:
    lines = raw_result.splitlines()
    metric = _extract_count_metric(raw_result)
    one_liner = f"Ran {tool_name or 'tool'} → {len(lines)} lines, {len(raw_result):,} chars"
    if metric:
        one_liner += f", {metric}"

    issues = _extract_issue_lines(raw_result, limit=8)
    key_parts = [
        f"tool: {tool_name or 'unknown'}",
        f"line_count: {len(lines)}",
        f"char_count: {len(raw_result)}",
    ]
    if issues:
        key_parts.append("issues:\n" + "\n".join(f"- {_clean_line(line, 220)}" for line in issues))
    elif lines:
        key_parts.append("preview:\n" + "\n".join(f"{idx:02d}: {_clean_line(line, 220)}" for idx, line in enumerate(lines[:8], start=1)))

    return ProgressiveToolResult(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        turn_created=turn_created,
        one_liner=one_liner,
        key_facts=_fit_text("\n".join(key_parts), 500),
        full_content=raw_result,
        category="generic",
    )


def _extract_first(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _extract_status_from_json(text: str) -> str | None:
    match = re.search(r'"status_code"\s*:\s*"?(\d+)"?', text)
    return match.group(1) if match else None


def _extract_command(text: str) -> str | None:
    patterns = (
        r"^\s*(?:command|cmd|\$)\s*[:=]\s*(.+)$",
        r"^\s*\$\s+(.+)$",
        r"Ran\s+`([^`]+)`",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_exit_code(text: str) -> str | None:
    match = re.search(r"\bexit(?:\s+code)?\s*[:=]?\s*(-?\d+)\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    if re.search(r"\b(returncode|status)\s*[:=]\s*0\b", text, flags=re.IGNORECASE):
        return "0"
    return None


def _extract_file_path(text: str) -> str | None:
    patterns = (
        r"(?:file|path|filename)\s*[:=]\s*([^\s'\"<>]+)",
        r"(s3://[^\s'\"<>]+|/[A-Za-z0-9._/\-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).rstrip(".,);]")
    return None


def _extract_titles(text: str, limit: int = 10) -> list[str]:
    titles: list[str] = []
    patterns = (
        r"^\s*(?:title|name)\s*[:=]\s*(.+)$",
        r"\[([^\]]{3,120})\]\(https?://[^)]+\)",
        r"\[([^\]]{3,120})\]\s*->\s*https?://\S+",
        r'"title"\s*:\s*"([^"]{3,120})"',
    )
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.MULTILINE | re.IGNORECASE):
            title = _clean_line(match, 120)
            if title and title not in titles:
                titles.append(title)
            if len(titles) >= limit:
                return titles
    return titles


def _extract_result_count(text: str) -> int | None:
    patterns = (
        r"\b(?:total|count|items?|rows?|results?)\D{0,20}(\d{1,7})\b",
        r'"(?:total|count|items|rows|results)"\s*:\s*(\d{1,7})',
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def _shorten_url(url: str, max_len: int = 96) -> str:
    if len(url) <= max_len:
        return url
    keep_tail = max(20, max_len // 3)
    return url[: max_len - keep_tail - 3] + "..." + url[-keep_tail:]


def _extract_count_metric(text: str) -> str:
    link_count = max(text.count("] ->"), text.count("href"))
    if link_count:
        return f"{link_count} links"

    table_rows = _count_table_rows(text)
    if table_rows:
        return f"{table_rows} table rows"

    event_match = re.search(r'"data_events_found"\s*:\s*(\d+)', text)
    if event_match:
        return f"{event_match.group(1)} data events"

    item_match = re.search(r'(?i)\b(total|count|items?|rows?|results?)\D{0,20}(\d{1,6})\b', text)
    if item_match:
        return f"{item_match.group(2)} {item_match.group(1).lower()}"

    char_count = len(text)
    if char_count:
        return f"{char_count:,} chars"
    return ""


def _generate_key_facts(raw_result: str, token_budget: int = 300) -> str:
    parts: list[str] = []

    headings = re.findall(r"^#{1,4}\s+(.+)$", raw_result, flags=re.MULTILINE)[:12]
    if headings:
        parts.append("Sections: " + "; ".join(_clean_line(h, 80) for h in headings))

    issue_lines = _extract_issue_lines(raw_result, limit=6)
    if issue_lines:
        parts.append("Issues: " + " | ".join(issue_lines))

    metrics = _extract_metrics(raw_result)
    if metrics:
        parts.append("Metrics: " + "; ".join(metrics[:12]))

    table_rows = _count_table_rows(raw_result)
    if table_rows:
        parts.append(f"Tables: approximately {table_rows} data rows retained in source result")

    links = re.findall(r"\[([^\]]{1,80})\]\s*->\s*(\S+)", raw_result)[:10]
    if links:
        rendered_links = [f"{_clean_line(text, 50)} -> {_shorten_url(url, 70)}" for text, url in links]
        parts.append("Top links: " + " | ".join(rendered_links))

    if not parts:
        body = raw_result.split("\n---\n\n", 1)[-1]
        preview = re.sub(r"\s+", " ", body).strip()[:1000]
        parts.append("Preview: " + preview if preview else "No key facts extracted.")

    key_facts = "\n".join(parts)
    return _fit_text(key_facts, token_budget)


def _extract_issue_lines(text: str, limit: int = 8) -> list[str]:
    issue_terms = ("error", "warning", "failed", "failure", "exception", "timeout", "denied", "traceback")
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(term in stripped.lower() for term in issue_terms):
            lines.append(_clean_line(stripped, 260))
        if len(lines) >= limit:
            break
    return lines


def _extract_metrics(text: str) -> list[str]:
    candidates = []
    seen = set()
    metric_patterns = [
        r"\b[A-Za-z][A-Za-z0-9 _/-]{0,40}:\s*-?\d[\d,.%:/-]*",
        r"\b\d[\d,.%]*\s+(?:items?|rows?|results?|errors?|warnings?|events?|seconds?|ms|tokens?|lines?)\b",
    ]
    for pattern in metric_patterns:
        for match in re.findall(pattern, text):
            metric = _clean_line(match, 100)
            key = metric.lower()
            if key not in seen:
                seen.add(key)
                candidates.append(metric)
            if len(candidates) >= 16:
                return candidates
    return candidates


def _count_table_rows(text: str) -> int:
    rows = 0
    for line in text.splitlines():
        stripped = line.strip()
        if "|" in stripped and not re.fullmatch(r"[-|: ]+", stripped):
            rows += 1
    return max(0, rows - 1) if rows else 0


def _fit_text(text: str, token_budget: int) -> str:
    while count_tokens(text) > token_budget and len(text) > 200:
        text = text[: int(len(text) * 0.85)].rstrip() + "..."
    return text


def _clean_line(line: Any, max_len: int) -> str:
    if not isinstance(line, str):
        line = json.dumps(line, ensure_ascii=False, default=str)
    cleaned = re.sub(r"\s+", " ", line).strip()
    return cleaned if len(cleaned) <= max_len else cleaned[: max_len - 3] + "..."
