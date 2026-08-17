"""
Content-aware compression for oversized LangChain ToolMessages.

Large tool results are not all the same shape.  Raw STB log slices benefit from
log-specific extraction, while JSON reports, JIRA/Confluence payloads, browser
HTML, and unknown plain text need different compression strategies.  This module
keeps that routing explicit so generic metadata responses are never force-fit
into an STB log schema with meaningless ``unknown`` fields.
"""

from __future__ import annotations

import html
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 12_000
DEFAULT_FIRST_LINES = 40
DEFAULT_TAIL_LINES = 10
JSON_MAX_ITEMS = 20
JSON_MAX_KEYS = 80
JSON_MAX_DEPTH = 8
JSON_MAX_STRING_CHARS = 1_500

_RECEIVER_RE = re.compile(r"\bR\d{6,}\b", re.IGNORECASE)
_PATH_RE = re.compile(r"(s3://[^\s'\"<>]+|https?://[^\s'\"<>]+|/[A-Za-z0-9._/\-]+)")
_TIMESTAMP_LINE_RE = re.compile(
    r"^\s*(?:\d{4}-\d{2}-\d{2}[T\s]|\d{2}:\d{2}:\d{2}(?:\.\d+)?\b|\d{2}/\d{2}/\d{2,4}\b)"
)
_HTML_RE = re.compile(r"<\s*(?:html|body|div|span|script|style|table|a|p|h[1-6])\b", re.IGNORECASE)
_SCRIPT_STYLE_RE = re.compile(r"<\s*(script|style)\b[\s\S]*?<\s*/\s*\1\s*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"[ \t]+")

_JSON_REPORT_TOOL_HINTS = (
    "jira",
    "confluence",
    "validate_mcp_token_efficiency",
    "get_tool_info",
    "get_heavy_auth_status",
    "health",
    "status",
    "catalog",
    "report",
)

_WEB_TOOL_HINTS = (
    "public_web_search",
    "web_browse",
    "local_web_browse",
    "browser",
    "browse_api",
    "headless",
)

_RAW_STB_LOG_TOOL_HINTS = (
    "read_log",
    "read_parsed_logs",
    "filter_log",
    "filter_log_lines",
)

_LOG_TEXT_HINTS = (
    "qt_gui",
    "invididebuglog",
    "sg_server",
    "input_mgr",
    "stbctrl",
    "procmgr",
    "nal",
    "ccshare/",
)

_PROTECTED_TOP_LEVEL_KEYS = (
    "_meta",
    "ok",
    "blocked",
    "status",
    "validation_mode",
    "detail_level",
    "issue_profile",
    "heavy_tools_invoked",
    "heavy_tools_blocked_by_design",
    "runtime_policy",
    "calls",
    "compact_results",
    "token_safety_note",
    "inline_chars_est",
    "full_report_s3",
    "summary_s3",
    "limits",
    "errors",
    "error",
)


class ToolContentType(str, Enum):
    """High-level content classes used for ToolMessage compression routing."""

    STB_LOG = "stb_log"
    JSON_REPORT = "json_report"
    WEB = "web"
    TEXT = "text"


@dataclass(slots=True)
class ToolCompressionResult:
    """Result returned by the content-aware compression dispatcher."""

    content: str
    content_type: ToolContentType
    strategy: str
    original_chars: int
    compressed_chars: int

    @property
    def was_compressed(self) -> bool:
        """Return True when the compressed content is smaller than the source."""

        return self.compressed_chars < self.original_chars


def compress_tool_message_content(
    *,
    tool_name: str,
    content: Any,
    tool_call_id: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    first_lines: int = DEFAULT_FIRST_LINES,
    tail_lines: int = DEFAULT_TAIL_LINES,
) -> ToolCompressionResult:
    """
    Compress a large tool result using a strategy matched to its actual content.

    Args:
        tool_name: LangChain tool name or MCP tool name when known.
        content: Raw ToolMessage content.  Strings are used directly; JSON-like
            objects are serialized with stable formatting.
        tool_call_id: Optional tool-call identifier for traceability.
        max_chars: Maximum compressed content length.
        first_lines: Number of leading lines retained for head/tail strategies.
        tail_lines: Number of trailing lines retained for head/tail strategies.

    Returns:
        ToolCompressionResult with compressed content and routing metadata.
    """

    text = _tool_content_to_text(content)
    obj = _maybe_json_loads(text)
    content_type = classify_tool_content(tool_name=tool_name, text=text, parsed_json=obj)

    try:
        if content_type is ToolContentType.STB_LOG:
            compressed = compress_stb_log_tool_output(
                content=text,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                max_chars=max_chars,
                first_lines=first_lines,
                tail_lines=tail_lines,
            )
            strategy = "stb_log_extractor"
        elif content_type is ToolContentType.JSON_REPORT:
            compressed = compress_json_tool_output(
                content=text,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                max_chars=max_chars,
            )
            strategy = "compact_json_preserve_fields"
        elif content_type is ToolContentType.WEB:
            compressed = compress_web_tool_output(
                content=text,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                max_chars=max_chars,
                first_lines=first_lines,
                tail_lines=tail_lines,
            )
            strategy = "html_text_preview"
        else:
            compressed = compress_plain_tool_output(
                content=text,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                max_chars=max_chars,
                first_lines=first_lines,
                tail_lines=tail_lines,
            )
            strategy = "head_tail_truncation"
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Content-aware tool compression failed for %s: %s", tool_name, exc)
        compressed = compress_plain_tool_output(
            content=text,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            max_chars=max_chars,
            first_lines=first_lines,
            tail_lines=tail_lines,
        )
        content_type = ToolContentType.TEXT
        strategy = "head_tail_truncation_after_error"

    if len(compressed) > max_chars:
        compressed = _fit_chars(compressed, max_chars)

    return ToolCompressionResult(
        content=compressed,
        content_type=content_type,
        strategy=strategy,
        original_chars=len(text),
        compressed_chars=len(compressed),
    )


def classify_tool_content(tool_name: str, text: str, parsed_json: Any | None = None) -> ToolContentType:
    """
    Classify tool output by explicit server hint, tool name, and payload shape.

    Explicit MCP ``_meta.content_type`` hints win over heuristics.  Name-based
    routing is intentionally conservative: only raw log read/search/filter tools
    use the STB-log extractor; log-capsule/report tools remain JSON reports.
    """

    name = (tool_name or "").lower()
    obj = parsed_json if parsed_json is not None else _maybe_json_loads(text)

    hinted = _extract_meta_content_type(obj)
    if hinted:
        return hinted

    if any(hint in name for hint in _WEB_TOOL_HINTS):
        return ToolContentType.WEB

    if any(hint in name for hint in _JSON_REPORT_TOOL_HINTS):
        return ToolContentType.JSON_REPORT

    if _is_raw_stb_log_tool_name(name):
        return ToolContentType.STB_LOG

    if obj is not None:
        return ToolContentType.JSON_REPORT

    if _looks_like_html(text):
        return ToolContentType.WEB

    if _looks_like_stb_log_text(text):
        return ToolContentType.STB_LOG

    return ToolContentType.TEXT


def compress_stb_log_tool_output(
    *,
    content: Any,
    tool_name: str = "",
    tool_call_id: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    first_lines: int = DEFAULT_FIRST_LINES,
    tail_lines: int = DEFAULT_TAIL_LINES,
) -> str:
    """
    Compress raw STB log responses without fabricating unknown log records.

    If the payload does not contain log-specific signals, this function falls
    back to plain head/tail truncation instead of emitting ``receiver_id:
    unknown`` / ``log_name: unknown`` placeholder records.
    """

    text = _tool_content_to_text(content)
    obj = _maybe_json_loads(text)
    log_text = _extract_log_text_from_json(obj) if obj is not None else text

    if not _has_log_signal(text, obj=obj, extracted_log_text=log_text):
        return compress_plain_tool_output(
            content=text,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            max_chars=max_chars,
            first_lines=first_lines,
            tail_lines=tail_lines,
            header="[TRUNCATED NON-LOG TOOL OUTPUT]",
        )

    metadata = _extract_log_metadata(obj, text)
    raw_lines = [line.rstrip() for line in log_text.splitlines() if line.strip()]
    receivers = sorted(set(_RECEIVER_RE.findall(text)))
    paths = _extract_paths_from_text(text, limit=10)
    errors = _extract_error_lines(log_text, limit=8)

    lines: list[str] = [
        "[COMPRESSED LOG TOOL OUTPUT]",
        f"tool_name: {tool_name or 'unknown'}",
        f"tool_call_id: {tool_call_id or 'unknown'}",
        "content_type: stb_log",
        f"original_characters: {len(text)}",
        f"original_lines: {len(raw_lines)}",
    ]

    if metadata:
        lines.append("")
        lines.append("metadata:")
        for key, value in metadata.items():
            lines.append(f"- {key}: {_short_scalar(value, 500)}")

    if receivers:
        lines.append(f"receiver_ids: {', '.join(receivers[:20])}")
    if paths:
        lines.append("paths:")
        lines.extend(f"- {path}" for path in paths)
    if errors:
        lines.append("error_summary:")
        lines.extend(f"- {line}" for line in errors)

    record_previews = _extract_log_record_previews(obj, limit=20) if obj is not None else []
    if record_previews:
        lines.append("")
        lines.append("records_preview:")
        lines.extend(record_previews)

    lines.append("")
    lines.append(f"first_{first_lines}_lines:")
    if raw_lines:
        lines.extend(f"{idx:02d}: {_clean_line(line, 500)}" for idx, line in enumerate(raw_lines[:first_lines], start=1))
    else:
        lines.append("(no non-empty log lines detected)")

    if len(raw_lines) > first_lines and tail_lines > 0:
        lines.append("")
        lines.append(f"last_{tail_lines}_lines:")
        tail_start = max(first_lines + 1, len(raw_lines) - tail_lines + 1)
        lines.extend(f"{idx:02d}: {_clean_line(line, 500)}" for idx, line in enumerate(raw_lines[-tail_lines:], start=tail_start))

    return _fit_chars("\n".join(lines), max_chars)


def compress_json_tool_output(
    *,
    content: Any,
    tool_name: str = "",
    tool_call_id: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """
    Compress structured JSON while preserving keys and scalar semantics.

    Large arrays are sampled and annotated with omitted counts.  This is the
    correct fallback for metadata/report tools such as ``validate_mcp_token_efficiency``.
    """

    text = _tool_content_to_text(content)
    obj = _maybe_json_loads(text)
    if obj is None:
        return compress_plain_tool_output(
            content=text,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            max_chars=max_chars,
            header="[TRUNCATED STRUCTURED TOOL OUTPUT]",
        )

    compact = _compact_json(obj)
    rendered = _render_json(compact)
    lines = [
        "[COMPRESSED JSON TOOL OUTPUT]",
        f"tool_name: {tool_name or 'unknown'}",
        f"tool_call_id: {tool_call_id or 'unknown'}",
        "content_type: json_report",
        "compression_strategy: preserve_keys_sample_large_arrays",
        f"original_characters: {len(text)}",
        "json_summary:",
        rendered,
    ]
    out = "\n".join(lines)

    if len(out) <= max_chars:
        return out

    # Reduce list samples/string lengths progressively before falling back to a
    # final character fit.  Keep keys visible rather than flattening to unknowns.
    for max_items, max_string_chars, max_depth in ((10, 1000, 7), (5, 700, 6), (3, 500, 5)):
        compact = _compact_json(obj, max_items=max_items, max_string_chars=max_string_chars, max_depth=max_depth)
        rendered = _render_json(compact)
        out = "\n".join(lines[:-1] + [rendered])
        if len(out) <= max_chars:
            return out

    return _fit_chars(out, max_chars)


def compress_web_tool_output(
    *,
    content: Any,
    tool_name: str = "",
    tool_call_id: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    first_lines: int = DEFAULT_FIRST_LINES,
    tail_lines: int = DEFAULT_TAIL_LINES,
) -> str:
    """Compress web/browser output by stripping HTML and retaining URL/title cues."""

    text = _tool_content_to_text(content)
    obj = _maybe_json_loads(text)
    html_or_text = _extract_web_text_from_json(obj) if obj is not None else text
    plain = _html_to_text(html_or_text)
    raw_lines = [line.rstrip() for line in plain.splitlines() if line.strip()]
    urls = _extract_paths_from_text(text, limit=12)
    title = _extract_title(html_or_text) or _extract_json_scalar(obj, ("title", "page_title")) or "unknown"

    lines = [
        "[COMPRESSED WEB TOOL OUTPUT]",
        f"tool_name: {tool_name or 'unknown'}",
        f"tool_call_id: {tool_call_id or 'unknown'}",
        "content_type: web",
        f"original_characters: {len(text)}",
        f"title: {_clean_line(str(title), 300)}",
    ]
    if urls:
        lines.append("urls:")
        lines.extend(f"- {url}" for url in urls)

    lines.append("")
    lines.append(f"first_{first_lines}_text_lines:")
    if raw_lines:
        lines.extend(f"{idx:02d}: {_clean_line(line, 500)}" for idx, line in enumerate(raw_lines[:first_lines], start=1))
    else:
        lines.append("(no readable text detected)")

    if len(raw_lines) > first_lines and tail_lines > 0:
        lines.append("")
        lines.append(f"last_{tail_lines}_text_lines:")
        tail_start = max(first_lines + 1, len(raw_lines) - tail_lines + 1)
        lines.extend(f"{idx:02d}: {_clean_line(line, 500)}" for idx, line in enumerate(raw_lines[-tail_lines:], start=tail_start))

    return _fit_chars("\n".join(lines), max_chars)


def compress_plain_tool_output(
    *,
    content: Any,
    tool_name: str = "",
    tool_call_id: str | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    first_lines: int = DEFAULT_FIRST_LINES,
    tail_lines: int = DEFAULT_TAIL_LINES,
    header: str = "[TRUNCATED TOOL OUTPUT]",
) -> str:
    """Fallback compression: retain head and tail without schema extraction."""

    text = _tool_content_to_text(content)
    raw_lines = text.splitlines()
    non_empty_lines = [line.rstrip() for line in raw_lines if line.strip()]
    source_lines = non_empty_lines or raw_lines

    lines = [
        header,
        f"tool_name: {tool_name or 'unknown'}",
        f"tool_call_id: {tool_call_id or 'unknown'}",
        "content_type: text",
        "compression_strategy: head_tail_pass_through",
        f"original_characters: {len(text)}",
        f"original_lines: {len(raw_lines)}",
        "note: No log schema extraction was applied.",
        "",
        f"first_{first_lines}_lines:",
    ]

    if source_lines:
        lines.extend(f"{idx:02d}: {_clean_line(line, 700)}" for idx, line in enumerate(source_lines[:first_lines], start=1))
    else:
        lines.append("(empty output)")

    if len(source_lines) > first_lines and tail_lines > 0:
        lines.append("")
        lines.append(f"last_{tail_lines}_lines:")
        tail_start = max(first_lines + 1, len(source_lines) - tail_lines + 1)
        lines.extend(f"{idx:02d}: {_clean_line(line, 700)}" for idx, line in enumerate(source_lines[-tail_lines:], start=tail_start))

    return _fit_chars("\n".join(lines), max_chars)


def _tool_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content or "")


def _maybe_json_loads(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except Exception:
        return None


def _extract_meta_content_type(obj: Any) -> ToolContentType | None:
    if not isinstance(obj, dict):
        return None
    meta = obj.get("_meta")
    if not isinstance(meta, dict):
        return None
    raw = str(meta.get("content_type") or meta.get("tool_content_type") or "").strip().lower()
    if not raw:
        return None
    if raw in {"stb_log", "raw_log", "log", "log_slice", "filtered_log"}:
        return ToolContentType.STB_LOG
    if raw in {"json_report", "metadata", "report", "json", "structured_json"}:
        return ToolContentType.JSON_REPORT
    if raw in {"html", "web", "web_page", "browser"}:
        return ToolContentType.WEB
    if raw in {"text", "plain_text"}:
        return ToolContentType.TEXT
    return None


def _is_raw_stb_log_tool_name(name: str) -> bool:
    return any(
        name == hint
        or name.endswith(f"_{hint}")
        or name.endswith(f"__{hint}")
        or f".{hint}" in name
        for hint in _RAW_STB_LOG_TOOL_HINTS
    )


def _looks_like_html(text: str) -> bool:
    return bool(_HTML_RE.search(text[:20_000]))


def _looks_like_stb_log_text(text: str) -> bool:
    lower = text[:50_000].lower()
    if _RECEIVER_RE.search(text) and any(hint in lower for hint in _LOG_TEXT_HINTS):
        return True
    lines = [line for line in text.splitlines()[:200] if line.strip()]
    timestamp_hits = sum(1 for line in lines if _TIMESTAMP_LINE_RE.search(line))
    issue_hits = sum(1 for line in lines if any(term in line.lower() for term in ("stb", "sgs", "tuner", "playback", "channel", "guide", "error")))
    return timestamp_hits >= 3 and issue_hits >= 1


def _has_log_signal(text: str, *, obj: Any | None, extracted_log_text: str) -> bool:
    """Return True only when payload has raw-log semantics, not generic metadata.

    Structured JSON reports often contain receiver IDs, S3 keys, or field names
    like ``results`` without containing raw log lines.  Those signals alone are
    not sufficient to use the STB-log extractor; otherwise validation reports
    and metadata payloads get flattened into meaningless log-shaped records.
    """

    if isinstance(obj, dict):
        meta = obj.get("_meta")
        if isinstance(meta, dict):
            hinted = _extract_meta_content_type(obj)
            if hinted is ToolContentType.STB_LOG:
                return True
            if hinted is not None:
                return False

        keys = {str(k).lower() for k in obj.keys()}
        raw_text_keys = {"content", "filtered_log", "log", "raw_log", "text"}
        if keys & raw_text_keys:
            return _looks_like_stb_log_text(extracted_log_text) or bool(
                _RECEIVER_RE.search(text) and any(hint in extracted_log_text.lower() for hint in _LOG_TEXT_HINTS)
            )

        # Metadata-only JSON with IDs, paths, or result arrays is a report, not
        # raw log content.  Require actual timestamp/log-line structure.
        return _looks_like_stb_log_text(extracted_log_text)

    return _looks_like_stb_log_text(extracted_log_text)


def _extract_log_text_from_json(obj: Any) -> str:
    if isinstance(obj, dict):
        for key in ("content", "filtered_log", "log", "raw_log", "text"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value
        if isinstance(obj.get("results"), list):
            return "\n".join(
                str(item.get("context") or item.get("match_line") or item)
                for item in obj["results"][:100]
                if item is not None
            )
        if isinstance(obj.get("records"), list):
            return "\n".join(str(item) for item in obj["records"][:100])
    return _render_json(obj) if obj is not None else ""


def _extract_log_metadata(obj: Any, text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key in (
            "receiver_id",
            "date",
            "filename",
            "log_type",
            "source",
            "total_lines",
            "returned_line_count",
            "returned_start_line",
            "returned_end_line",
            "truncated",
            "content_truncated",
            "hard_limit_truncated",
            "next_start_line",
        ):
            value = obj.get(key)
            if value not in (None, "", [], {}):
                metadata[key] = value
    if not metadata:
        receivers = sorted(set(_RECEIVER_RE.findall(text)))
        if receivers:
            metadata["receiver_ids"] = receivers[:20]
    return metadata


def _extract_log_record_previews(obj: Any, limit: int = 20) -> list[str]:
    records: Iterable[Any] = []
    if isinstance(obj, dict):
        for key in ("results", "records", "matches"):
            if isinstance(obj.get(key), list):
                records = obj[key]
                break
    elif isinstance(obj, list):
        records = obj

    previews: list[str] = []
    for idx, record in enumerate(records, start=1):
        if idx > limit:
            break
        if isinstance(record, dict):
            pieces = []
            for key in ("file", "filename", "log_type", "line_number", "match_line", "context", "receiver_id"):
                value = record.get(key)
                if value not in (None, "", [], {}):
                    pieces.append(f"{key}={_short_scalar(value, 220)}")
            if pieces:
                previews.append(f"- record {idx}: " + "; ".join(pieces))
        else:
            previews.append(f"- record {idx}: {_clean_line(str(record), 500)}")
    return previews


def _compact_json(
    value: Any,
    *,
    max_items: int = JSON_MAX_ITEMS,
    max_keys: int = JSON_MAX_KEYS,
    max_depth: int = JSON_MAX_DEPTH,
    max_string_chars: int = JSON_MAX_STRING_CHARS,
    _depth: int = 0,
) -> Any:
    if _depth >= max_depth:
        return _summarize_deep_value(value, max_string_chars=max_string_chars)

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        keys = list(value.keys())
        ordered_keys = [key for key in _PROTECTED_TOP_LEVEL_KEYS if key in value]
        ordered_keys.extend(key for key in keys if key not in set(ordered_keys))

        kept = 0
        for key in ordered_keys:
            if kept >= max_keys:
                break
            out[str(key)] = _compact_json(
                value[key],
                max_items=max_items,
                max_keys=max_keys,
                max_depth=max_depth,
                max_string_chars=max_string_chars,
                _depth=_depth + 1,
            )
            kept += 1
        if len(keys) > kept:
            out["_omitted_keys"] = len(keys) - kept
        return out

    if isinstance(value, list):
        sampled = [
            _compact_json(
                item,
                max_items=max_items,
                max_keys=max_keys,
                max_depth=max_depth,
                max_string_chars=max_string_chars,
                _depth=_depth + 1,
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            sampled.append({"_omitted_items": len(value) - max_items})
        return sampled

    if isinstance(value, tuple):
        return _compact_json(list(value), max_items=max_items, max_keys=max_keys, max_depth=max_depth, max_string_chars=max_string_chars, _depth=_depth)

    if isinstance(value, str):
        return _truncate_long_string(value, max_string_chars)

    if value is None or isinstance(value, (int, float, bool)):
        return value

    return _truncate_long_string(str(value), max_string_chars)


def _summarize_deep_value(value: Any, *, max_string_chars: int) -> Any:
    if isinstance(value, dict):
        return {"_type": "dict", "keys": list(value.keys())[:20], "_omitted_keys": max(0, len(value) - 20)}
    if isinstance(value, list):
        return {"_type": "list", "items": len(value)}
    if isinstance(value, str):
        return _truncate_long_string(value, max_string_chars)
    return value if value is None or isinstance(value, (int, float, bool)) else str(type(value).__name__)


def _render_json(value: Any) -> str:
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _extract_json_scalar(obj: Any, keys: tuple[str, ...]) -> Any | None:
    if not isinstance(obj, dict):
        return None
    lowered = {str(k).lower(): v for k, v in obj.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if isinstance(value, (str, int, float, bool)) and value not in ("", None):
            return value
    return None


def _extract_web_text_from_json(obj: Any) -> str:
    if isinstance(obj, dict):
        for key in ("html", "content", "text", "body", "markdown"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return _render_json(obj) if obj is not None else ""


def _html_to_text(raw: str) -> str:
    without_scripts = _SCRIPT_STYLE_RE.sub(" ", raw)
    with_breaks = re.sub(r"<\s*(?:br|/p|/div|/tr|/li|/h[1-6])\s*/?>", "\n", without_scripts, flags=re.IGNORECASE)
    stripped = _TAG_RE.sub(" ", with_breaks)
    unescaped = html.unescape(stripped)
    lines = [_SPACE_RE.sub(" ", line).strip() for line in unescaped.splitlines()]
    return "\n".join(line for line in lines if line)


def _extract_title(raw: str) -> str | None:
    match = re.search(r"<\s*title[^>]*>([\s\S]*?)<\s*/\s*title\s*>", raw, flags=re.IGNORECASE)
    if match:
        return _clean_line(html.unescape(match.group(1)), 300)
    return None


def _extract_paths_from_text(text: str, limit: int = 10) -> list[str]:
    seen: list[str] = []
    for match in _PATH_RE.findall(text):
        clean = match.rstrip(".,);]")
        if clean not in seen:
            seen.append(clean)
        if len(seen) >= limit:
            break
    return seen


def _extract_error_lines(text: str, limit: int = 8) -> list[str]:
    error_terms = ("error", "exception", "traceback", "failed", "failure", "denied", "timeout")
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(term in stripped.lower() for term in error_terms):
            lines.append(_clean_line(stripped, 500))
        if len(lines) >= limit:
            break
    return lines


def _truncate_long_string(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars < 100:
        return text[:max_chars]
    head = max_chars // 2
    tail = max_chars - head - 40
    return f"{text[:head]}...<truncated {len(text) - max_chars} chars>...{text[-tail:]}"


def _fit_chars(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = f"\n\n[... compressed tool output truncated to {max_chars} characters ...]"
    if max_chars <= len(marker) + 20:
        return text[:max_chars]
    return text[: max_chars - len(marker)] + marker


def _clean_line(line: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(line)).strip()
    return _truncate_long_string(cleaned, max_chars)


def _short_scalar(value: Any, max_chars: int) -> str:
    if isinstance(value, (dict, list, tuple)):
        value = _render_json(value)
    return _clean_line(str(value), max_chars)
