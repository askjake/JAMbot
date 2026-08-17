"""Regression tests for content-aware ToolMessage compression routing."""

from __future__ import annotations

import json

import pytest

from app.message.tool_message_compressor import (
    ToolContentType,
    classify_tool_content,
    compress_tool_message_content,
)


def _large_validation_report() -> str:
    payload = {
        "_meta": {
            "content_type": "json_report",
            "tool_name": "validate_mcp_token_efficiency",
            "compression_hint": "preserve_validation_keys_drop_large_arrays",
        },
        "validation_mode": "token_efficiency",
        "heavy_tools_blocked_by_design": True,
        "inline_chars_est": 39277,
        "calls": [
            {
                "tool": "validate_mcp_token_efficiency",
                "status": "passed",
                "inline_chars_est": 2048 + idx,
                "notes": "metadata report, not an STB log",
            }
            for idx in range(80)
        ],
        "full_report_s3": "s3://bucket/reports/token_efficiency_validation.json",
        "token_safety_note": "Preserve these validation keys for client-side diagnosis.",
    }
    return json.dumps(payload, indent=2)


def test_validate_mcp_token_efficiency_routes_to_json_not_log_schema() -> None:
    """Metadata validation JSON must not be force-fit into STB log records."""

    result = compress_tool_message_content(
        tool_name="validate_mcp_token_efficiency",
        content=_large_validation_report(),
        tool_call_id="call-json",
        max_chars=3500,
    )

    assert result.content_type is ToolContentType.JSON_REPORT
    assert "[COMPRESSED JSON TOOL OUTPUT]" in result.content
    assert "heavy_tools_blocked_by_design" in result.content
    assert "inline_chars_est" in result.content
    assert "full_report_s3" in result.content
    assert "[COMPRESSED LOG TOOL OUTPUT]" not in result.content
    assert "receiver_id: unknown" not in result.content
    assert "log_name: unknown" not in result.content


def test_stb_log_compressor_falls_back_for_non_log_json() -> None:
    """The STB-log strategy should fall back when content is metadata JSON."""

    from app.message.tool_message_compressor import compress_stb_log_tool_output

    out = compress_stb_log_tool_output(
        content=_large_validation_report(),
        tool_call_id="call-legacy",
        tool_name="validate_mcp_token_efficiency",
        max_chars=2500,
    )

    assert "[TRUNCATED NON-LOG TOOL OUTPUT]" in out
    assert "No log schema extraction was applied" in out
    assert "heavy_tools_blocked_by_design" in out
    assert "[COMPRESSED LOG TOOL OUTPUT]" not in out
    assert "receiver_id: unknown" not in out


def test_raw_read_log_still_uses_stb_log_extractor() -> None:
    """Raw log tools keep the existing STB-log extraction behavior."""

    raw_log = {
        "_meta": {"content_type": "stb_log", "tool_name": "read_log"},
        "receiver_id": "R1956409151",
        "filename": "qt_gui.log",
        "content": "\n".join(
            f"2026-07-06T10:{idx:02d}:00 R1956409151 qt_gui ERROR Guide 1031 playback channel issue"
            for idx in range(70)
        ),
    }

    result = compress_tool_message_content(
        tool_name="read_log",
        content=json.dumps(raw_log),
        tool_call_id="call-log",
        max_chars=5000,
    )

    assert result.content_type is ToolContentType.STB_LOG
    assert "[COMPRESSED LOG TOOL OUTPUT]" in result.content
    assert "R1956409151" in result.content
    assert "qt_gui.log" in result.content
    assert "error_summary" in result.content


def test_unknown_tool_uses_plain_head_tail_not_log_extractor() -> None:
    """Unknown non-JSON, non-log payloads should be simply truncated."""

    text = "\n".join(f"plain diagnostic line {idx}" for idx in range(300))
    result = compress_tool_message_content(
        tool_name="custom_metadata_dump",
        content=text,
        tool_call_id="call-text",
        max_chars=1800,
        first_lines=8,
        tail_lines=5,
    )

    assert result.content_type is ToolContentType.TEXT
    assert "[TRUNCATED TOOL OUTPUT]" in result.content
    assert "No log schema extraction was applied" in result.content
    assert "plain diagnostic line 0" in result.content
    assert "plain diagnostic line 299" in result.content
    assert "[COMPRESSED LOG TOOL OUTPUT]" not in result.content


def test_progressive_memory_classifies_metadata_as_json_report() -> None:
    """Old validation reports degrade as JSON reports, not log payloads."""

    pytest.importorskip("langchain_core.messages")
    from app.tools.progressive_tool_memory import generate_progressive_levels

    progressive = generate_progressive_levels(
        "validate_mcp_token_efficiency",
        _large_validation_report(),
        tool_call_id="call-progressive",
        turn_created=2,
    )

    assert progressive.category == "json_report"
    assert "structured JSON report" in progressive.one_liner
    assert "[COMPRESSED JSON TOOL OUTPUT]" in progressive.key_facts
    assert "[COMPRESSED LOG TOOL OUTPUT]" not in progressive.key_facts


def test_token_efficiency_adapter_never_codebook_encodes_tool_messages() -> None:
    """Codebook encoding is limited to conversational messages, not ToolMessages."""

    langchain_messages = pytest.importorskip("langchain_core.messages")
    from app.message.token_efficiency_adapter import apply_token_efficiency_layer, reset_session

    AIMessage = langchain_messages.AIMessage
    HumanMessage = langchain_messages.HumanMessage
    ToolMessage = langchain_messages.ToolMessage

    reset_session()
    repeated_phrase = "Superset Okta dashboard token efficiency validation report"
    tool_payload = json.dumps(
        {
            "path": f"/tmp/{repeated_phrase.replace(' ', '_')}.json",
            "message": repeated_phrase,
        }
    )
    messages = [
        HumanMessage(content=f"Earlier context: {repeated_phrase}. {repeated_phrase}. {repeated_phrase}."),
        AIMessage(content=f"Decision: keep investigating {repeated_phrase}. " * 30),
        ToolMessage(content=tool_payload, tool_call_id="tool-codebook", name="validate_mcp_token_efficiency"),
        HumanMessage(content="Current question must remain verbatim."),
    ]

    out = apply_token_efficiency_layer(
        messages,
        min_messages_to_activate=0,
        min_tokens_to_activate=0,
    )

    tool_messages = [msg for msg in out if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == tool_payload
    assert classify_tool_content("validate_mcp_token_efficiency", tool_payload) is ToolContentType.JSON_REPORT
