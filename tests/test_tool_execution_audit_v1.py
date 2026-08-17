"""Invariant tests for tool_execution_audit.v1.

These tests are the enforcement mechanism for the Phase C redaction contract.
Canary values are deliberately distinctive so a leak is unambiguous; none of
them is a real credential.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.audited_tool_node import AuditedToolNode
from app.agent.tool_execution_audit import _FORBIDDEN_KEY_RE  # noqa: PLC2701 - guard under test
from app.agent.tool_execution_audit import (
    AUTHORIZATION_FLAG_KEYS,
    DECISION_ALLOWED,
    DECISION_BLOCKED,
    DECISION_FAILED,
    DECISION_REWRITTEN,
    DEFAULT_QUERY_LIMIT,
    MAX_QUERY_LIMIT,
    RESULT_BLOCKED_HEAVY_UNAUTHORIZED,
    RESULT_BLOCKED_PERSISTENCE_UNAUTHORIZED,
    RESULT_BLOCKED_UNBOUND_TOOL,
    RESULT_EXECUTED,
    RESULT_FAILED_EXECUTOR_EXCEPTION,
    RESULT_REWRITTEN_LIMIT_BOUNDED,
    RESULT_REWRITTEN_PERSIST_FALSE,
    TOOL_EXECUTION_AUDIT_SCHEMA,
    AuditRequestState,
    ToolExecutionAuditSink,
    bind_audit_request_state,
    build_event,
    digest_thread_id,
    query_events,
    sanitize_event,
    validate_event,
)

# --------------------------------------------------------------------------
# Canaries
# --------------------------------------------------------------------------
CANARY_ARG_VALUE = "scene-6f21a9d46f21a9d46f21a9d4"
CANARY_RESULT_BODY = "CANARY-TOOL-RESULT-BODY-b8c07e15"
CANARY_TOKEN = "CANARY-AUTHORIZATION-TOKEN-VALUE-4e19"
CANARY_EXCEPTION_TEXT = "CANARY-EXCEPTION-MESSAGE-WITH-https://chat.example/secret-3a7f"
CANARY_RAW_THREAD_ID = "CANARY-RAW-THREAD-ID-1d9e7c40"
CANARY_WEBHOOK = "https://chat.example.invalid/v1/spaces/CANARY-WEBHOOK-9911"

ALL_CANARIES = (
    CANARY_ARG_VALUE,
    CANARY_RESULT_BODY,
    CANARY_TOKEN,
    CANARY_EXCEPTION_TEXT,
    CANARY_RAW_THREAD_ID,
    CANARY_WEBHOOK,
)


class _CanaryError(RuntimeError):
    pass


class FakeTool:
    """Minimal BaseTool-compatible stand-in accepting a tool_call dict."""

    def __init__(self, name: str, *, result: str = "ok", raises: BaseException | None = None):
        self.name = name
        self._result = result
        self._raises = raises
        self.seen_args: dict | None = None

    async def ainvoke(self, tool_call, config=None):
        self.seen_args = dict(tool_call.get("args") or {})
        if self._raises is not None:
            raise self._raises
        return ToolMessage(content=self._result, tool_call_id=tool_call.get("id") or "", name=self.name)


class FailingSink(ToolExecutionAuditSink):
    """A sink that raises on every write, to prove failure isolation."""

    def record(self, event):  # type: ignore[override]
        raise OSError("simulated audit sink failure")


@pytest.fixture()
def sink(tmp_path: Path) -> ToolExecutionAuditSink:
    return ToolExecutionAuditSink(tmp_path / "audit", max_bytes=4096, max_files=3)


def _state(tool_name: str, args: dict, call_id: str = "call-1") -> dict:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "args": dict(args), "id": call_id, "type": "tool_call"}],
            )
        ]
    }


def _run(node: AuditedToolNode, state: dict) -> list:
    return asyncio.run(node.ainvoke(state))["messages"]


def _bind(flags: dict | None = None, *, thread_id: str = CANARY_RAW_THREAD_ID, bound=()):
    request_state = AuditRequestState(request_id="req-canary-0000000001")
    request_state.record_binding(
        bound_tool_names=bound,
        authorization_flags=flags or {},
        thread_id=thread_id,
    )
    bind_audit_request_state(request_state)
    return request_state


def _events(sink: ToolExecutionAuditSink) -> list[dict]:
    return list(sink.iter_events(max_records=200))


def _raw_audit_text(sink: ToolExecutionAuditSink) -> str:
    return "".join(path.read_text(encoding="utf-8") for path in sink.files_newest_first())


# --------------------------------------------------------------------------
# 1-2. No argument values, no tool-result bodies
# --------------------------------------------------------------------------
def test_audit_contains_no_argument_values_and_no_result_body(sink):
    _bind({"heavy_tools_authorized": False}, bound=["query_incident_scene"])
    tool = FakeTool("query_incident_scene", result=CANARY_RESULT_BODY)
    node = AuditedToolNode([tool], sink=sink)
    messages = _run(node, _state("query_incident_scene", {"scene_id": CANARY_ARG_VALUE, "limit": 5}))

    assert len(messages) == 1
    assert CANARY_RESULT_BODY in str(messages[0].content)

    text = _raw_audit_text(sink)
    assert CANARY_ARG_VALUE not in text
    assert CANARY_RESULT_BODY not in text
    event = _events(sink)[0]
    assert event["argument_field_names"] == ["limit", "scene_id"]
    assert "result" not in event and "args" not in event


# --------------------------------------------------------------------------
# 3-4. Authorization token fields dropped, values never hashed
# --------------------------------------------------------------------------
def test_authorization_token_field_dropped_and_never_hashed(sink):
    _bind({"heavy_tools_authorized": True}, bound=["build_incident_scene"])
    tool = FakeTool("build_incident_scene", result="scene-built")
    node = AuditedToolNode([tool], sink=sink)
    _run(
        node,
        _state("build_incident_scene", {"capsule_id": "cap-1", "heavy_auth_token": CANARY_TOKEN}),
    )

    text = _raw_audit_text(sink)
    assert CANARY_TOKEN not in text
    assert "heavy_auth_token" not in text

    # A hash of a secret is still secret-derived material: prove no digest of the
    # canary token in any common form was persisted.
    encoded = CANARY_TOKEN.encode()
    for digest in (
        hashlib.sha256(encoded).hexdigest(),
        hashlib.sha1(encoded).hexdigest(),
        hashlib.md5(encoded).hexdigest(),
        hashlib.sha512(encoded).hexdigest(),
    ):
        assert digest not in text
        assert digest[:12] not in text
    for fragment in (CANARY_TOKEN[:8], CANARY_TOKEN[-8:]):
        assert fragment not in text

    event = _events(sink)[0]
    assert "heavy_auth_token" not in event["argument_field_names"]
    assert "heavy_auth_token" not in event["rewritten_fields"]
    # The server-side switch is applied but the token never reaches the executor.
    assert tool.seen_args is not None and "heavy_auth_token" not in tool.seen_args


# --------------------------------------------------------------------------
# 5-6. Exception messages absent; only safe class names retained
# --------------------------------------------------------------------------
def test_executor_exception_records_class_name_only(sink):
    _bind({}, bound=["query_incident_scene"])
    tool = FakeTool("query_incident_scene", raises=_CanaryError(CANARY_EXCEPTION_TEXT))
    node = AuditedToolNode([tool], sink=sink)
    messages = _run(node, _state("query_incident_scene", {"scene_id": "scene-0123456789abcdef01234567"}))

    assert len(messages) == 1
    text = _raw_audit_text(sink)
    assert CANARY_EXCEPTION_TEXT not in text
    assert "chat.example" not in text

    event = _events(sink)[0]
    assert event["error_type"] == "_CanaryError"
    assert event["decision"] == DECISION_FAILED
    assert event["result_code"] == RESULT_FAILED_EXECUTOR_EXCEPTION
    assert event["executed"] is False


def test_error_type_rejects_message_like_values():
    event = build_event(error_type="ValueError: leaked " + CANARY_WEBHOOK)
    assert event["error_type"] == "UnsafeErrorTypeRedacted"
    assert CANARY_WEBHOOK not in json.dumps(event)


# --------------------------------------------------------------------------
# 7-9. Block codes
# --------------------------------------------------------------------------
def test_unbound_tool_call_is_blocked_and_audited(sink):
    _bind({}, bound=["list_dates"])
    tool = FakeTool("search_logs", result="should-not-run")
    node = AuditedToolNode([tool], sink=sink)
    messages = _run(node, _state("search_logs", {"pattern": CANARY_ARG_VALUE}))

    event = _events(sink)[0]
    assert event["result_code"] == RESULT_BLOCKED_UNBOUND_TOOL
    assert event["decision"] == DECISION_BLOCKED
    assert event["executed"] is False
    assert event["tool_was_bound"] is False
    assert tool.seen_args is None
    assert len(messages) == 1
    assert CANARY_ARG_VALUE not in _raw_audit_text(sink)


def test_unauthorized_heavy_call_is_blocked_with_heavy_code(sink):
    _bind({"heavy_tools_authorized": False}, bound=["build_incident_scene"])
    tool = FakeTool("build_incident_scene")
    node = AuditedToolNode([tool], sink=sink)
    _run(node, _state("build_incident_scene", {"capsule_id": "cap-1"}))

    event = _events(sink)[0]
    assert event["result_code"] == RESULT_BLOCKED_HEAVY_UNAUTHORIZED
    assert event["executed"] is False
    assert tool.seen_args is None
    assert event["authorization_flags"]["heavy_tools_authorized"] is False


def test_unauthorized_persistence_is_blocked_or_safely_rewritten(sink):
    # Always-persist tool without persistence authorization must be blocked.
    _bind({"heavy_tools_authorized": True}, bound=["build_complete_log_capsule"])
    blocked_tool = FakeTool("build_complete_log_capsule")
    node = AuditedToolNode([blocked_tool], sink=sink)
    _run(node, _state("build_complete_log_capsule", {"receiver_id": "R1234567"}, call_id="c-block"))
    blocked_event = _events(sink)[0]
    assert blocked_event["result_code"] == RESULT_BLOCKED_PERSISTENCE_UNAUTHORIZED
    assert blocked_event["executed"] is False
    assert blocked_tool.seen_args is None

    # A heavy read with persist=True is downgraded rather than blocked.
    _bind({"heavy_tools_authorized": True}, bound=["build_log_capsule"])
    rewrite_tool = FakeTool("build_log_capsule")
    node = AuditedToolNode([rewrite_tool], sink=sink)
    _run(node, _state("build_log_capsule", {"receiver_id": "R1234567", "persist": True}, call_id="c-rewrite"))
    rewrite_event = _events(sink)[0]
    assert rewrite_event["result_code"] == RESULT_REWRITTEN_PERSIST_FALSE
    assert rewrite_event["decision"] == DECISION_REWRITTEN
    assert rewrite_event["executed"] is True
    assert rewrite_tool.seen_args is not None
    assert rewrite_tool.seen_args["persist"] is False


# --------------------------------------------------------------------------
# 10-11. Rewrites and bounds recorded by name only
# --------------------------------------------------------------------------
def test_rewritten_and_bounded_fields_are_names_only(sink):
    _bind({}, bound=["query_incident_scene"])
    tool = FakeTool("query_incident_scene")
    node = AuditedToolNode([tool], sink=sink)
    _run(node, _state("query_incident_scene", {"scene_id": CANARY_ARG_VALUE, "limit": 999999}))

    event = _events(sink)[0]
    assert event["bounded_fields"] == ["limit"]
    assert event["result_code"] == RESULT_REWRITTEN_LIMIT_BOUNDED
    assert event["decision"] == DECISION_REWRITTEN
    assert tool.seen_args is not None and tool.seen_args["limit"] == 500
    text = _raw_audit_text(sink)
    assert "999999" not in text
    assert "500" not in json.dumps(event["bounded_fields"])
    assert CANARY_ARG_VALUE not in text


# --------------------------------------------------------------------------
# 12-14. Execution flags and paired results
# --------------------------------------------------------------------------
def test_allowed_execution_and_blocked_execution_flags(sink):
    _bind({}, bound=["list_dates"])
    node = AuditedToolNode([FakeTool("list_dates", result="dates")], sink=sink)
    _run(node, _state("list_dates", {"receiver_id": "R1234567"}))
    allowed = _events(sink)[0]
    assert allowed["executed"] is True
    assert allowed["decision"] == DECISION_ALLOWED
    assert allowed["result_code"] == RESULT_EXECUTED
    assert allowed["duration_ms"] >= 0

    _bind({}, bound=["list_dates"])
    node = AuditedToolNode([FakeTool("other_tool")], sink=sink)
    _run(node, _state("other_tool", {}))
    blocked = _events(sink)[0]
    assert blocked["executed"] is False
    assert blocked["decision"] == DECISION_BLOCKED


def test_every_emitted_call_receives_one_paired_tool_message(sink):
    _bind({}, bound=["list_dates", "build_incident_scene"])
    tools = [FakeTool("list_dates", result="ok"), FakeTool("build_incident_scene")]
    node = AuditedToolNode(tools, sink=sink)
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "list_dates", "args": {"receiver_id": "R1234567"}, "id": "c1", "type": "tool_call"},
                    {"name": "build_incident_scene", "args": {"capsule_id": "cap"}, "id": "c2", "type": "tool_call"},
                    {"name": "not_bound_tool", "args": {}, "id": "c3", "type": "tool_call"},
                ],
            )
        ]
    }
    messages = asyncio.run(node.ainvoke(state))["messages"]
    assert len(messages) == 3
    assert [m.tool_call_id for m in messages] == ["c1", "c2", "c3"]

    events = _events(sink)
    assert len(events) == 3
    assert all(event["paired_tool_result"] is True for event in events)
    assert {event["tool_call_id"] for event in events} == {"c1", "c2", "c3"}


# --------------------------------------------------------------------------
# 15. Request correlation
# --------------------------------------------------------------------------
def test_request_id_correlates_binding_gate_and_tool_message(sink):
    request_state = _bind({}, bound=["list_dates"])
    node = AuditedToolNode([FakeTool("list_dates", result="ok")], sink=sink)
    messages = _run(node, _state("list_dates", {"receiver_id": "R1234567"}, call_id="c-corr"))

    event = _events(sink)[0]
    assert event["request_id"] == request_state.request_id
    assert event["binding_signature"] == request_state.binding_signature
    assert event["binding_signature"].startswith("bind-0001-")
    assert event["binding_known"] is True
    assert event["thread_id_digest"] == digest_thread_id(CANARY_RAW_THREAD_ID)
    assert event["tool_call_id"] == messages[0].tool_call_id == "c-corr"


# --------------------------------------------------------------------------
# 16-17. Audit failure isolation
# --------------------------------------------------------------------------
def test_audit_failure_does_not_execute_a_blocked_call(tmp_path):
    _bind({}, bound=["list_dates"])
    tool = FakeTool("build_incident_scene")
    node = AuditedToolNode([tool], sink=FailingSink(tmp_path / "fail"))
    messages = _run(node, _state("build_incident_scene", {"capsule_id": "cap"}))
    assert tool.seen_args is None
    assert len(messages) == 1
    assert "TOOL_NOT_IN_LAST_BINDING" in str(messages[0].content)


def test_audit_failure_does_not_crash_an_allowed_call(tmp_path):
    _bind({}, bound=["list_dates"])
    tool = FakeTool("list_dates", result="ok")
    node = AuditedToolNode([tool], sink=FailingSink(tmp_path / "fail"))
    messages = _run(node, _state("list_dates", {"receiver_id": "R1234567"}))
    assert tool.seen_args == {"receiver_id": "R1234567"}
    assert str(messages[0].content) == "ok"


def test_sink_write_failure_is_counted_and_never_raises(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("not-a-directory")
    sink = ToolExecutionAuditSink(blocker)
    assert sink.record(build_event(tool_name="x")) is False
    assert sink.write_failure_count == 1


# --------------------------------------------------------------------------
# 18-20. Permissions, rotation, retention
# --------------------------------------------------------------------------
def test_file_permissions_are_restricted(sink):
    sink.record(build_event(tool_name="list_dates"))
    assert stat.S_IMODE(os.stat(sink.active_path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(sink.directory).st_mode) == 0o700


def test_rotation_and_retention_are_bounded(tmp_path):
    sink = ToolExecutionAuditSink(tmp_path / "audit", max_bytes=1024, max_files=2)
    for _ in range(200):
        sink.record(build_event(tool_name="list_dates"))
    rotated = list(sink.directory.glob("tool_execution_audit.*.jsonl"))
    assert len(rotated) <= 2
    assert sink.active_path.stat().st_size <= 1024 + 4096
    assert all(stat.S_IMODE(os.stat(path).st_mode) == 0o600 for path in rotated)
    assert sink.write_failure_count == 0


# --------------------------------------------------------------------------
# 21-22. Query bounds and validation
# --------------------------------------------------------------------------
def test_query_limits_are_enforced(sink):
    for _ in range(30):
        sink.record(build_event(tool_name="list_dates"))
    assert len(query_events(sink=sink, limit=5)) == 5
    assert len(query_events(sink=sink, limit=10**6)) <= MAX_QUERY_LIMIT
    # A falsy limit falls back to the bounded default, never to "unlimited".
    assert len(query_events(sink=sink, limit=0)) <= DEFAULT_QUERY_LIMIT
    assert len(query_events(sink=sink, limit=-5)) == 1
    assert query_events(sink=sink, tool_name="nonexistent-tool") == []


def test_malformed_records_are_rejected_by_validation(sink):
    sink.record(build_event(tool_name="list_dates"))
    with sink.active_path.open("a", encoding="utf-8") as handle:
        handle.write("{not json at all\n")
    records = _events(sink)
    malformed = [record for record in records if record.get("_malformed")]
    assert malformed, "malformed line should surface as a malformed record"
    assert validate_event(malformed[0]), "malformed record must fail validation"

    assert validate_event({"schema_version": "wrong"}), "wrong schema must fail"
    assert validate_event({**build_event(tool_name="x"), "decision": "MADE_UP"})
    blocked_claiming_execution = {**build_event(tool_name="x", decision=DECISION_BLOCKED), "executed": True}
    assert "blocked event claims execution" in validate_event(blocked_claiming_execution)
    assert validate_event(build_event(tool_name="x")) == []


def test_sanitize_event_drops_forbidden_and_unknown_keys():
    dirty = {
        **build_event(tool_name="list_dates"),
        "heavy_auth_token": CANARY_TOKEN,
        "token_hash": hashlib.sha256(CANARY_TOKEN.encode()).hexdigest(),
        "prompt_text": "raw prompt",
        "assistant_text": "assistant prose",
        "tool_result_body": CANARY_RESULT_BODY,
        "webhook_url": CANARY_WEBHOOK,
        "thread_id": CANARY_RAW_THREAD_ID,
        "args": {"a": CANARY_ARG_VALUE},
    }
    clean = sanitize_event(dirty)
    serialized = json.dumps(clean)
    for canary in ALL_CANARIES:
        assert canary not in serialized
    for key in ("heavy_auth_token", "token_hash", "prompt_text", "assistant_text", "tool_result_body", "webhook_url", "thread_id", "args"):
        assert key not in clean
    assert validate_event(clean) == []


# --------------------------------------------------------------------------
# 23. Raw thread IDs never appear
# --------------------------------------------------------------------------
def test_raw_thread_id_never_appears(sink):
    _bind({}, bound=["list_dates"], thread_id=CANARY_RAW_THREAD_ID)
    node = AuditedToolNode([FakeTool("list_dates", result="ok")], sink=sink)
    _run(node, _state("list_dates", {"receiver_id": "R1234567"}))
    text = _raw_audit_text(sink)
    assert CANARY_RAW_THREAD_ID not in text
    digest = digest_thread_id(CANARY_RAW_THREAD_ID)
    assert digest in text
    assert digest.startswith("thr-")
    assert digest_thread_id("") == ""
    assert digest != CANARY_RAW_THREAD_ID


def test_authorization_flags_are_booleans_only(sink):
    _bind({"operator_authorized": True, "unexpected_flag": True}, bound=["list_dates"])
    node = AuditedToolNode([FakeTool("list_dates", result="ok")], sink=sink)
    _run(node, _state("list_dates", {"receiver_id": "R1234567"}))
    event = _events(sink)[0]
    assert set(event["authorization_flags"]) == set(AUTHORIZATION_FLAG_KEYS)
    assert all(isinstance(value, bool) for value in event["authorization_flags"].values())
    assert "unexpected_flag" not in json.dumps(event)


# --------------------------------------------------------------------------
# Property-style leak sweep
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {"pattern": CANARY_ARG_VALUE},
        {"nested": {"deep": {"secret": CANARY_TOKEN}}},
        {"webhook": CANARY_WEBHOOK},
        {"email": "canary.person@example.invalid"},
        {"db": "postgresql://canary:CANARY-PASSWORD@127.0.0.1:5432/db"},
        {"long": CANARY_ARG_VALUE * 200},
        {"list": [CANARY_ARG_VALUE, CANARY_TOKEN, CANARY_WEBHOOK]},
    ],
)
def test_no_argument_payload_shape_leaks_values(sink, payload):
    _bind({}, bound=["query_incident_scene"])
    node = AuditedToolNode([FakeTool("query_incident_scene", result=CANARY_RESULT_BODY)], sink=sink)
    _run(node, _state("query_incident_scene", payload))
    text = _raw_audit_text(sink)
    for canary in (*ALL_CANARIES, "canary.person@example.invalid", "CANARY-PASSWORD"):
        assert canary not in text
    event = _events(sink)[0]
    # Credential-shaped *field names* are dropped as well, so the recorded name
    # list is the payload keys minus anything matching the forbidden-key guard.
    expected = sorted(name for name in payload if not _FORBIDDEN_KEY_RE.search(name))
    assert sorted(event["argument_field_names"]) == expected
    assert validate_event(event) == []


def test_every_persisted_record_validates(sink):
    _bind({"heavy_tools_authorized": True}, bound=["list_dates", "build_log_capsule", "query_incident_scene"])
    node = AuditedToolNode(
        [
            FakeTool("list_dates", result="ok"),
            FakeTool("build_log_capsule"),
            FakeTool("query_incident_scene"),
            FakeTool("boom", raises=_CanaryError(CANARY_EXCEPTION_TEXT)),
        ],
        sink=sink,
    )
    for name, args in (
        ("list_dates", {"receiver_id": "R1234567"}),
        ("build_log_capsule", {"receiver_id": "R1234567", "persist": True}),
        ("query_incident_scene", {"limit": 10**9}),
        ("boom", {}),
        ("never_bound", {"x": CANARY_ARG_VALUE}),
    ):
        _run(node, _state(name, args, call_id=f"c-{name}"))

    records = _events(sink)
    assert len(records) == 5
    for record in records:
        assert validate_event(record) == [], record
        assert record["schema_version"] == TOOL_EXECUTION_AUDIT_SCHEMA
    text = _raw_audit_text(sink)
    for canary in ALL_CANARIES:
        assert canary not in text


def test_credential_shaped_field_names_are_dropped(sink):
    """A field *name* that looks credential-bearing is not recorded either."""
    _bind({}, bound=["query_incident_scene"])
    node = AuditedToolNode([FakeTool("query_incident_scene")], sink=sink)
    _run(
        node,
        _state(
            "query_incident_scene",
            {
                "scene_id": "s1",
                "webhook_url": CANARY_WEBHOOK,
                "api_key": "CANARY-API-KEY",
                "session_cookie": "CANARY-COOKIE",
                "db_password": "CANARY-PASSWORD",
            },
        ),
    )
    event = _events(sink)[0]
    assert event["argument_field_names"] == ["scene_id"]
    text = _raw_audit_text(sink)
    for name in ("webhook_url", "api_key", "session_cookie", "db_password"):
        assert name not in text
