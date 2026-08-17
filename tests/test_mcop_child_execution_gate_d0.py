"""Phase D0 tests: the MCOP child graph must use the audited execution gate.

Evidence in these tests is the audit event and the paired ToolMessage.  No test
asserts a model claim, and no live heavy or persistent S3 tool is invoked --
every tool here is synthetic.
"""

from __future__ import annotations

import asyncio
import glob
import inspect
import json
import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.audited_tool_node import AuditedToolNode, make_audited_tool_node
from app.agent.tool_execution_audit import (
    AUTHORIZATION_FLAG_KEYS,
    DECISION_ALLOWED,
    DECISION_BLOCKED,
    SCOPE_MCOP_CHILD,
    SCOPE_PARENT,
    AuditRequestState,
    ToolExecutionAuditSink,
    bind_audit_request_state,
    current_audit_request_state,
    reset_audit_request_state,
    validate_event,
)
from app.agent_mode import child_conversation
from app.agent_mode.child_tool_policy import (
    ChildToolPolicy,
    build_child_policy_snapshot,
    child_audit_scope,
    child_safe_tools,
    clamp_authorization_flags,
    recursive_spawn_tool_names,
)

CANARY_ARG = "CANARY-D0-ARG-7f31"
CANARY_RESULT = "CANARY-D0-RESULT-b902"

HEAVY_TOOL = "build_log_capsule"
READ_ONLY_TOOL = "list_dates"


class FakeTool:
    """Synthetic tool that records the exact effective args it received."""

    def __init__(self, name: str, *, result: str = "ok"):
        self.name = name
        self._result = result
        self.calls: list[dict] = []

    async def ainvoke(self, tool_call, config=None):
        self.calls.append(dict(tool_call.get("args") or {}))
        return ToolMessage(
            content=self._result,
            tool_call_id=tool_call.get("id") or "",
            name=self.name,
        )


def _ai(calls) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": n, "args": dict(a), "id": i, "type": "tool_call"} for n, a, i in calls
        ],
    )


def _run(coro):
    return asyncio.run(coro)


def _events(sink_dir: Path) -> list[dict]:
    out: list[dict] = []
    for path in sorted(glob.glob(str(sink_dir / "**" / "*.jsonl"), recursive=True)):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


@pytest.fixture()
def sink_dir(tmp_path: Path) -> Path:
    return tmp_path / "audit"


@pytest.fixture()
def sink(sink_dir: Path) -> ToolExecutionAuditSink:
    return ToolExecutionAuditSink(sink_dir, max_bytes=400_000, max_files=3)


@pytest.fixture()
def parent_state():
    """A parent correlation state with every authorization boolean false."""
    state = AuditRequestState(thread_id_digest="thr-digest-abcdef")
    token = bind_audit_request_state(state)
    try:
        yield state
    finally:
        reset_audit_request_state(token)


def _child_exec(sink, tools, bound, ai_message, policy=None):
    """Execute one child turn through the audited gate inside child scope."""
    policy = policy or build_child_policy_snapshot()
    node = make_audited_tool_node(tools, sink=sink, node_name="mcop_child_tools")
    with child_audit_scope(policy, bound_tool_names=bound) as child_state:
        result = _run(node({"messages": [ai_message]}))
    return result, child_state, policy


# ---------------------------------------------------------------- 1. no raw ToolNode
def test_child_graph_does_not_use_raw_toolnode(parent_state):
    source = inspect.getsource(child_conversation._build_child_graph)
    assert "ToolNode(tools=" not in source
    assert "ToolNode(" not in source.replace("make_audited_tool_node(", "")
    assert "make_audited_tool_node(" in source


# ---------------------------------------------------------------- 2. unbound blocked
def test_child_unbound_call_is_blocked(sink, sink_dir, parent_state):
    tool = FakeTool("other_tool", result=CANARY_RESULT)
    ai = _ai([("other_tool", {}, "c1")])
    result, _, _ = _child_exec(sink, [tool], [READ_ONLY_TOOL], ai)

    assert tool.calls == []
    events = _events(sink_dir)
    assert len(events) == 1
    assert events[0]["decision"] == DECISION_BLOCKED
    assert events[0]["executed"] is False
    assert events[0]["result_code"] == "BLOCKED_UNBOUND_TOOL"
    assert len(result["messages"]) == 1


# ---------------------------------------------------------------- 3. heavy blocked
def test_child_unauthorized_heavy_call_is_blocked(sink, sink_dir, parent_state):
    tool = FakeTool(HEAVY_TOOL)
    ai = _ai([(HEAVY_TOOL, {"receiver_id": "R1234567890"}, "c2")])
    _child_exec(sink, [tool], [HEAVY_TOOL], ai)

    assert tool.calls == [], "heavy tool executed without authorization"
    event = _events(sink_dir)[0]
    assert event["decision"] == DECISION_BLOCKED
    assert event["executed"] is False
    assert event["result_code"] == "BLOCKED_HEAVY_UNAUTHORIZED"
    assert event["authorization_flags"]["heavy_tools_authorized"] is False


# ------------------------------------------------------- 4. persistence not honored
def test_child_unauthorized_persistence_is_blocked_or_rewritten(sink, sink_dir, parent_state):
    tool = FakeTool(READ_ONLY_TOOL)
    ai = _ai([(READ_ONLY_TOOL, {"receiver_id": "R1234567", "persist": True}, "c3")])
    _child_exec(sink, [tool], [READ_ONLY_TOOL], ai)

    # The security property: persistence is never actually performed.
    for call_args in tool.calls:
        assert call_args.get("persist") is not True
    event = _events(sink_dir)[0]
    if event["executed"]:
        assert "persist" in event["rewritten_fields"]
    else:
        assert event["decision"] == DECISION_BLOCKED


# ---------------------------------------------------------------- 5. allowed executes
def test_child_allowed_read_only_call_executes(sink, sink_dir, parent_state):
    tool = FakeTool(READ_ONLY_TOOL, result=CANARY_RESULT)
    ai = _ai([(READ_ONLY_TOOL, {"receiver_id": "R1234567"}, "c4")])
    result, _, _ = _child_exec(sink, [tool], [READ_ONLY_TOOL], ai)

    assert len(tool.calls) == 1
    event = _events(sink_dir)[0]
    assert event["decision"] == DECISION_ALLOWED
    assert event["executed"] is True
    assert str(result["messages"][0].content) == CANARY_RESULT


# ------------------------------------------------------------ 6. paired ToolMessages
def test_every_child_call_gets_exactly_one_paired_tool_message(sink, sink_dir, parent_state):
    allowed = FakeTool(READ_ONLY_TOOL, result=CANARY_RESULT)
    heavy = FakeTool(HEAVY_TOOL)
    ai = _ai(
        [
            (READ_ONLY_TOOL, {"receiver_id": "R1234567"}, "p1"),
            (HEAVY_TOOL, {"receiver_id": "R1234567"}, "p2"),
            ("unbound_tool", {}, "p3"),
        ]
    )
    result, _, _ = _child_exec(
        sink, [allowed, heavy, FakeTool("unbound_tool")], [READ_ONLY_TOOL, HEAVY_TOOL], ai
    )

    messages = result["messages"]
    assert len(messages) == 3
    ids = [getattr(m, "tool_call_id", "") for m in messages]
    assert sorted(ids) == ["p1", "p2", "p3"]
    assert len(set(ids)) == 3
    assert len(_events(sink_dir)) == 3


# --------------------------------------------------- 7 + 8. audit emitted with scope
def test_child_audit_record_uses_mcop_child_scope(sink, sink_dir, parent_state):
    tool = FakeTool(READ_ONLY_TOOL, result=CANARY_RESULT)
    ai = _ai([(READ_ONLY_TOOL, {"receiver_id": "R1234567"}, "c5")])
    _, child_state, policy = _child_exec(sink, [tool], [READ_ONLY_TOOL], ai)

    events = _events(sink_dir)
    assert len(events) == 1
    event = events[0]
    assert event["scope"] == SCOPE_MCOP_CHILD
    assert event["child_run_id"] == policy.child_run_id
    assert event["child_run_id"]
    assert event["parent_request_id"] == parent_state.request_id
    assert event["thread_id_digest"] == "thr-digest-abcdef"
    assert event["request_id"] != parent_state.request_id
    assert validate_event(event) == []


def test_parent_scope_remains_parent_and_state_is_not_mutated(sink, sink_dir, parent_state):
    parent_binding_before = parent_state.binding_signature
    parent_request_before = parent_state.request_id

    tool = FakeTool(READ_ONLY_TOOL)
    ai = _ai([(READ_ONLY_TOOL, {"receiver_id": "R1234567"}, "c6")])
    _child_exec(sink, [tool], [READ_ONLY_TOOL], ai)

    # After the child scope exits the parent state is restored untouched.
    assert current_audit_request_state() is parent_state
    assert parent_state.binding_signature == parent_binding_before
    assert parent_state.request_id == parent_request_before
    assert parent_state.scope == SCOPE_PARENT


# --------------------------------------------------------- 9. no values or secrets
def test_child_audit_contains_no_values_or_secrets(sink, sink_dir, parent_state):
    tool = FakeTool(READ_ONLY_TOOL, result=CANARY_RESULT)
    ai = _ai(
        [
            (
                READ_ONLY_TOOL,
                {"receiver_id": CANARY_ARG, "heavy_auth_token": "SECRET-TOKEN-VALUE"},
                "c7",
            )
        ]
    )
    _child_exec(sink, [tool], [READ_ONLY_TOOL], ai)

    events = _events(sink_dir)
    blob = json.dumps(events)
    assert CANARY_ARG not in blob
    assert CANARY_RESULT not in blob
    assert "SECRET-TOKEN-VALUE" not in blob
    for event in events:
        assert validate_event(event) == []
        for key in event:
            assert "token" not in key.lower()
            assert "secret" not in key.lower()


# -------------------------------------------------------- 10. recursion excluded
def test_recursive_mcop_spawn_tools_remain_excluded(parent_state):
    spawn_names = recursive_spawn_tool_names()
    assert spawn_names, "spawn exclusion set must not be empty"

    candidates = [FakeTool(name) for name in sorted(spawn_names)] + [FakeTool(READ_ONLY_TOOL)]
    kept = [t.name for t in child_safe_tools(candidates)]
    assert kept == [READ_ONLY_TOOL]
    for name in spawn_names:
        assert name not in kept


def test_child_graph_builder_filters_spawn_tools(parent_state):
    source = inspect.getsource(child_conversation._build_child_graph)
    assert "child_safe_tools(" in source


# ----------------------------------------------- 11. child cannot elevate authorization
def test_child_cannot_receive_more_authorization_than_parent(parent_state):
    all_true = dict.fromkeys(AUTHORIZATION_FLAG_KEYS, True)

    # Parent all false -> child stays all false even when requesting everything.
    policy = build_child_policy_snapshot(requested_authorization=all_true)
    assert policy.flags == dict.fromkeys(AUTHORIZATION_FLAG_KEYS, False)

    # Parent heavy true -> child may inherit heavy, but nothing else.
    parent_state.authorization_flags = {
        **dict.fromkeys(AUTHORIZATION_FLAG_KEYS, False),
        "heavy_tools_authorized": True,
    }
    policy = build_child_policy_snapshot(requested_authorization=all_true)
    assert policy.flags["heavy_tools_authorized"] is True
    assert policy.flags["persistence_authorized"] is False
    assert policy.flags["mutation_authorized"] is False

    # A child may narrow, never widen.
    narrowed = policy.narrow(authorization_flags={"heavy_tools_authorized": False})
    assert narrowed.flags["heavy_tools_authorized"] is False

    # Direct clamp semantics.
    assert clamp_authorization_flags({"heavy_tools_authorized": False}, all_true)[
        "heavy_tools_authorized"
    ] is False


def test_child_scope_state_uses_clamped_flags(sink, sink_dir, parent_state):
    parent_state.authorization_flags = dict.fromkeys(AUTHORIZATION_FLAG_KEYS, False)
    policy = build_child_policy_snapshot(
        requested_authorization=dict.fromkeys(AUTHORIZATION_FLAG_KEYS, True)
    )
    tool = FakeTool(HEAVY_TOOL)
    ai = _ai([(HEAVY_TOOL, {"receiver_id": "R1234567"}, "c8")])
    _child_exec(sink, [tool], [HEAVY_TOOL], ai, policy=policy)

    assert tool.calls == []
    event = _events(sink_dir)[0]
    assert event["executed"] is False
    assert all(value is False for value in event["authorization_flags"].values())


def test_child_policy_snapshot_carries_no_runtime_objects(parent_state):
    policy = build_child_policy_snapshot(
        eligible_toolsets=["s3_stb_logs"], eligible_extra_tools=[HEAVY_TOOL]
    )
    payload = policy.as_safe_dict()
    # Must be JSON serializable: no tool objects, connections, or requests.
    blob = json.dumps(payload, sort_keys=True)
    assert "s3_stb_logs" in blob
    assert isinstance(policy, ChildToolPolicy)
    with pytest.raises(Exception):
        policy.child_run_id = "mutated"  # frozen dataclass


# --------------------------------------------- 12. shared core gate implementation
def test_child_and_parent_gate_share_one_implementation(parent_state):
    node = make_audited_tool_node([FakeTool(READ_ONLY_TOOL)], node_name="mcop_child_tools")
    assert isinstance(node.audited_tool_node, AuditedToolNode)

    child_source = inspect.getsource(child_conversation)
    assert "from app.agent.audited_tool_node import make_audited_tool_node" in child_source

    from app.agent.agents import agentic_rag as parent_graph

    parent_source = inspect.getsource(parent_graph)
    assert "make_audited_tool_node" in parent_source

    # Both call sites resolve to the same function object.
    assert child_conversation.make_audited_tool_node is make_audited_tool_node
    assert parent_graph.make_audited_tool_node is make_audited_tool_node


def test_child_records_exact_per_turn_binding_helper_exists(parent_state):
    assert hasattr(child_conversation, "_record_child_binding")
    agent_source = inspect.getsource(child_conversation._child_agent_node)
    # Both the initial binding and the narrower retry binding must be recorded.
    assert agent_source.count("_record_child_binding(") == 2
    assert "_record_child_binding(child_tools)" in agent_source
    assert "_record_child_binding(retry_tools)" in agent_source
