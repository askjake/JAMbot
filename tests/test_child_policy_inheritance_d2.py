"""Phase D2 tests: bounded parent-to-child policy inheritance.

The child is an ephemeral, bounded delegate of one parent turn. The parent
snapshot is an upper bound: a child may narrow it, never broaden it.

All tools and registries here are synthetic. No live heavy or persistent S3
operation is invoked.
"""

from __future__ import annotations

import inspect
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.audited_tool_node import make_audited_tool_node
from app.agent.tool_execution_audit import (
    AUTHORIZATION_FLAG_KEYS,
    CHILD_SNAPSHOT_STATUSES,
    SCOPE_MCOP_CHILD,
    SNAPSHOT_ACCEPTED,
    SNAPSHOT_GENERATION_MISMATCH_NARROWED,
    SNAPSHOT_MALFORMED_BLOCKED,
    SNAPSHOT_NARROWED,
    AuditRequestState,
    ToolExecutionAuditSink,
    bind_audit_request_state,
    current_audit_request_state,
    reset_audit_request_state,
    validate_event,
)
from app.agent_mode import child_conversation
from app.agent_mode.child_tool_policy import (
    CHILD_POLICY_VERSION,
    CHILD_TOOL_POLICY_SCHEMA,
    MAX_CHILD_EXTRA_TOOLS,
    MAX_CHILD_TOOLSETS,
    ChildToolPolicy,
    build_child_policy_snapshot,
    child_audit_scope,
    child_safe_tools,
    clamp_authorization_flags,
    fail_closed_policy,
    narrow_policy_for_task,
    reconcile_snapshot_with_registry,
    recursive_spawn_tool_names,
    validate_child_policy_snapshot,
)

HEAVY = "s3_stb_logs:build_log_capsule"
SCENE = "s3_stb_logs:build_incident_scene"
READ_ONLY = "list_dates"
ALL_TRUE = dict.fromkeys(AUTHORIZATION_FLAG_KEYS, True)
ALL_FALSE = dict.fromkeys(AUTHORIZATION_FLAG_KEYS, False)


class FakeTool:
    def __init__(self, name: str, *, result: str = "ok"):
        self.name = name
        self._result = result
        self.calls: list[dict] = []

    async def ainvoke(self, tool_call, config=None):
        self.calls.append(dict(tool_call.get("args") or {}))
        return ToolMessage(content=self._result, tool_call_id=tool_call.get("id") or "", name=self.name)


@pytest.fixture()
def parent_state():
    state = AuditRequestState(thread_id_digest="thr-d2-digest")
    token = bind_audit_request_state(state)
    try:
        yield state
    finally:
        reset_audit_request_state(token)


def _policy(**kw):
    flags = kw.pop("flags", ALL_FALSE)
    return ChildToolPolicy(
        authorization_flags=tuple(sorted(dict(flags).items())),
        eligible_toolsets=tuple(kw.pop("toolsets", ())),
        eligible_extra_tools=tuple(kw.pop("extras", ())),
        registry_generation=kw.pop("generation", "gen-1"),
        profile_signature=kw.pop("signature", "sha256:sig"),
        parent_request_id=kw.pop("parent_request_id", "req-parent"),
        parent_thread_digest=kw.pop("digest", "thr-d2-digest"),
        child_run_id=kw.pop("child_run_id", "child-abc123"),
        snapshot_status=kw.pop("status", SNAPSHOT_ACCEPTED),
    )


# ------------------------------------------------- 1-2. defaults / malformed
def test_snapshot_defaults_are_safe(parent_state):
    policy = build_child_policy_snapshot()
    assert policy.flags == ALL_FALSE
    assert policy.schema_version == CHILD_TOOL_POLICY_SCHEMA
    assert policy.policy_version == CHILD_POLICY_VERSION
    assert policy.child_run_id


def test_fail_closed_policy_grants_nothing():
    policy = fail_closed_policy()
    assert policy.flags == ALL_FALSE
    assert policy.eligible_toolsets == ()
    assert policy.eligible_extra_tools == ()
    assert policy.snapshot_status == SNAPSHOT_MALFORMED_BLOCKED
    assert validate_child_policy_snapshot(policy) == []


def test_malformed_snapshot_is_detected_and_fails_closed():
    assert validate_child_policy_snapshot(None) == ["snapshot is not a ChildToolPolicy"]
    assert validate_child_policy_snapshot({"flags": {}}) == ["snapshot is not a ChildToolPolicy"]

    bad_schema = ChildToolPolicy(child_run_id="c1", schema_version="wrong.v9")
    assert "schema_version mismatch" in validate_child_policy_snapshot(bad_schema)

    missing_id = ChildToolPolicy(
        authorization_flags=tuple(sorted(ALL_FALSE.items()))
    )
    assert "missing child_run_id" in validate_child_policy_snapshot(missing_id)

    bad_status = _policy(status="NOT_A_STATUS")
    assert "snapshot_status not enumerated" in validate_child_policy_snapshot(bad_status)


def test_child_conversation_fails_closed_on_malformed_snapshot():
    source = inspect.getsource(child_conversation.run_child_conversation)
    assert "validate_child_policy_snapshot(" in source
    assert "fail_closed_policy(" in source


# ------------------------------------------------- 3-6. non-elevation
def test_parent_false_stays_false_in_child(parent_state):
    parent_state.authorization_flags = dict(ALL_FALSE)
    policy = build_child_policy_snapshot(requested_authorization=ALL_TRUE)
    assert policy.flags == ALL_FALSE


def test_parent_true_may_remain_true(parent_state):
    parent_state.authorization_flags = {**ALL_FALSE, "heavy_tools_authorized": True}
    policy = build_child_policy_snapshot(requested_authorization=ALL_TRUE)
    assert policy.flags["heavy_tools_authorized"] is True
    assert policy.flags["persistence_authorized"] is False
    assert policy.flags["mutation_authorized"] is False
    assert policy.flags["operator_authorized"] is False


def test_child_cannot_elevate_false_to_true():
    for key in AUTHORIZATION_FLAG_KEYS:
        clamped = clamp_authorization_flags(ALL_FALSE, {key: True})
        assert clamped[key] is False


def test_child_task_may_narrow_true_to_false():
    policy = _policy(flags={**ALL_FALSE, "heavy_tools_authorized": True})
    narrowed = policy.narrow(authorization_flags={"heavy_tools_authorized": False})
    assert narrowed.flags["heavy_tools_authorized"] is False
    assert narrowed.snapshot_status == SNAPSHOT_NARROWED


def test_narrow_cannot_reintroduce_a_denied_flag():
    policy = _policy(flags=ALL_FALSE)
    assert policy.narrow(authorization_flags=ALL_TRUE).flags == ALL_FALSE


# ------------------------------------------------- 7-9. upper bounds
def test_parent_toolset_upper_bound_is_enforced():
    policy = _policy(toolsets=("s3_stb_logs",))
    narrowed = policy.narrow(eligible_toolsets=["s3_stb_logs", "jira_mcp"])
    assert narrowed.eligible_toolsets == ("s3_stb_logs",)
    assert "jira_mcp" not in narrowed.eligible_toolsets


def test_parent_exact_tool_upper_bound_is_enforced():
    policy = _policy(extras=(HEAVY,))
    narrowed = policy.narrow(eligible_extra_tools=[HEAVY, SCENE])
    assert narrowed.eligible_extra_tools == (HEAVY,)
    assert SCENE not in narrowed.eligible_extra_tools


def test_task_narrowing_intersects(parent_state):
    policy = _policy(toolsets=("s3_stb_logs", "qos_mcp"), extras=(HEAVY, SCENE))
    narrowed = narrow_policy_for_task(
        policy, task_required_toolsets=["qos_mcp", "jira_mcp"], task_required_tools=[SCENE, "x:y"]
    )
    assert narrowed.eligible_toolsets == ("qos_mcp",)
    assert narrowed.eligible_extra_tools == (SCENE,)


def test_empty_task_requirement_does_not_grant_everything():
    policy = _policy(toolsets=("s3_stb_logs",), extras=(HEAVY,))
    assert narrow_policy_for_task(policy) == policy


# ------------------------------------------------- 10-15. tool filtering
def test_recursive_spawn_tools_remain_excluded():
    spawn = recursive_spawn_tool_names()
    assert spawn
    candidates = [FakeTool(n) for n in sorted(spawn)] + [FakeTool(READ_ONLY)]
    kept = [t.name for t in child_safe_tools(candidates)]
    assert kept == [READ_ONLY]


def test_unrelated_families_removed_by_exact_upper_bound():
    policy = _policy(extras=("s3_stb_logs:list_dates",))
    tools = [FakeTool("list_dates"), FakeTool("jira_search"), FakeTool("qos_get_coverage")]
    kept = [t.name for t in child_safe_tools(tools, policy)]
    assert kept == ["list_dates"]


def test_heavy_tool_excluded_when_heavy_false():
    policy = _policy(flags=ALL_FALSE, extras=(HEAVY, "s3_stb_logs:list_dates"))
    tools = [FakeTool("build_log_capsule"), FakeTool("list_dates")]
    kept = [t.name for t in child_safe_tools(tools, policy)]
    assert "build_log_capsule" not in kept
    assert "list_dates" in kept


def test_heavy_tool_allowed_only_when_heavy_true():
    policy = _policy(flags={**ALL_FALSE, "heavy_tools_authorized": True}, extras=(HEAVY,))
    kept = [t.name for t in child_safe_tools([FakeTool("build_log_capsule")], policy)]
    assert kept == ["build_log_capsule"]


def test_persistent_tool_excluded_when_persistence_false():
    policy = _policy(
        flags={**ALL_FALSE, "heavy_tools_authorized": True},
        extras=("s3_stb_logs:build_complete_log_capsule",),
    )
    kept = [t.name for t in child_safe_tools([FakeTool("build_complete_log_capsule")], policy)]
    assert kept == [], "persistent tool bound without persistence authorization"


def test_mutation_tool_excluded_when_mutation_false():
    policy = _policy(flags=ALL_FALSE, extras=("grasshopper_mcp:grasshopper_upload",))
    kept = [t.name for t in child_safe_tools([FakeTool("grasshopper_upload")], policy)]
    assert kept == []


def test_task_required_names_further_restrict():
    policy = _policy(extras=("s3_stb_logs:list_dates", "s3_stb_logs:get_summary"))
    tools = [FakeTool("list_dates"), FakeTool("get_summary")]
    kept = [t.name for t in child_safe_tools(tools, policy, task_required_names=["list_dates"])]
    assert kept == ["list_dates"]


# ------------------------------------------------- 16-20. registry mismatch
def test_generation_match_returns_policy_unchanged():
    policy = _policy(extras=(HEAVY,), generation="gen-live")
    out, match = reconcile_snapshot_with_registry(policy, current_generation="gen-live")
    assert match is True
    assert out == policy


def test_generation_mismatch_is_detected():
    policy = _policy(extras=(HEAVY,), generation="gen-old")
    _out, match = reconcile_snapshot_with_registry(policy, current_generation="gen-new")
    assert match is False


def test_generation_mismatch_removes_disappeared_tools(monkeypatch):
    policy = _policy(extras=(HEAVY, SCENE), generation="gen-old")
    import app.agent.tool_policy_state as tps

    monkeypatch.setattr(tps, "resolve_upstream_availability", lambda names: ([HEAVY], [SCENE]))
    out, match = reconcile_snapshot_with_registry(policy, current_generation="gen-new")
    assert match is False
    assert out.eligible_extra_tools == (HEAVY,)
    assert SCENE not in out.eligible_extra_tools
    assert out.snapshot_status == SNAPSHOT_GENERATION_MISMATCH_NARROWED


def test_generation_mismatch_never_adds_new_tools(monkeypatch):
    """A newly available tool must not be added just because it now exists."""
    policy = _policy(extras=(HEAVY,), generation="gen-old")
    import app.agent.tool_policy_state as tps

    monkeypatch.setattr(
        tps, "resolve_upstream_availability", lambda names: ([HEAVY, SCENE, "x:new_tool"], [])
    )
    out, _match = reconcile_snapshot_with_registry(policy, current_generation="gen-new")
    assert out.eligible_extra_tools == (HEAVY,)
    assert SCENE not in out.eligible_extra_tools
    assert "x:new_tool" not in out.eligible_extra_tools


def test_generation_mismatch_cannot_broaden_toolsets(monkeypatch):
    policy = _policy(toolsets=("s3_stb_logs",), extras=(HEAVY,), generation="gen-old")
    import app.agent.tool_policy_state as tps

    monkeypatch.setattr(tps, "resolve_upstream_availability", lambda names: ([HEAVY], []))
    out, _ = reconcile_snapshot_with_registry(policy, current_generation="gen-new")
    assert set(out.eligible_toolsets) <= set(policy.eligible_toolsets)


def test_availability_failure_fails_closed(monkeypatch):
    policy = _policy(extras=(HEAVY,), generation="gen-old")
    import app.agent.tool_policy_state as tps

    def boom(names):
        raise RuntimeError("registry down")

    monkeypatch.setattr(tps, "resolve_upstream_availability", boom)
    out, match = reconcile_snapshot_with_registry(policy, current_generation="gen-new")
    assert match is False
    assert out.eligible_extra_tools == ()
    assert out.snapshot_status == SNAPSHOT_GENERATION_MISMATCH_NARROWED


# ------------------------------------------------- 21-22. child binding
def test_child_binding_is_built_from_permitted_intersection():
    source = inspect.getsource(child_conversation._child_agent_node)
    assert "child_safe_tools(child_tools, _child_policy)" in source
    assert "child_safe_tools(retry_tools, _child_policy)" in source
    assert source.count("_record_child_binding(") == 2


def test_graph_executor_is_also_bounded_by_policy():
    source = inspect.getsource(child_conversation._build_child_graph)
    assert "child_safe_tools(child_tools, policy)" in source


# ------------------------------------------------- 23-25. audit correlation
def test_child_audit_scope_carries_snapshot_correlation(parent_state):
    policy = _policy(generation="gen-snap")
    with child_audit_scope(policy, generation_match=False, current_generation="gen-live") as state:
        assert state.scope == SCOPE_MCOP_CHILD
        assert state.child_run_id == policy.child_run_id
        assert state.snapshot_policy_version == CHILD_POLICY_VERSION
        assert state.snapshot_registry_generation == "gen-snap"
        assert state.current_registry_generation == "gen-live"
        assert state.generation_match is False
        assert state.snapshot_status in CHILD_SNAPSHOT_STATUSES
    assert current_audit_request_state() is parent_state


def test_child_audit_event_has_correlation_and_no_secrets(tmp_path, parent_state):
    sink = ToolExecutionAuditSink(tmp_path / "audit", max_bytes=200_000, max_files=2)
    node = make_audited_tool_node([FakeTool(READ_ONLY, result="CANARY-RESULT")], sink=sink,
                                  node_name="mcop_child_tools")
    policy = _policy(extras=("s3_stb_logs:list_dates",), generation="gen-snap")
    ai = AIMessage(content="", tool_calls=[{"name": READ_ONLY,
                                            "args": {"receiver_id": "CANARY-ARG"},
                                            "id": "c1", "type": "tool_call"}])
    import asyncio

    with child_audit_scope(policy, bound_tool_names=[READ_ONLY],
                           generation_match=False, current_generation="gen-live"):
        asyncio.run(node({"messages": [ai]}))

    events = []
    for path in sorted((tmp_path / "audit").glob("*.jsonl")):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                events.append(json.loads(line))
    assert len(events) == 1
    event = events[0]
    assert event["scope"] == SCOPE_MCOP_CHILD
    assert event["child_run_id"] == policy.child_run_id
    assert event["parent_request_id"] == policy.parent_request_id
    assert event["snapshot_registry_generation"] == "gen-snap"
    assert event["current_registry_generation"] == "gen-live"
    assert event["generation_match"] is False
    assert event["snapshot_policy_version"] == CHILD_POLICY_VERSION
    assert validate_event(event) == []
    blob = json.dumps(event)
    assert "CANARY-ARG" not in blob
    assert "CANARY-RESULT" not in blob


def test_snapshot_contains_no_prohibited_material():
    policy = _policy(extras=(HEAVY,), toolsets=("s3_stb_logs",))
    payload = policy.as_safe_dict()
    blob = json.dumps(payload, sort_keys=True)
    for marker in ("object at 0x", "<function", "Connection", "ContextVar", "token", "webhook"):
        assert marker not in blob
    assert payload["authorization_material_stored"] is False
    # No raw thread/chat/user id: only a digest crosses the boundary.
    assert "parent_thread_digest" in payload
    assert "thread_id" not in payload
    assert "chat_id" not in payload


# ------------------------------------------------- 26. bounded completion
def test_child_completion_contract_is_bounded():
    from app.agent_mode.child_conversation import ChildResult

    fields = set(ChildResult.__dataclass_fields__)
    for required in ("child_run_id", "toolsets_permitted", "tools_executed", "audit_event_ids"):
        assert required in fields
    for forbidden in ("child_policy", "policy_snapshot", "messages", "graph_state",
                      "authorization_flags", "tool_arguments"):
        assert forbidden not in fields


def test_executed_tool_names_is_bounded_and_deduped():
    msgs = [ToolMessage(content="x", tool_call_id=str(i), name="list_dates") for i in range(5)]
    msgs += [ToolMessage(content="y", tool_call_id="z", name="get_summary")]
    names = child_conversation._executed_tool_names(msgs)
    assert names == ["list_dates", "get_summary"]
    many = [ToolMessage(content="x", tool_call_id=str(i), name="t%03d" % i) for i in range(200)]
    assert len(child_conversation._executed_tool_names(many, limit=10)) == 10


def test_audit_event_id_helper_never_raises():
    assert child_conversation._child_audit_event_ids("") == []
    assert isinstance(child_conversation._child_audit_event_ids("child-does-not-exist"), list)


# ------------------------------------------------- 27-29. isolation
def test_parent_state_is_not_mutated_by_child_scope(parent_state):
    parent_state.authorization_flags = {**ALL_FALSE, "heavy_tools_authorized": True}
    before_flags = dict(parent_state.authorization_flags)
    before_request = parent_state.request_id
    before_binding = parent_state.binding_signature

    policy = build_child_policy_snapshot(requested_authorization={"heavy_tools_authorized": False})
    with child_audit_scope(policy, bound_tool_names=["list_dates"]):
        pass

    assert parent_state.authorization_flags == before_flags
    assert parent_state.request_id == before_request
    assert parent_state.binding_signature == before_binding
    assert current_audit_request_state() is parent_state


def test_two_children_do_not_share_mutable_policy_state(parent_state):
    parent_state.authorization_flags = {**ALL_FALSE, "heavy_tools_authorized": True}
    a = build_child_policy_snapshot(eligible_toolsets=["s3_stb_logs"], eligible_extra_tools=[HEAVY])
    b = build_child_policy_snapshot(eligible_toolsets=["qos_mcp"], eligible_extra_tools=[])

    assert a.child_run_id != b.child_run_id
    assert a.eligible_toolsets != b.eligible_toolsets
    # Narrowing one must not affect the other.
    a_narrowed = a.narrow(eligible_extra_tools=[])
    assert a.eligible_extra_tools == (HEAVY,)
    assert a_narrowed.eligible_extra_tools == ()
    assert b.eligible_extra_tools == ()

    states = []
    with child_audit_scope(a) as sa:
        states.append(sa)
        with child_audit_scope(b) as sb:
            states.append(sb)
            assert sb.child_run_id == b.child_run_id
        assert current_audit_request_state() is sa
    assert states[0] is not states[1]
    assert states[0].authorization_flags is not states[1].authorization_flags


def test_sequential_children_reflect_current_parent_state(parent_state):
    parent_state.authorization_flags = {**ALL_FALSE, "heavy_tools_authorized": True}
    first = build_child_policy_snapshot()
    assert first.flags["heavy_tools_authorized"] is True

    # Parent revokes between spawns.
    parent_state.authorization_flags = dict(ALL_FALSE)
    second = build_child_policy_snapshot()
    assert second.flags["heavy_tools_authorized"] is False
    assert first.flags["heavy_tools_authorized"] is True, "running child snapshot was mutated"


# ------------------------------------------------- 30. revocation to new child
def test_new_child_after_revocation_inherits_false(parent_state):
    parent_state.authorization_flags = {**ALL_FALSE, "heavy_tools_authorized": True}
    granted = build_child_policy_snapshot(eligible_extra_tools=[HEAVY])
    assert granted.flags["heavy_tools_authorized"] is True
    assert [t.name for t in child_safe_tools([FakeTool("build_log_capsule")], granted)] == [
        "build_log_capsule"
    ]

    parent_state.authorization_flags = dict(ALL_FALSE)
    revoked = build_child_policy_snapshot(eligible_extra_tools=[HEAVY])
    assert revoked.flags["heavy_tools_authorized"] is False
    assert child_safe_tools([FakeTool("build_log_capsule")], revoked) == []


# ------------------------------------------------- bounds and persistence
def test_snapshot_bounds_are_enforced(parent_state):
    policy = build_child_policy_snapshot(
        eligible_toolsets=["ts%03d" % i for i in range(500)],
        eligible_extra_tools=["s3_stb_logs:t%03d" % i for i in range(500)],
    )
    assert len(policy.eligible_toolsets) <= MAX_CHILD_TOOLSETS
    assert len(policy.eligible_extra_tools) <= MAX_CHILD_EXTRA_TOOLS
    assert "eligible_toolsets" in policy.truncated_fields
    assert "eligible_extra_tools" in policy.truncated_fields
    assert validate_child_policy_snapshot(policy) == []


def test_snapshot_is_immutable():
    policy = _policy()
    with pytest.raises(Exception):
        policy.child_run_id = "mutated"
    with pytest.raises(Exception):
        policy.eligible_extra_tools = (HEAVY,)


def test_child_remains_ephemeral_memorysaver():
    source = inspect.getsource(child_conversation._build_child_graph)
    assert "MemorySaver()" in source
    assert "get_checkpointer" not in source
    assert "Postgres" not in source


# ------------------------------------------------- family upper bound on binding
def test_family_upper_bound_is_enforced_on_the_binding(monkeypatch):
    """A family absent from the parent snapshot cannot appear in the binding."""
    import app.agent.tool_execution_policy as tep

    def fake_expand(toolsets, **kw):
        table = {
            "s3_stb_logs": [FakeTool("list_dates"), FakeTool("get_summary")],
            "jira_mcp": [FakeTool("jira_search")],
        }
        out = []
        for ts in toolsets or ():
            out.extend(table.get(ts, []))
        return out

    monkeypatch.setattr(tep, "get_tools_for_toolsets", fake_expand)

    policy = _policy(toolsets=("s3_stb_logs",))
    candidates = [FakeTool("list_dates"), FakeTool("jira_search"), FakeTool("get_summary")]
    kept = [t.name for t in child_safe_tools(candidates, policy)]
    assert sorted(kept) == ["get_summary", "list_dates"]
    assert "jira_search" not in kept, "tool from an unpermitted family reached the binding"


def test_family_expansion_failure_fails_closed(monkeypatch):
    import app.agent.tool_execution_policy as tep

    def boom(toolsets, **kw):
        raise RuntimeError("registry down")

    monkeypatch.setattr(tep, "get_tools_for_toolsets", boom)
    policy = _policy(toolsets=("s3_stb_logs",))
    assert child_safe_tools([FakeTool("list_dates")], policy) == []


def test_exact_permitted_tool_survives_family_expansion(monkeypatch):
    """An explicitly permitted exact tool is honoured even if family expansion misses it."""
    import app.agent.tool_execution_policy as tep

    monkeypatch.setattr(tep, "get_tools_for_toolsets", lambda toolsets, **kw: [])
    policy = _policy(toolsets=("s3_stb_logs",), extras=("s3_stb_logs:list_dates",))
    kept = [t.name for t in child_safe_tools([FakeTool("list_dates")], policy)]
    assert kept == ["list_dates"]


def test_no_family_bound_when_snapshot_declares_none():
    policy = _policy(toolsets=())
    kept = [t.name for t in child_safe_tools([FakeTool("list_dates")], policy)]
    assert kept == ["list_dates"]
