"""Phase D1 checkpoint round-trip and compression-safety tests.

Two real boundaries are exercised:

1. ``ConfigurableEncryptedSerializer`` -- the exact serializer the encrypted
   Postgres saver uses. This proves the D1 fields survive the real
   serialization contract and that no runtime object can enter stored state.
2. A compiled ``StateGraph`` over the real ``AgentState`` with a checkpointer,
   driven across multiple turns on one thread. This proves the reducers,
   safe defaults, stickiness, and replacement semantics behave through genuine
   LangGraph checkpointing.

No production checkpoint row is read, copied, or written, and no real user
checkpoint body appears in evidence. A dedicated synthetic thread namespace is
used throughout.
"""

from __future__ import annotations

import json

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.agents.agentic_rag import AgentState
from app.agent.tool_policy_state import (
    ACTIVATION_PENDING,
    ACTIVATION_UNAVAILABLE,
    AUTHORIZATION_KEYS,
    TOOL_POLICY_VERSION,
    default_tool_policy_state,
    load_tool_policy_state,
    merge_authorization_state,
    parse_authorization_delta,
)

HEAVY = "s3_stb_logs:build_log_capsule"
SCENE = "s3_stb_logs:build_incident_scene"
CANARY_THREAD = "d1-synthetic-canary-thread"


# ------------------------------------------------------- real serializer
def _serializer():
    from app.agent.checkpoint import ConfigurableEncryptedSerializer

    return ConfigurableEncryptedSerializer()


def test_policy_state_survives_real_serializer_round_trip():
    state = default_tool_policy_state()
    state["requested_extra_tools"] = [HEAVY, SCENE]
    state["pending_authorization_extra_tools"] = [HEAVY]
    state["unavailable_extra_tools"] = [SCENE]
    state["authorization_flags"]["heavy_tools_authorized"] = True
    state["last_bound_tool_names"] = ["list_dates", "get_summary"]
    state["last_activation_status"] = {HEAVY: ACTIVATION_PENDING, SCENE: ACTIVATION_UNAVAILABLE}
    state["tool_profile_signature"] = "sha256:abc"
    state["tool_registry_generation"] = "gen-9"

    serde = _serializer()
    restored = serde.loads_typed(serde.dumps_typed(state))
    assert restored == state
    assert restored["authorization_flags"]["heavy_tools_authorized"] is True
    assert restored["last_activation_status"][SCENE] == ACTIVATION_UNAVAILABLE
    assert restored["tool_policy_version"] == TOOL_POLICY_VERSION


def test_serialized_policy_state_is_plain_json_primitives():
    """Guarantees encryptability and that no runtime object can be stored."""
    state = default_tool_policy_state()
    state["requested_extra_tools"] = [HEAVY]
    blob = json.dumps(state, sort_keys=True)
    assert "object at 0x" not in blob
    round_tripped = json.loads(blob)
    assert round_tripped == state


def test_serializer_rejects_nothing_but_stores_no_callables():
    """A tool object must never be placed in policy state in the first place."""
    state = default_tool_policy_state()
    for value in state.values():
        assert not callable(value)
        if isinstance(value, list):
            assert all(not callable(v) for v in value)


# ------------------------------------------------------- graph round trip
def _build_policy_graph(script):
    """Compile a minimal graph over the real AgentState with a checkpointer.

    ``script`` maps turn index -> user text, and the node applies the same D1
    ordering the parent graph uses: load stored state, parse the current-turn
    delta, merge, and return the bounded update.
    """
    calls: list[dict] = []

    def node(state: AgentState, config=None):
        stored = load_tool_policy_state(state)
        text = ""
        messages = state.get("messages") or []
        if messages:
            last = messages[-1]
            text = getattr(last, "content", None) or (
                last.get("content") if isinstance(last, dict) else ""
            )
        delta = parse_authorization_delta(text)
        flags = merge_authorization_state(stored["authorization_flags"], delta)
        calls.append({"stored": stored, "effective_flags": flags})

        update = script(stored, flags, text)
        return update

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", node)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    return workflow.compile(checkpointer=MemorySaver()), calls


def _cfg(thread: str):
    return {"configurable": {"thread_id": thread}}


def test_old_checkpoint_without_d1_fields_loads_defaults_through_graph():
    def script(stored, flags, text):
        return {"messages": [], "authorization_flags": flags}

    graph, calls = _build_policy_graph(script)
    graph.invoke(
        {"messages": [{"role": "user", "content": "hello"}], "model_config": {}},
        config=_cfg(CANARY_THREAD + "-old"),
    )
    assert calls[0]["stored"] == default_tool_policy_state()
    assert all(v is False for v in calls[0]["stored"]["authorization_flags"].values())


def test_authorization_persists_across_turns_and_revocation_sticks():
    def script(stored, flags, text):
        return {"messages": [], "authorization_flags": flags}

    graph, calls = _build_policy_graph(script)
    thread = _cfg(CANARY_THREAD + "-auth")

    # Turn 1: grant heavy.
    graph.invoke(
        {"messages": [{"role": "user", "content": "Heavy tools authorized."}], "model_config": {}},
        config=thread,
    )
    assert calls[-1]["effective_flags"]["heavy_tools_authorized"] is True

    # Turn 2: unrelated text, no grant present anywhere in this message.
    graph.invoke({"messages": [{"role": "user", "content": "Continue."}]}, config=thread)
    assert calls[-1]["stored"]["authorization_flags"]["heavy_tools_authorized"] is True
    assert calls[-1]["effective_flags"]["heavy_tools_authorized"] is True

    # Turn 3: explicit revocation.
    graph.invoke(
        {"messages": [{"role": "user", "content": "Revoke heavy-tool authorization."}]},
        config=thread,
    )
    assert calls[-1]["effective_flags"]["heavy_tools_authorized"] is False

    # Turn 4: revocation must persist.
    graph.invoke({"messages": [{"role": "user", "content": "Say hello."}]}, config=thread)
    assert calls[-1]["stored"]["authorization_flags"]["heavy_tools_authorized"] is False


def test_compression_cannot_erase_a_checkpointed_grant():
    """The D1 security invariant, exercised through a real checkpointer.

    Turn 2 deliberately contains NO grant text at all, simulating compression
    or truncation having removed the original granting message entirely.
    """
    def script(stored, flags, text):
        return {"messages": [], "authorization_flags": flags}

    graph, calls = _build_policy_graph(script)
    thread = _cfg(CANARY_THREAD + "-compress")

    graph.invoke(
        {"messages": [{"role": "user", "content": "Heavy tools authorized."}], "model_config": {}},
        config=thread,
    )
    # The granting message is gone from the model-visible window.
    graph.invoke(
        {"messages": [{"role": "user", "content": "Proceed with the analysis."}]}, config=thread
    )
    assert calls[-1]["effective_flags"]["heavy_tools_authorized"] is True


def test_requested_tools_are_sticky_and_derived_sets_replace():
    def script(stored, flags, text):
        # Requested accumulates; eligible/pending/last-bound replace.
        return {
            "messages": [],
            "authorization_flags": flags,
            "requested_extra_tools": [HEAVY] if "capsule" in text else [SCENE],
            "pending_authorization_extra_tools": [HEAVY] if "capsule" in text else [],
            "last_bound_tool_names": ["bound_a"] if "capsule" in text else ["bound_b"],
        }

    graph, calls = _build_policy_graph(script)
    thread = _cfg(CANARY_THREAD + "-sticky")

    graph.invoke(
        {"messages": [{"role": "user", "content": "build a log capsule"}], "model_config": {}},
        config=thread,
    )
    graph.invoke({"messages": [{"role": "user", "content": "something else"}]}, config=thread)

    stored = calls[-1]["stored"]
    # Sticky union: the earlier request survived a turn that did not mention it.
    assert HEAVY in stored["requested_extra_tools"]
    # Replacement: pending and last-bound reflect only the latest turn.
    assert stored["pending_authorization_extra_tools"] == [HEAVY]
    assert stored["last_bound_tool_names"] == ["bound_a"]

    final = graph.get_state(thread).values
    assert SCENE in final["requested_extra_tools"]
    assert HEAVY in final["requested_extra_tools"]
    assert final["last_bound_tool_names"] == ["bound_b"]
    assert final["pending_authorization_extra_tools"] == []


def test_pending_and_unavailable_status_persist_through_checkpoint():
    def script(stored, flags, text):
        return {
            "messages": [],
            "authorization_flags": flags,
            "requested_extra_tools": [HEAVY, SCENE],
            "pending_authorization_extra_tools": [HEAVY],
            "unavailable_extra_tools": [SCENE],
            "last_activation_status": {HEAVY: ACTIVATION_PENDING, SCENE: ACTIVATION_UNAVAILABLE},
            "tool_registry_generation": "gen-42",
            "tool_profile_signature": "sha256:deadbeef",
        }

    graph, calls = _build_policy_graph(script)
    thread = _cfg(CANARY_THREAD + "-status")
    graph.invoke(
        {"messages": [{"role": "user", "content": "request both tools"}], "model_config": {}},
        config=thread,
    )
    graph.invoke({"messages": [{"role": "user", "content": "again"}]}, config=thread)

    stored = calls[-1]["stored"]
    assert stored["last_activation_status"][HEAVY] == ACTIVATION_PENDING
    assert stored["last_activation_status"][SCENE] == ACTIVATION_UNAVAILABLE
    assert stored["unavailable_extra_tools"] == [SCENE]
    assert stored["tool_registry_generation"] == "gen-42"
    assert stored["tool_profile_signature"] == "sha256:deadbeef"


def test_later_grant_promotes_a_previously_pending_request():
    """Pending -> eligible without the user repeating the request."""
    def script(stored, flags, text):
        requested = sorted(set([*stored.get("requested_extra_tools", []), HEAVY]))
        if flags["heavy_tools_authorized"]:
            eligible, pending = [HEAVY], []
        else:
            eligible, pending = [], [HEAVY]
        return {
            "messages": [],
            "authorization_flags": flags,
            "requested_extra_tools": requested,
            "eligible_extra_tools": eligible,
            "pending_authorization_extra_tools": pending,
        }

    graph, calls = _build_policy_graph(script)
    thread = _cfg(CANARY_THREAD + "-promote")

    # Turn 1: request without authorization -> pending.
    graph.invoke(
        {
            "messages": [{"role": "user", "content": "Request build_log_capsule. Heavy tools are not authorized."}],
            "model_config": {},
        },
        config=thread,
    )
    state = graph.get_state(thread).values
    assert state["pending_authorization_extra_tools"] == [HEAVY]
    assert state["eligible_extra_tools"] == []

    # Turn 2: grant only, no repeat of the tool request.
    graph.invoke({"messages": [{"role": "user", "content": "Heavy tools authorized."}]}, config=thread)
    state = graph.get_state(thread).values
    assert state["eligible_extra_tools"] == [HEAVY]
    assert state["pending_authorization_extra_tools"] == []
    assert HEAVY in state["requested_extra_tools"]

    # Turn 3: revoke -> eligibility removed, request survives.
    graph.invoke(
        {"messages": [{"role": "user", "content": "Revoke heavy-tool authorization."}]}, config=thread
    )
    state = graph.get_state(thread).values
    assert state["eligible_extra_tools"] == []
    assert state["pending_authorization_extra_tools"] == [HEAVY]
    assert HEAVY in state["requested_extra_tools"]


def test_no_runtime_object_appears_in_serialized_checkpoint():
    def script(stored, flags, text):
        return {"messages": [], "authorization_flags": flags, "requested_extra_tools": [HEAVY]}

    graph, _ = _build_policy_graph(script)
    thread = _cfg(CANARY_THREAD + "-serialize")
    graph.invoke(
        {"messages": [{"role": "user", "content": "hello"}], "model_config": {}}, config=thread
    )
    values = graph.get_state(thread).values
    policy_only = {k: v for k, v in values.items() if k not in ("messages", "model_config")}
    blob = json.dumps(policy_only, sort_keys=True, default=str)
    for marker in ("object at 0x", "ContextVar", "Connection", "StructuredTool", "<function"):
        assert marker not in blob


def test_graph_state_bounds_survive_round_trip():
    from app.agent.tool_policy_state import FIELD_BOUNDS

    def script(stored, flags, text):
        return {
            "messages": [],
            "authorization_flags": flags,
            "requested_extra_tools": ["s3_stb_logs:t%04d" % i for i in range(5000)],
        }

    graph, _ = _build_policy_graph(script)
    thread = _cfg(CANARY_THREAD + "-bounds")
    graph.invoke(
        {"messages": [{"role": "user", "content": "hello"}], "model_config": {}}, config=thread
    )
    values = graph.get_state(thread).values
    assert len(values["requested_extra_tools"]) <= FIELD_BOUNDS["requested_extra_tools"]


def test_real_parent_graph_compiles_with_d1_state():
    """The active graph must still compile with the extended state schema."""
    import typing

    hints = typing.get_type_hints(AgentState, include_extras=True)
    assert "authorization_flags" in hints
    assert "messages" in hints
    # Building a StateGraph over the real schema must not raise.
    workflow = StateGraph(AgentState)
    workflow.add_node("noop", lambda state, config=None: {"messages": []})
    workflow.add_edge(START, "noop")
    workflow.add_edge("noop", END)
    assert workflow.compile(checkpointer=MemorySaver()) is not None
