"""Regression tests for the 2026-08-18 agent loop / history-collapse fixes.

Incident summary (session a597c8f2-1e16-44d3-b5a0-bc6903e52372):
truncate_messages() called trim_messages(strategy="last") without
start_on="human", so the surviving window began mid-tool-chain. The Bedrock
shape repair then "fixed" that by deleting forward to the earliest surviving
HumanMessage, destroying completed tool work (36 -> 10 messages, ~47k tokens
in one pass). The model lost results it had already received and re-issued the
same tool calls, producing a message-count sawtooth (45/10/22/18/25/10) while
tool calls climbed monotonically 21 -> 31 with no terminal message.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app.agent.agents.agentic_rag as agentic_rag
from app.agent import tool_activation_intent as tai


def _pair(tag: str, size: int = 200):
    """One AIMessage(tool_call) + its matching ToolMessage."""
    return [
        AIMessage(content="", tool_calls=[{"id": tag, "name": "search", "args": {}}], id="a" + tag),
        ToolMessage(content="R" * size, tool_call_id=tag, name="search", id="tm" + tag),
    ]


# ------------------------------------------------------- history collapse
def test_bedrock_repair_does_not_destroy_completed_tool_work():
    """The exact production shape: 36 messages, earliest Human at index 22."""
    window = []
    for i in range(11):
        window += _pair(f"old{i}")
    window += [HumanMessage(content="compile into an html brochure", id="hNEW")]
    for i in range(6):
        window += _pair(f"cur{i}")
    window += [AIMessage(content="working", id="aTail")]

    assert len(window) == 36
    tool_msgs_before = sum(1 for m in window if isinstance(m, ToolMessage))

    repaired = agentic_rag.ensure_bedrock_converse_message_shape(
        agentic_rag.sanitize_tool_messages(list(window))
    )

    # Bedrock shape must still be valid.
    assert isinstance(repaired[0], HumanMessage)
    # Zero tool results may be discarded (previously 22 of 36 were deleted).
    assert sum(1 for m in repaired if isinstance(m, ToolMessage)) == tool_msgs_before
    # The real user turn must survive.
    assert any(getattr(m, "id", None) == "hNEW" for m in repaired)
    # Net growth of exactly one synthetic turn, never a collapse.
    assert len(repaired) >= len(window)


def test_bedrock_repair_is_noop_when_already_valid():
    """A already-valid window must not gain a synthetic turn."""
    valid = [HumanMessage(content="hi", id="h1"), *_pair("x0"), AIMessage(content="done", id="f")]
    once = agentic_rag.ensure_bedrock_converse_message_shape(
        agentic_rag.sanitize_tool_messages(list(valid))
    )
    twice = agentic_rag.ensure_bedrock_converse_message_shape(
        agentic_rag.sanitize_tool_messages(list(once))
    )
    assert len(once) == len(valid)
    assert len(twice) == len(once)
    assert not any("Continue this task" in str(m.content) for m in once)


# ------------------------------------------------------- chronology
def test_restored_latest_human_is_chronologically_last():
    """The latest-human failsafe must append, never prepend.

    Prepending put the newest request ahead of older assistant/tool output,
    so the model read stale results as the answer to a request not yet made.
    """
    latest = HumanMessage(content="CURRENT REQUEST: make the brochure", id="hCUR")
    older = [*_pair("p0"), *_pair("p1"), AIMessage(content="earlier answer", id="aOld")]
    fixed = agentic_rag.sanitize_tool_messages([*older, latest])
    assert fixed[-1] is latest or getattr(fixed[-1], "id", None) == "hCUR"


# ------------------------------------------------------- loop safety net
def test_tool_call_counter_scopes_to_current_turn():
    msgs = [
        HumanMessage(content="older turn", id="h0"),
        *_pair("a0"),
        HumanMessage(content="current turn", id="h1"),
        *_pair("b0"),
        *_pair("b1"),
        *_pair("b2"),
    ]
    # Only the 3 pairs after the most recent HumanMessage count.
    assert agentic_rag._count_tool_calls_since_last_human(msgs) == 3


def test_tool_loop_limit_is_wired_to_settings():
    """settings.MAX_TOOL_CALLS_PER_TURN was declared but never referenced."""
    import inspect
    from app.config import get_settings

    assert getattr(get_settings(), "MAX_TOOL_CALLS_PER_TURN", None) is not None
    source = inspect.getsource(agentic_rag.call_model)
    assert "MAX_TOOL_CALLS_PER_TURN" in source
    assert "_count_tool_calls_since_last_human(" in source


# ------------------------------------------------------- activation deadlock
def test_capability_name_requested_as_toolset_does_not_stick_pending():
    """filesystem_write is a capability, never a registry family.

    It previously resolved to PENDING_REGISTRY (meaning "not loaded yet") and
    was re-derived from checkpointed state every turn, so it never drained.
    """
    registry = tai.RegistryToolIndex.live()
    checkpoint = {
        "requested_toolsets": ["filesystem_write"],
        "requested_extra_tools": [],
        "pending_registry_requests": [],
        "activation_request_revision": 2,
    }
    intent = tai.activation_intent_from_checkpoint(checkpoint, registry_index=registry)
    assert "filesystem_write" not in intent.requested_toolsets
    assert "filesystem_write" not in intent.pending_registry_requests
    assert "filesystem_write" in intent.unavailable_requests

    # And it must stay drained across subsequent restores.
    checkpoint["requested_toolsets"] = list(intent.requested_toolsets)
    again = tai.activation_intent_from_checkpoint(checkpoint, registry_index=registry)
    assert "filesystem_write" not in again.requested_toolsets
    assert "filesystem_write" not in again.pending_registry_requests


def test_real_toolsets_are_not_blocked_by_capability_filter():
    registry = tai.RegistryToolIndex.live()
    for name in ("search", "agent_mode", "dish_internal"):
        intent = tai.activation_intent_from_request(
            requested_toolsets=[name], registry_index=registry
        )
        assert name in intent.requested_toolsets, name
        assert name not in intent.unavailable_requests, name


def test_unknown_mcp_family_still_allowed_to_pend():
    """A genuinely unknown *_mcp family must still be checkpointable."""
    registry = tai.RegistryToolIndex.live()
    intent = tai.activation_intent_from_request(
        requested_toolsets=["brandnew_mcp"], registry_index=registry
    )
    assert "brandnew_mcp" in intent.requested_toolsets
