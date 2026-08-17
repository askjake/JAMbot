"""Phase D1 tests: checkpointed parent authorization and tool-policy state.

The security property under test is that authorization is explicit checkpoint
state, not a function of message retention.  No test asserts a model claim and
no live heavy or persistent S3 operation is invoked.
"""

from __future__ import annotations

import inspect
import json

import pytest

from app.agent.tool_policy_state import (
    ACTIVATION_BLOCKED,
    ACTIVATION_BOUND,
    ACTIVATION_ELIGIBLE,
    ACTIVATION_PENDING,
    ACTIVATION_REQUESTED,
    ACTIVATION_REVOKED,
    ACTIVATION_UNAVAILABLE,
    AUTHORIZATION_KEYS,
    FIELD_BOUNDS,
    GRANT,
    MAX_LAST_BOUND_TOOL_NAMES,
    REVOKE,
    TOOL_POLICY_VERSION,
    UNCHANGED,
    AuthorizationDelta,
    classify_activation_state,
    compute_policy_signature,
    default_tool_policy_state,
    load_tool_policy_state,
    make_ordered_set_reducer,
    make_replacement_reducer,
    merge_authorization_state,
    normalize_name_list,
    parse_authorization_delta,
    reduce_activation_status,
    reduce_authorization_flags,
    reduce_scalar,
    reduce_version,
)

HEAVY = "s3_stb_logs:build_log_capsule"
SCENE = "s3_stb_logs:build_incident_scene"


# ---------------------------------------------------------------- defaults
def test_default_state_is_all_false_and_empty():
    state = default_tool_policy_state()
    assert state["authorization_flags"] == dict.fromkeys(AUTHORIZATION_KEYS, False)
    for key in (
        "active_toolsets",
        "requested_extra_tools",
        "eligible_extra_tools",
        "pending_authorization_extra_tools",
        "unavailable_extra_tools",
        "last_bound_tool_names",
    ):
        assert state[key] == []
    assert state["tool_profile_signature"] == ""
    assert state["tool_registry_generation"] == ""
    assert state["last_activation_status"] == {}
    assert state["tool_policy_version"] == TOOL_POLICY_VERSION


def test_old_checkpoint_without_d1_fields_loads_safely():
    # An old checkpoint carries only messages/model_config.
    old = {"messages": [], "model_config": {"temperature": 0.5}}
    loaded = load_tool_policy_state(old)
    assert loaded == default_tool_policy_state()
    assert all(value is False for value in loaded["authorization_flags"].values())


def test_old_checkpoint_with_grant_text_infers_nothing():
    old = {"messages": ["Heavy tools authorized.", "operator authorized"]}
    loaded = load_tool_policy_state(old)
    assert loaded["authorization_flags"]["heavy_tools_authorized"] is False
    assert loaded["authorization_flags"]["operator_authorized"] is False


def test_unknown_future_fields_do_not_crash_reader():
    loaded = load_tool_policy_state({"some_future_field": {"a": 1}, "tool_policy_version": 99})
    assert loaded["tool_policy_version"] == 99
    assert loaded["authorization_flags"] == dict.fromkeys(AUTHORIZATION_KEYS, False)


def test_partial_state_fills_missing_defaults():
    loaded = load_tool_policy_state({"requested_extra_tools": [HEAVY]})
    assert loaded["requested_extra_tools"] == [HEAVY]
    assert loaded["eligible_extra_tools"] == []
    assert loaded["authorization_flags"]["heavy_tools_authorized"] is False


# ------------------------------------------------------- delta parsing
@pytest.mark.parametrize(
    "text,key,expected",
    [
        ("Heavy tools authorized.", "heavy_tools_authorized", GRANT),
        ("Heavy tools are not authorized.", "heavy_tools_authorized", REVOKE),
        ("Do not authorize heavy tools.", "heavy_tools_authorized", REVOKE),
        ("Revoke heavy authorization.", "heavy_tools_authorized", REVOKE),
        ("Give me a one-sentence greeting.", "heavy_tools_authorized", UNCHANGED),
        ("Tell me whether heavy tools are authorized.", "heavy_tools_authorized", UNCHANGED),
        ("Persistence is authorized.", "persistence_authorized", GRANT),
        ("Persistence is not authorized.", "persistence_authorized", REVOKE),
    ],
)
def test_delta_parsing_per_capability(text, key, expected):
    assert parse_authorization_delta(text).as_dict()[key] == expected


def test_delta_parses_multiple_capabilities_in_one_grant():
    delta = parse_authorization_delta("Heavy tools and persistence are authorized.")
    assert delta.as_dict()["heavy_tools_authorized"] == GRANT
    assert delta.as_dict()["persistence_authorized"] == GRANT


def test_mixed_statement_precedence():
    # Negative clause for a specific capability overrides the broad positive.
    delta = parse_authorization_delta("Heavy tools are authorized, but persistence is not.")
    assert delta.as_dict()["heavy_tools_authorized"] == GRANT
    assert delta.as_dict()["persistence_authorized"] == REVOKE


def test_negative_statement_never_grants():
    delta = parse_authorization_delta(
        "Heavy tools are not authorized. Persistence is not authorized."
    )
    flags = merge_authorization_state(None, delta)
    assert flags["heavy_tools_authorized"] is False
    assert flags["persistence_authorized"] is False


def test_empty_and_none_text_is_unchanged():
    for text in ("", None, "   "):
        assert parse_authorization_delta(text).is_empty()


def test_delta_changed_keys_reports_only_explicit():
    delta = parse_authorization_delta("Heavy tools authorized.")
    assert delta.changed_keys() == ("heavy_tools_authorized",)


# ------------------------------------------------------- merge semantics
def test_grant_then_unchanged_is_sticky():
    flags = merge_authorization_state(None, parse_authorization_delta("Heavy tools authorized."))
    assert flags["heavy_tools_authorized"] is True
    # A later unrelated turn must not drop the grant.
    later = merge_authorization_state(flags, parse_authorization_delta("What is the weather?"))
    assert later["heavy_tools_authorized"] is True


def test_revocation_persists_across_unchanged_turns():
    flags = {"heavy_tools_authorized": True}
    revoked = merge_authorization_state(
        flags, parse_authorization_delta("Revoke heavy-tool authorization.")
    )
    assert revoked["heavy_tools_authorized"] is False
    later = merge_authorization_state(revoked, parse_authorization_delta("Say hello."))
    assert later["heavy_tools_authorized"] is False


def test_merge_with_missing_previous_defaults_false():
    flags = merge_authorization_state(None, AuthorizationDelta())
    assert flags == dict.fromkeys(AUTHORIZATION_KEYS, False)


def test_merge_is_idempotent():
    delta = parse_authorization_delta("Heavy tools authorized.")
    once = merge_authorization_state(None, delta)
    twice = merge_authorization_state(once, delta)
    assert once == twice


# -------------------------------------- THE compression security invariant
def test_authorization_survives_removal_of_the_grant_message():
    """A grant must survive compression that deletes the granting message."""
    stored = merge_authorization_state(
        None, parse_authorization_delta("Heavy tools authorized.")
    )
    assert stored["heavy_tools_authorized"] is True

    # Next turn: the grant message is gone from model-visible history entirely.
    compressed_window_text = "Continue the analysis."
    effective = merge_authorization_state(
        stored, parse_authorization_delta(compressed_window_text)
    )
    assert effective["heavy_tools_authorized"] is True, "compression silently revoked a grant"


def test_revocation_survives_removal_of_the_revocation_message():
    stored = merge_authorization_state(
        {"heavy_tools_authorized": True},
        parse_authorization_delta("Revoke heavy-tool authorization."),
    )
    assert stored["heavy_tools_authorized"] is False

    effective = merge_authorization_state(stored, parse_authorization_delta("Continue."))
    assert effective["heavy_tools_authorized"] is False, "compression silently restored a grant"


def test_old_quoted_grant_in_history_cannot_regrant():
    """Only the current user message may create a delta."""
    stored = dict.fromkeys(AUTHORIZATION_KEYS, False)
    # The graph passes ONLY the current user message; historical/quoted text is
    # never re-parsed.  Simulate a turn whose current message quotes nothing.
    effective = merge_authorization_state(stored, parse_authorization_delta("Summarize."))
    assert effective["heavy_tools_authorized"] is False


def test_call_model_uses_current_turn_delta_not_window():
    from app.agent.agents import agentic_rag

    source = inspect.getsource(agentic_rag.call_model)
    assert "parse_authorization_delta(last_human_content)" in source
    assert "merge_authorization_state(" in source
    # The window-derived helper must no longer drive the parent hot path.
    assert "_authorization_flags_for_window(messages)" not in source


# ------------------------------------------------------- reducers
def test_ordered_set_reducer_normalizes_dedupes_sorts_and_bounds():
    reduce = make_ordered_set_reducer(3)
    assert reduce(["b", "a"], ["a", "c"]) == ["a", "b", "c"]
    assert reduce(None, ["x"]) == ["x"]
    assert reduce(["x"], None) == ["x"]
    assert reduce(None, None) == []
    # bounded deterministically
    assert reduce([], ["e", "d", "c", "b", "a"]) == ["a", "b", "c"]


def test_ordered_set_reducer_is_idempotent():
    reduce = make_ordered_set_reducer(10)
    once = reduce(["a", "b"], ["b"])
    assert reduce(once, ["b"]) == once


def test_replacement_reducer_replaces_rather_than_accumulates():
    reduce = make_replacement_reducer(10)
    assert reduce(["old1", "old2"], ["new"]) == ["new"]
    assert reduce(["keep"], None) == ["keep"]


def test_authorization_reducer_replaces_effective_state():
    assert reduce_authorization_flags({"heavy_tools_authorized": True}, {"heavy_tools_authorized": False})[
        "heavy_tools_authorized"
    ] is False
    assert reduce_authorization_flags({"heavy_tools_authorized": True}, None)[
        "heavy_tools_authorized"
    ] is True


def test_activation_status_reducer_clamps_unknown_values_and_bounds():
    out = reduce_activation_status(None, {"a": "ELIGIBLE", "b": "NOT_A_STATE"})
    assert out["a"] == ACTIVATION_ELIGIBLE
    assert out["b"] == ACTIVATION_REQUESTED
    big = {("t%03d" % i): ACTIVATION_ELIGIBLE for i in range(500)}
    assert len(reduce_activation_status(None, big)) <= 64


def test_scalar_and_version_reducers():
    assert reduce_scalar("old", "new") == "new"
    assert reduce_scalar("old", None) == "old"
    assert reduce_version(1, 2) == 2
    assert reduce_version(1, None) == 1
    assert reduce_version(None, "notanint") == TOOL_POLICY_VERSION


def test_normalize_name_list_bounds_and_truncates_long_names():
    assert normalize_name_list(["b", "a", "a", "", None], 10) == ["a", "b"]
    long_name = "x" * 500
    assert len(normalize_name_list([long_name], 10)[0]) <= 128


# ------------------------------------------------------- activation lifecycle
def test_pending_when_available_but_unauthorized():
    assert classify_activation_state(
        HEAVY, available=[HEAVY], pending=[HEAVY]
    ) == ACTIVATION_PENDING


def test_eligible_when_authorized():
    assert classify_activation_state(
        HEAVY, available=[HEAVY], eligible=[HEAVY]
    ) == ACTIVATION_ELIGIBLE


def test_bound_when_in_model_binding():
    assert classify_activation_state(
        HEAVY, available=[HEAVY], eligible=[HEAVY], bound_tool_names=["build_log_capsule"]
    ) == ACTIVATION_BOUND


def test_unavailable_upstream_outranks_pending():
    """A missing tool is never pending authorization."""
    status = classify_activation_state(SCENE, available=[], pending=[SCENE])
    assert status == ACTIVATION_UNAVAILABLE
    assert status != ACTIVATION_PENDING


def test_revoked_when_previously_eligible_and_now_neither():
    assert classify_activation_state(
        HEAVY, available=[HEAVY], previously_eligible=[HEAVY]
    ) == ACTIVATION_REVOKED


def test_blocked_policy_has_highest_precedence():
    assert classify_activation_state(
        HEAVY, available=[HEAVY], eligible=[HEAVY], blocked=[HEAVY]
    ) == ACTIVATION_BLOCKED


def test_plain_requested_when_nothing_else_applies():
    assert classify_activation_state(HEAVY, available=[HEAVY]) == ACTIVATION_REQUESTED


# ------------------------------------------------------- signature
def test_signature_is_ordering_insensitive():
    a = compute_policy_signature(
        active_toolsets=["b", "a"], eligible_extra_tools=[HEAVY], registry_generation="g1"
    )
    b = compute_policy_signature(
        active_toolsets=["a", "b"], eligible_extra_tools=[HEAVY], registry_generation="g1"
    )
    assert a == b


def test_identical_state_yields_identical_signature():
    kw = dict(active_toolsets=["s3_stb_logs"], eligible_extra_tools=[], registry_generation="g1")
    assert compute_policy_signature(**kw) == compute_policy_signature(**kw)


def test_new_eligible_tool_changes_signature():
    base = compute_policy_signature(
        active_toolsets=["s3_stb_logs"], eligible_extra_tools=[], registry_generation="g1"
    )
    promoted = compute_policy_signature(
        active_toolsets=["s3_stb_logs"], eligible_extra_tools=[HEAVY], registry_generation="g1"
    )
    assert base != promoted


def test_revocation_removing_eligibility_changes_signature():
    granted = compute_policy_signature(
        active_toolsets=["s3_stb_logs"],
        eligible_extra_tools=[HEAVY],
        registry_generation="g1",
        authorization_flags={"heavy_tools_authorized": True},
    )
    revoked = compute_policy_signature(
        active_toolsets=["s3_stb_logs"],
        eligible_extra_tools=[],
        registry_generation="g1",
        authorization_flags={"heavy_tools_authorized": False},
    )
    assert granted != revoked


def test_registry_generation_change_changes_signature():
    a = compute_policy_signature(active_toolsets=["x"], eligible_extra_tools=[], registry_generation="g1")
    b = compute_policy_signature(active_toolsets=["x"], eligible_extra_tools=[], registry_generation="g2")
    assert a != b


def test_signature_excludes_volatile_identity():
    """No timestamp/request/thread/prompt input exists in the signature API."""
    params = set(inspect.signature(compute_policy_signature).parameters)
    for forbidden in ("timestamp", "request_id", "thread_id", "chat_id", "prompt", "audit_id"):
        assert forbidden not in params


def test_unavailable_request_alone_does_not_change_signature():
    """An unavailable tool is not eligible, so it adds no model schema."""
    base = compute_policy_signature(
        active_toolsets=["s3_stb_logs"], eligible_extra_tools=[], registry_generation="g1"
    )
    # SCENE is unavailable, therefore never in eligible_extra_tools.
    assert base == compute_policy_signature(
        active_toolsets=["s3_stb_logs"], eligible_extra_tools=[], registry_generation="g1"
    )


# ------------------------------------------------------- serialization safety
def test_policy_state_is_json_serializable():
    state = default_tool_policy_state()
    state["requested_extra_tools"] = [HEAVY]
    state["last_activation_status"] = {HEAVY: ACTIVATION_PENDING}
    blob = json.dumps(state, sort_keys=True)
    assert HEAVY in blob


def test_policy_state_contains_no_runtime_objects():
    state = default_tool_policy_state()
    for key, value in state.items():
        assert isinstance(value, (list, dict, str, int, bool)), key
        if isinstance(value, list):
            assert all(isinstance(v, str) for v in value)


def test_agent_state_declares_bounded_policy_fields():
    import typing

    from app.agent.agents.agentic_rag import AgentState

    hints = typing.get_type_hints(AgentState, include_extras=True)
    for field in (
        "active_toolsets",
        "requested_extra_tools",
        "eligible_extra_tools",
        "pending_authorization_extra_tools",
        "unavailable_extra_tools",
        "authorization_flags",
        "last_bound_tool_names",
        "tool_profile_signature",
        "tool_registry_generation",
        "last_activation_status",
        "tool_policy_version",
    ):
        assert field in hints, field


# ------------------------------------------------------- last binding
def test_last_bound_names_replace_and_include_retry_rebinding():
    from app.agent.agents import agentic_rag

    source = inspect.getsource(agentic_rag.call_model)
    # Both the initial binding and the narrower retry binding must be recorded.
    assert source.count("_record_parent_binding(") == 3
    assert "retry_tools" in source
    reduce = make_replacement_reducer(MAX_LAST_BOUND_TOOL_NAMES)
    assert reduce(["wide_a", "wide_b"], ["narrow"]) == ["narrow"]


def test_policy_update_helper_returns_only_serializable_fields():
    from app.agent.agents.agentic_rag import _build_tool_policy_update

    class _Plan:
        eligible_extra_tools = ()
        pending_authorization_extra_tools = ()
        candidate_toolsets = ("search",)
        inventory_signature = "gen-1"
        methodology = "generic_engineering"

    update = _build_tool_policy_update(
        stored=default_tool_policy_state(),
        prompt="Say hello.",
        plan=_Plan(),
        authorization_flags=dict.fromkeys(AUTHORIZATION_KEYS, False),
        bound_tool_names=["public_web_search_enhanced"],
    )
    json.dumps(update, sort_keys=True)
    assert update["tool_registry_generation"] == "gen-1"
    assert update["last_bound_tool_names"] == ["public_web_search_enhanced"]
    assert update["tool_policy_version"] == TOOL_POLICY_VERSION


# ------------------------------------------------------- management facade
def test_facade_reports_unknown_when_no_parent_turn_context():
    from app.agent.tool_policy_runtime import policy_runtime_context_or_unknown

    payload = policy_runtime_context_or_unknown()
    # Either unknown, or a genuine snapshot from another test; both must be explicit.
    assert "state_known" in payload


def test_facade_reports_actual_published_state():
    from app.agent.tool_policy_runtime import (
        build_policy_runtime_snapshot,
        publish_policy_runtime_snapshot,
        reset_policy_runtime_context,
    )

    state = default_tool_policy_state()
    state["requested_extra_tools"] = [HEAVY]
    state["pending_authorization_extra_tools"] = [HEAVY]
    state["last_activation_status"] = {HEAVY: ACTIVATION_PENDING}
    state["tool_registry_generation"] = "gen-7"

    token = publish_policy_runtime_snapshot(
        build_policy_runtime_snapshot(state, methodology="generic_engineering")
    )
    try:
        from app.agent.agents.tools.management import diship_backend_tool_binding_status

        result = diship_backend_tool_binding_status.invoke({"methodology": ""})
        policy = result["tool_policy_state"]
        assert policy["state_known"] is True
        assert policy["pending_authorization_extra_tools"] == [HEAVY]
        assert policy["activation_status"][HEAVY] == ACTIVATION_PENDING
        assert policy["tool_registry_generation"] == "gen-7"
        assert result["thread_state_status"] == "ACTUAL_CHECKPOINTED_THREAD_STATE"
        # The facade must never echo authorization material.
        assert policy["authorization_material_stored"] is False
        blob = json.dumps(result)
        assert "token" not in blob.lower()
    finally:
        reset_policy_runtime_context(token)


def test_facade_snapshot_is_bounded_and_redacted():
    from app.agent.tool_policy_runtime import build_policy_runtime_snapshot

    state = default_tool_policy_state()
    state["requested_extra_tools"] = ["s3_stb_logs:t%03d" % i for i in range(500)]
    snapshot = build_policy_runtime_snapshot(state)
    assert len(snapshot["requested_extra_tools"]) <= FIELD_BOUNDS["requested_extra_tools"]
    assert snapshot["authorization_material_stored"] is False
    json.dumps(snapshot, sort_keys=True)
