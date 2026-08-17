"""Phase D3B0b: negated authorization language must never grant.

Discovered by a D3B0 live canary. The sentence "operator authorization and
heavy-tool authorization are not granted" matched the bare positive verb
"granted" and was recorded as a grant of both capabilities, so a prompt that
explicitly denied authorization elevated it instead.
"""

from __future__ import annotations

import pytest

from app.agent.tool_authorization import (
    authorization_state_for_turn,
    parse_authorization_updates,
)

DENIALS = [
    "Operator authorization and heavy-tool authorization are not granted.",
    "Operator authorization is not granted and heavy tools are not authorized.",
    "Heavy tools are not granted.",
    "Operator actions are not permitted.",
    "Persistence is not enabled.",
    "Mutation is not allowed.",
    "Heavy tools are not approved.",
    "Operator authorization was never granted.",
    "Proceed without operator authorization.",
    "Heavy tools aren't authorized.",
    "Operator authorization cannot be granted.",
    "No operator authorization is granted.",
]


@pytest.mark.parametrize("text", DENIALS)
def test_denial_never_produces_a_grant(text):
    updates = parse_authorization_updates(text)
    assert updates, f"denial produced no explicit update: {text!r}"
    assert not any(updates.values()), f"denial granted something: {text!r} -> {updates}"


def test_the_exact_canary_sentence_denies_both_capabilities():
    updates = parse_authorization_updates(
        "Operator authorization and heavy-tool authorization are not granted. "
        "Do not execute code."
    )
    assert updates.get("operator_authorized") is False
    assert updates.get("heavy_tools_authorized") is False


def test_denial_revokes_a_previously_sticky_grant():
    granted = authorization_state_for_turn(None, "Operator authorized. Heavy tools authorized.")
    flags = granted["authorization_flags"]
    assert flags["operator_authorized"] is True
    assert flags["heavy_tools_authorized"] is True

    denied = authorization_state_for_turn(
        flags, "Operator authorization and heavy-tool authorization are not granted."
    )
    assert denied["authorization_flags"]["operator_authorized"] is False
    assert denied["authorization_flags"]["heavy_tools_authorized"] is False
    assert denied["authorization_material_stored"] is False


GRANTS = [
    ("Operator authorized.", "operator_authorized"),
    ("Heavy tools are authorized.", "heavy_tools_authorized"),
    ("Operator authorization is granted.", "operator_authorized"),
    ("I grant heavy tools.", "heavy_tools_authorized"),
    ("Persistence is enabled.", "persistence_authorized"),
    ("Mutation is allowed.", "mutation_authorized"),
]


@pytest.mark.parametrize("text,key", GRANTS)
def test_genuine_grants_still_work(text, key):
    updates = parse_authorization_updates(text)
    assert updates.get(key) is True, f"{text!r} -> {updates}"


def test_contrastive_clause_still_overrides():
    updates = parse_authorization_updates(
        "Heavy tools are authorized, but persistence is not."
    )
    assert updates.get("heavy_tools_authorized") is True
    assert updates.get("persistence_authorized") is False


def test_inquiry_still_produces_no_update():
    assert parse_authorization_updates("Are heavy tools authorized?") == {}
    assert parse_authorization_updates("Tell me whether operator is authorized") == {}


def test_denial_of_one_capability_does_not_touch_another():
    updates = parse_authorization_updates("Heavy tools are not granted.")
    assert updates.get("heavy_tools_authorized") is False
    assert "mutation_authorized" not in updates


def test_code_execution_stays_blocked_under_denial_wording():
    """The D3B0 canary invariant: denial wording must not enable code execution."""
    from app.agent.tool_execution_gate import tool_is_permitted_by_flags

    state = authorization_state_for_turn(
        None, "Operator authorization and heavy-tool authorization are not granted."
    )
    assert not tool_is_permitted_by_flags("agent_run_python", state["authorization_flags"])
    assert not tool_is_permitted_by_flags("agent_run_shell", state["authorization_flags"])
