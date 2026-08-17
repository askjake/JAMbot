from app.agent.tool_authorization import (
    authorization_state_for_turn,
    merge_authorization_flags,
    parse_authorization_updates,
)


def test_explicit_same_turn_grants_and_combined_grants():
    assert parse_authorization_updates("Operator authorized. Heavy tools authorized.") == {
        "operator_authorized": True,
        "heavy_tools_authorized": True,
    }
    assert parse_authorization_updates("Authorize persistence and mutations.") == {
        "persistence_authorized": True,
        "mutation_authorized": True,
    }


def test_negative_and_revoke_win_without_false_discussion_grants():
    assert parse_authorization_updates("Heavy tools are not authorized.") == {"heavy_tools_authorized": False}
    assert parse_authorization_updates("Revoke operator authorization and disable persistence.") == {
        "operator_authorized": False,
        "persistence_authorized": False,
    }
    assert parse_authorization_updates("Operator authorization is required before use.") == {}
    assert parse_authorization_updates("Do not authorize heavy tools, but authorize persistence.") == {
        "heavy_tools_authorized": False,
        "persistence_authorized": True,
    }


def test_flags_are_sticky_except_explicit_updates_and_no_material_is_stored():
    current = {"operator_authorized": True, "heavy_tools_authorized": True}
    merged = merge_authorization_flags(current, {"heavy_tools_authorized": False})
    assert merged["operator_authorized"] is True
    assert merged["heavy_tools_authorized"] is False
    state = authorization_state_for_turn(merged, "Persistence authorized.")
    assert state["authorization_flags"]["operator_authorized"] is True
    assert state["authorization_flags"]["persistence_authorized"] is True
    assert state["authorization_material_stored"] is False
