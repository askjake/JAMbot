from __future__ import annotations

import json

from app.agent.identifier_validation import (
    IDENTIFIER_VALIDATION_FAILED,
    validate_tool_arguments,
)
from app.agent.tool_execution_gate import evaluate_tool_call


VALID_SCENE = "scene-480a120d3947e38f7a4a5c5f"
CONTEXT_UUID = "4a761642-6a5b-4ec8-96d8-c767f1e38a45"


def _call(name: str, args: dict, ident: str = "call-id-1") -> dict:
    return {"id": ident, "name": name, "args": args}


def test_valid_scene_id_matches_authoritative_s3_format():
    result = validate_tool_arguments({"scene_id": VALID_SCENE}, tool_call_id="c1")
    assert result.allowed is True
    assert result.validations[0].identifier_type == "scene_id"
    assert result.validations[0].validation_status == "VALID"


def test_arbitrary_uuid_shape_is_not_a_scene_id():
    result = validate_tool_arguments({"scene_id": CONTEXT_UUID}, tool_call_id="c1")
    assert result.allowed is False
    assert result.result_code == IDENTIFIER_VALIDATION_FAILED
    assert result.validations[0].validation_status == "INVALID_FORMAT"


def test_context_uuid_cannot_populate_scene_id_even_with_context_source():
    result = validate_tool_arguments(
        {"scene_id": CONTEXT_UUID},
        context_identifiers={"chat_id": CONTEXT_UUID},
        argument_sources={"scene_id": "context.chat_id"},
        tool_call_id="c1",
        audit_event_id="tea-1",
    )
    assert result.allowed is False
    item = result.validations[0]
    assert item.validation_status == "CROSS_TYPE_CONTEXT_IDENTIFIER"
    assert item.argument_source == "context.chat_id"
    assert item.source_identifier_type == "chat_id"


def test_receiver_validator_matches_authoritative_s3_contract():
    assert validate_tool_arguments({"receiver_id": "R1955706171"}).allowed
    assert validate_tool_arguments({"receiver_id": "1955706171"}).allowed
    assert not validate_tool_arguments({"receiver_id": "R123"}).allowed


def test_request_uuid_is_not_rejected_merely_for_being_uuid_shaped():
    result = validate_tool_arguments({"request_id": CONTEXT_UUID})
    assert result.allowed is True


def test_audit_metadata_never_contains_identifier_values():
    result = validate_tool_arguments(
        {"scene_id": CONTEXT_UUID},
        context_identifiers={"workspace_id": CONTEXT_UUID},
        argument_sources={"scene_id": "context.workspace_id"},
        tool_call_id="call-1",
        audit_event_id="tea-1",
    )
    text = json.dumps(result.audit_records, sort_keys=True)
    assert CONTEXT_UUID not in text
    assert "argument_value" not in text
    assert result.audit_records[0]["argument_source"] == "context.workspace_id"
    assert result.audit_records[0]["identifier_type"] == "scene_id"
    assert result.audit_records[0]["tool_call_id"] == "call-1"
    assert result.audit_records[0]["audit_event_id"] == "tea-1"


def test_execution_gate_blocks_cross_type_identifier_before_executor():
    decision = evaluate_tool_call(
        _call("query_incident_scene", {"scene_id": CONTEXT_UUID}),
        last_bound_tool_names={"query_incident_scene"},
        context_identifiers={"thread_id": CONTEXT_UUID},
        argument_sources={"scene_id": "context.thread_id"},
        audit_event_id="tea-1",
    )
    assert decision.allowed is False
    assert decision.result_code == IDENTIFIER_VALIDATION_FAILED
    assert decision.audit["identifier_validation_status"] == "BLOCKED"
    assert CONTEXT_UUID not in json.dumps(decision.audit)


def test_execution_gate_allows_valid_domain_identifier():
    decision = evaluate_tool_call(
        _call("query_incident_scene", {"scene_id": VALID_SCENE}),
        last_bound_tool_names={"query_incident_scene"},
    )
    assert decision.allowed is True
    assert decision.audit["identifier_validation_status"] == "PASS"


def test_context_fields_are_typed_but_not_treated_as_domain_fields():
    result = validate_tool_arguments({"chat_id": CONTEXT_UUID, "thread_id": "thread-abc"})
    assert result.allowed is True
    assert {v.identifier_type for v in result.validations} == {"chat_id", "thread_id"}
