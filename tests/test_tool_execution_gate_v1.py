from __future__ import annotations

import asyncio
import json

from app.agent.tool_execution_gate import (
    evaluate_tool_call,
    execute_gated_tool_calls,
    paired_result_payload,
    prepare_model_facing_tool,
)


class FakeMessage:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeTool:
    def __init__(self, name, result=None, error=None):
        self.name = name
        self.result = result
        self.error = error
        self.seen = []

    async def ainvoke(self, args, config=None):
        self.seen.append(dict(args))
        if self.error:
            raise self.error
        return self.result


def call(name, args=None, ident="call-1"):
    return {"id": ident, "name": name, "args": args or {}}


def decode(message):
    return json.loads(message.content)



def test_model_facing_clone_hides_server_authorization_arguments_without_mutating_executor():
    from pydantic import BaseModel

    class Args(BaseModel):
        receiver_id: str
        allow_heavy: bool = False
        heavy_auth_token: str = ""
        max_files: int = 40

    class ModelTool:
        name = "build_log_capsule"
        description = "Build capsule"
        args_schema = Args

    executor_tool = ModelTool()
    view = prepare_model_facing_tool(executor_tool)
    assert view is not executor_tool
    assert set(view.args_schema.model_fields) == {"receiver_id", "max_files"}
    assert set(executor_tool.args_schema.model_fields) == {"receiver_id", "allow_heavy", "heavy_auth_token", "max_files"}
    schema_text = json.dumps(view.args_schema.model_json_schema())
    assert "heavy_auth_token" not in schema_text
    assert "allow_heavy" not in schema_text
    assert "supplied server-side" in view.description

def test_unbound_and_unauthorized_tools_are_blocked_without_arguments_in_audit():
    unbound = evaluate_tool_call(call("search_logs", {"query": "secret-value"}), last_bound_tool_names={"list_dates"})
    assert not unbound.allowed
    assert unbound.result_code == "TOOL_NOT_IN_LAST_BINDING"
    assert "secret-value" not in json.dumps(unbound.audit)

    heavy = evaluate_tool_call(call("build_incident_scene", {"capsule_id": "cap"}), last_bound_tool_names={"build_incident_scene"})
    assert not heavy.allowed
    assert heavy.missing_authorizations == ("heavy_tools_authorized",)


def test_authorized_heavy_call_rewrites_persistence_and_bounds_arguments():
    decision = evaluate_tool_call(
        call("build_log_capsule", {"max_files": 9999, "max_events": 999999, "persist": True}),
        last_bound_tool_names={"build_log_capsule"},
        authorization_flags={"heavy_tools_authorized": True, "persistence_authorized": False},
    )
    assert decision.allowed
    assert decision.result_code == "TOOL_EXECUTION_ALLOWED_WITH_REWRITE"
    assert decision.effective_args["persist"] is False
    assert decision.effective_args["max_files"] == 60
    assert decision.effective_args["max_events"] == 120000
    assert decision.effective_args["allow_heavy"] is True
    assert set(decision.rewritten_arguments) == {"allow_heavy", "max_events", "max_files", "persist"}



def test_omitted_persist_defaults_are_forced_safe_and_tokens_are_removed():
    decision = evaluate_tool_call(
        call("build_incident_scene", {"capsule_id": "cap", "heavy_auth_token": "must-never-flow"}),
        last_bound_tool_names={"build_incident_scene"},
        authorization_flags={"heavy_tools_authorized": True},
    )
    assert decision.allowed
    assert decision.effective_args["persist"] is False
    assert decision.effective_args["allow_heavy"] is True
    assert "heavy_auth_token" not in decision.effective_args
    assert "must-never-flow" not in json.dumps(decision.to_dict())


def test_always_persistent_heavy_tools_require_persistence_authorization():
    blocked = evaluate_tool_call(
        call("create_log_bundle", {"receiver_id": "R1234567"}),
        last_bound_tool_names={"create_log_bundle"},
        authorization_flags={"heavy_tools_authorized": True},
    )
    assert not blocked.allowed
    assert blocked.missing_authorizations == ("persistence_authorized",)
    allowed = evaluate_tool_call(
        call("create_log_bundle", {"receiver_id": "R1234567"}),
        last_bound_tool_names={"create_log_bundle"},
        authorization_flags={"heavy_tools_authorized": True, "persistence_authorized": True},
    )
    assert allowed.allowed
    assert allowed.effective_args["allow_heavy"] is True

def test_mutation_requires_operator_and_mutation_authorization():
    blocked = evaluate_tool_call(
        call("persist_human_response", {"payload": "x"}),
        last_bound_tool_names={"persist_human_response"},
        authorization_flags={"operator_authorized": True},
    )
    assert not blocked.allowed
    assert set(blocked.missing_authorizations) == {"mutation_authorized"}
    allowed = evaluate_tool_call(
        call("persist_human_response", {"payload": "x"}),
        last_bound_tool_names={"persist_human_response"},
        authorization_flags={"operator_authorized": True, "mutation_authorized": True},
    )
    assert allowed.allowed


def test_executor_returns_one_paired_result_per_call_and_redacts_errors():
    tools = {
        "search_logs": FakeTool("search_logs", result={"matches": 3}),
        "list_dates": FakeTool("list_dates", error=RuntimeError("https://secret token=bad")),
    }
    audits = []
    messages = asyncio.run(execute_gated_tool_calls(
        [
            call("search_logs", {"query": "x"}, "a"),
            call("not_bound", {"password": "never-audit-this"}, "b"),
            call("list_dates", {}, "c"),
        ],
        tools_by_name=tools,
        last_bound_tool_names={"search_logs", "list_dates"},
        audit_sink=audits.append,
        tool_message_class=FakeMessage,
    ))
    assert len(messages) == 3
    assert [m.tool_call_id for m in messages] == ["a", "b", "c"]
    payloads = [decode(m) for m in messages]
    assert payloads[0]["ok"] is True
    assert payloads[1]["result_code"] == "TOOL_NOT_IN_LAST_BINDING"
    assert payloads[2]["result_code"] == "TOOL_EXECUTION_ERROR"
    assert payloads[2]["error_type"] == "RuntimeError"
    assert "secret" not in messages[2].content
    assert len(audits) == 3
    assert "never-audit-this" not in json.dumps(audits)
    assert audits[1]["argument_values_recorded"] is False


def test_blocked_payload_never_claims_write():
    decision = evaluate_tool_call(call("submit_external_upload"), last_bound_tool_names={"submit_external_upload"})
    payload = paired_result_payload(decision)
    assert payload["ok"] is False
    assert payload["write_performed"] is False
