from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.agent.no_progress_controller import evaluate_no_progress
from app.agent.tool_activation_intent import RegistryToolIndex, activation_intent_from_prompt
from app.agent.tool_authorization import parse_authorization_updates
from app.agent.tool_execution_gate import evaluate_tool_call, required_authorizations_for_tool
from apps.nightly_rca.grasshopper_contract import (
    AcquisitionState,
    ProfileMetadata,
    build_grasshopper_arguments,
    determine_acquisition_state,
    parse_plan_response,
    parse_upload_response,
)
from apps.nightly_rca.state import normalize_pending_identifier_provenance


REGISTRY = RegistryToolIndex.from_material(
    inventory=[
        {"toolset": "grasshopper_mcp", "tool_name": "grasshopper_plan_profile_upload"},
        {"toolset": "grasshopper_mcp", "tool_name": "grasshopper_upload_profile_logs"},
        {"toolset": "agent_mode", "tool_name": "agent_spawn_task"},
        {"toolset": "agent_mode", "tool_name": "agent_spawn_parallel"},
    ],
    family_status={"grasshopper_mcp": "HEALTHY", "agent_mode": "LOCAL"},
)


def _all_auth() -> dict[str, bool]:
    return {
        "operator_authorized": True,
        "heavy_tools_authorized": True,
        "mutation_authorized": True,
        "persistence_authorized": True,
    }


@pytest.mark.parametrize(
    "text",
    [
        "mail me at mailto:test@example.com",
        "review https://example.invalid/path",
        "clone git@git.example:team/repo.git",
        "the example token is toolset:tool",
        "configuration uses key:value and owner:name",
        "the event happened at 12:30",
        "Do not use grasshopper_mcp:grasshopper_upload_profile_logs.",
        "Example: Bind foo_mcp:some_tool.",
        'The prompt under test is "Bind foo_mcp:some_tool".',
        "```text\nBind foo_mcp:some_tool\n```",
    ],
)
def test_incidental_or_non_authoritative_qualified_text_is_not_activation(text: str):
    intent = activation_intent_from_prompt(text, registry_index=REGISTRY)
    assert intent.requested_exact_tools == ()
    assert intent.pending_registry_requests == ()


def test_explicit_known_and_unknown_dynamic_requests_still_work():
    known = activation_intent_from_prompt(
        "Bind grasshopper_mcp:grasshopper_plan_profile_upload.", registry_index=REGISTRY
    )
    assert known.requested_exact_tools == (
        "grasshopper_mcp:grasshopper_plan_profile_upload",
    )

    unknown = activation_intent_from_prompt(
        "Bind newserver_mcp:some_tool.", registry_index=REGISTRY
    )
    assert unknown.requested_exact_tools == ("newserver_mcp:some_tool",)
    assert unknown.pending_registry_requests == ("newserver_mcp:some_tool",)


def test_multiline_exact_binding_list_is_preserved():
    intent = activation_intent_from_prompt(
        """Activate grasshopper_mcp and bind exactly:
- grasshopper_mcp:grasshopper_plan_profile_upload
- grasshopper_mcp:grasshopper_upload_profile_logs
""",
        registry_index=REGISTRY,
    )
    assert intent.requested_toolsets == ("grasshopper_mcp",)
    assert set(intent.requested_exact_tools) == {
        "grasshopper_mcp:grasshopper_plan_profile_upload",
        "grasshopper_mcp:grasshopper_upload_profile_logs",
    }



def test_multiline_bare_operational_binding_list_is_preserved():
    registry = RegistryToolIndex.from_material(
        inventory=[
            {"toolset": "agent_mode", "tool_name": "agent_run_shell"},
            {"toolset": "agent_mode", "tool_name": "agent_run_python"},
        ],
        family_status={"agent_mode": "LOCAL"},
    )
    intent = activation_intent_from_prompt(
        """Bind exactly:
- agent_run_shell
- agent_run_python
""",
        registry_index=registry,
    )
    assert set(intent.requested_exact_tools) == {
        "agent_mode:agent_run_shell",
        "agent_mode:agent_run_python",
    }

def test_canonical_authorization_accepts_only_bare_true_false():
    updates = parse_authorization_updates(
        """operator_authorized=true
heavy_tools_authorized = TRUE
mutation_authorized=false
persistence_authorized = false"""
    )
    assert updates == {
        "operator_authorized": True,
        "heavy_tools_authorized": True,
        "mutation_authorized": False,
        "persistence_authorized": False,
    }
    assert parse_authorization_updates("operator_authorized=yes") == {}
    assert parse_authorization_updates("operator_authorized=1") == {}
    assert parse_authorization_updates('operator_authorized="true"') == {}


def test_authorization_examples_quotes_and_code_do_not_mutate_state():
    text = """
Example:
operator_authorized=true
but revoke operator authorization
must result operator_authorized=false

The phrase under test is "operator authorization is granted".
```text
Heavy tools authorized.
mutation_authorized=true
```
"""
    assert parse_authorization_updates(text) == {}


def test_direct_natural_language_authorization_remains_supported_and_revocation_wins():
    assert parse_authorization_updates(
        "Operator authorized. Heavy tools authorized. Mutation authorized. Persistence authorized."
    ) == {
        "operator_authorized": True,
        "heavy_tools_authorized": True,
        "mutation_authorized": True,
        "persistence_authorized": True,
    }
    mixed = parse_authorization_updates(
        "operator_authorized=true\nRevoke operator authorization."
    )
    assert mixed["operator_authorized"] is False


def test_parent_only_constraint_filters_and_blocks_spawn_tools():
    from app.agent.execution_constraints import (
        filter_mcop_spawn_tools,
        merge_mcop_children_forbidden,
        parse_mcop_constraint_delta,
    )

    assert parse_mcop_constraint_delta("Work in the parent thread only. Do not spawn MCOP children.") == "FORBID"
    assert parse_mcop_constraint_delta("How do MCOP children work?") == "UNCHANGED"
    assert merge_mcop_children_forbidden(False, "FORBID") is True
    assert merge_mcop_children_forbidden(True, "ALLOW") is False
    assert filter_mcop_spawn_tools(
        ["agent_run_shell", "agent_spawn_task", "agent_spawn_parallel"], True
    ) == ["agent_run_shell"]

    decision = evaluate_tool_call(
        {"name": "agent_spawn_task", "id": "spawn-1", "args": {}},
        last_bound_tool_names=["agent_spawn_task"],
        authorization_flags=_all_auth(),
        execution_constraints={"mcop_children_forbidden": True},
    )
    assert decision.allowed is False
    assert decision.result_code == "BLOCKED_PARENT_ONLY_TURN"


@dataclass
class HumanMessage:
    content: str


@dataclass
class ToolMessage:
    content: str
    name: str = ""


def test_no_progress_ignores_unrelated_child_or_phantom_failures():
    messages = [
        HumanMessage("Continue in the parent thread."),
        ToolMessage(
            '{"ok": false, "result_code": "TOOL_NOT_IN_LAST_BINDING", "tool_name": "agent_spawn_task"}',
            name="agent_spawn_task",
        ),
        ToolMessage(
            '{"ok": false, "result_code": "TOOL_EXECUTION_ERROR", "tool_name": "mailto:git"}',
            name="mailto:git",
        ),
    ]
    decision = evaluate_no_progress(messages, required_tools={"agent_mode:agent_run_shell"})
    assert decision.stop is False
    assert decision.no_progress_attempts == 0
    assert "git" not in decision.missing_tools


def test_no_progress_still_stops_for_real_current_required_tool():
    messages = [
        HumanMessage("Run the shell check."),
        ToolMessage(
            '{"ok": false, "result_code": "TOOL_NOT_IN_LAST_BINDING", "tool_name": "agent_run_shell"}',
            name="agent_run_shell",
        ),
    ]
    decision = evaluate_no_progress(messages, required_tools={"agent_mode:agent_run_shell"})
    assert decision.stop is True
    assert decision.missing_tools == ("agent_run_shell",)


@pytest.mark.parametrize(
    "tool_name",
    [
        "grasshopper_get_upload_history",
        "grasshopper_get_upload_request_status",
        "grasshopper_classify_profile_upload_request",
        "grasshopper_probe_upload_status_endpoints",
        "grasshopper_introspect_upload_api",
        "grasshopper_plan_profile_upload",
    ],
)
def test_grasshopper_read_only_metadata_is_not_lexically_mutation_gated(tool_name: str):
    assert required_authorizations_for_tool(tool_name) == ()


def test_upload_arguments_carry_explicit_live_write_confirmation():
    meta = ProfileMetadata(issue_profile="nal", grasshopper_profile="nal")
    preview = build_grasshopper_arguments(meta, "R1955706171", dry_run=True)
    live = build_grasshopper_arguments(meta, "R1955706171", dry_run=False)
    assert preview["dry_run"] is True
    assert preview["allow_live_upload"] is False
    assert live["dry_run"] is False
    assert live["allow_live_upload"] is True


def test_protocol_406_4002_is_typed_non_retryable_protocol_error():
    upload = parse_upload_response(
        {
            "status": "OK",
            "response": {
                "status": "error",
                "http_status": 406,
                "code": 4002,
                "error": "JSON Not readable",
            },
        }
    )
    assert upload.write_state == "protocol_error"
    assert upload.http_status == 406
    assert upload.upstream_code == "4002"
    assert upload.request_created is False
    assert upload.automatic_retry is False
    assert upload.retry_policy == "manual_review"

    plan = parse_plan_response(
        {
            "status": "OK",
            "response": {
                "status": "success",
                "requested_profile": "nal",
                "resolved_profile": "nal",
                "profile_alias_used": False,
                "receiver_id": "R1955706171",
                "plan": {"selected_file_count": 10, "selected_file_ids": list(range(10))},
            },
        },
        profile_metadata=ProfileMetadata(issue_profile="nal", grasshopper_profile="nal"),
    )
    assert determine_acquisition_state(plan, upload, dry_run=False) == AcquisitionState.UPLOAD_PROTOCOL_ERROR


def test_profile_resolution_evidence_is_preserved_and_mismatch_blocks():
    matching = parse_plan_response(
        {
            "status": "OK",
            "response": {
                "status": "success",
                "requested_profile": "nal",
                "resolved_profile": "nal",
                "profile_alias_used": False,
                "plan": {"selected_file_count": 1, "selected_file_ids": [408]},
            },
        },
        profile_metadata=ProfileMetadata(issue_profile="nal", grasshopper_profile="nal"),
    )
    assert matching.requested_profile == "nal"
    assert matching.resolved_profile == "nal"
    assert matching.profile_alias_used is False
    assert matching.profile_identity_matches is True

    mismatch = parse_plan_response(
        {
            "status": "OK",
            "response": {
                "status": "success",
                "requested_profile": "nal",
                "resolved_profile": "atv_core",
                "profile_alias_used": True,
                "plan": {"selected_file_count": 1, "selected_file_ids": [408]},
            },
        },
        profile_metadata=ProfileMetadata(issue_profile="nal", grasshopper_profile="nal"),
    )
    assert mismatch.profile_identity_matches is False
    assert determine_acquisition_state(mismatch, None, dry_run=True) == AcquisitionState.UPLOAD_PLAN_FAILED


def test_legacy_requested_log_types_normalize_boundedly_without_rewriting_raw():
    normalized = normalize_pending_identifier_provenance(
        {
            "request_id": "14054596881153157",
            "requested_log_types": ['["nal"]'],
            "receipt_status": "submitted",
            "workflow_status": "WAITING_FOR_LOGS",
        },
        current_run_id="run-current",
    )
    assert normalized["raw_requested_log_types"] == ['["nal"]']
    assert normalized["normalized_requested_log_types"] == ["nal"]
    assert normalized["requested_log_types"] == ['["nal"]']
    assert normalized["normalized_pending_state"] == "UPLOAD_REQUESTED_PENDING_RECEIPT"
    assert normalized["identifier_provenance"] == "LEGACY_UNTYPED_REQUEST_ID"


def test_completion_contract_detects_missing_fields_and_ignores_normal_prompts():
    from app.agent.completion_contract import extract_completion_contract

    assert not extract_completion_contract("Summarize the repository.").active
    contract = extract_completion_contract(
        """ABSOLUTE COMPLETION GATE
JAKE_TESTS_AFTER=<count>
CLEAN_ROOM=<PASS|FAIL>
MR_READY=<true|false>
"""
    )
    assert contract.required_fields == ("JAKE_TESTS_AFTER", "CLEAN_ROOM", "MR_READY")
    assert contract.missing_from("JAKE_TESTS_AFTER=10\nCLEAN_ROOM=PASS") == ("MR_READY",)


def test_completion_contract_incomplete_message_is_explicit_and_bounded():
    from app.agent.completion_contract import render_incomplete_contract

    text = render_incomplete_contract(["CLEAN_ROOM", "MR_READY"])
    assert text.startswith("INCOMPLETE_EXECUTION_CONTRACT")
    assert "CLEAN_ROOM" in text and "MR_READY" in text
    assert len(text) < 4000


def test_duplicate_preflight_fails_closed_and_blocks_equivalent_legacy_tracker():
    from apps.nightly_rca.state import evaluate_duplicate_preflight

    unavailable = evaluate_duplicate_preflight(
        [], inventory_available=False, receiver_id="R1955706171", requested_log_types=["nal"]
    )
    assert unavailable["status"] == "INCOMPLETE"

    blocked = evaluate_duplicate_preflight(
        [
            {
                "tracker_id": "legacy-nal",
                "receiver_id": "R1955706171",
                "requested_log_types": ['["nal"]'],
                "receipt_status": "submitted",
                "workflow_status": "WAITING_FOR_LOGS",
            }
        ],
        inventory_available=True,
        receiver_id="R1955706171",
        requested_log_types=["nal"],
        grasshopper_profile="nal",
    )
    assert blocked["status"] == "BLOCK"
    assert blocked["relevant_pending_tracker_ids"] == ["legacy-nal"]

    passed = evaluate_duplicate_preflight(
        [
            {
                "tracker_id": "done-nal",
                "receiver_id": "R1955706171",
                "requested_log_types": ["nal"],
                "receipt_status": "receipt_complete",
                "workflow_status": "COMPLETE",
            }
        ],
        inventory_available=True,
        receiver_id="R1955706171",
        requested_log_types=["nal"],
    )
    assert passed["status"] == "PASS"


def test_report_and_acceptance_templates_are_not_authorization_directives():
    text = """
FINAL REQUIRED REPORT
operator_authorized=true
heavy_tools_authorized=false

ACCEPTANCE CRITERIA:
Mutation authorization is granted.
"""
    assert parse_authorization_updates(text) == {}


def test_no_progress_supports_real_non_mcp_family_names():
    messages = [
        HumanMessage("Run the required viewership query."),
        ToolMessage(
            '{"ok": false, "result_code": "TOOL_NOT_IN_LAST_BINDING", "tool_name": "query_viewership"}',
            name="query_viewership",
        ),
    ]
    decision = evaluate_no_progress(
        messages, required_tools={"viewership:query_viewership"}
    )
    assert decision.stop is True
    assert decision.missing_tools == ("query_viewership",)


def test_completion_contract_rejects_unchanged_placeholders_but_accepts_honest_unknown():
    from app.agent.completion_contract import extract_completion_contract

    contract = extract_completion_contract(
        """FINAL REQUIRED REPORT
PATCH_STATUS=<complete|blocked>
CLEAN_ROOM=<PASS|FAIL|not_run>
"""
    )
    assert contract.missing_from(
        "PATCH_STATUS=<complete|blocked>\nCLEAN_ROOM=<PASS|FAIL|not_run>"
    ) == ("PATCH_STATUS", "CLEAN_ROOM")
    assert contract.missing_from(
        "PATCH_STATUS=blocked\nCLEAN_ROOM=not_run"
    ) == ()


@pytest.mark.parametrize(
    "tool_name",
    [
        "grasshopper_health",
        "grasshopper_get_log_type_catalog",
        "grasshopper_get_resource",
        "grasshopper_list_file_groups",
        "grasshopper_list_files",
        "grasshopper_list_uploadable_files",
    ],
)
def test_source_evidenced_grasshopper_inventory_reads_are_read_only(tool_name: str):
    assert required_authorizations_for_tool(tool_name) == ()


@pytest.mark.parametrize(
    "tool_name",
    [
        "grasshopper_upload_file",
        "grasshopper_batch_upload",
        "grasshopper_upload",
        "grasshopper_ccshare_upload",
    ],
)
def test_source_evidenced_grasshopper_writes_remain_gated(tool_name: str):
    assert set(required_authorizations_for_tool(tool_name)) == {
        "operator_authorized",
        "mutation_authorized",
        "persistence_authorized",
    }


def test_protocol_error_is_detected_in_executor_and_nested_envelopes():
    executor_error = parse_upload_response(
        {
            "status": "ERROR",
            "error": "HTTP 406 upstream code 4002: JSON Not readable",
        }
    )
    assert executor_error.write_state == "protocol_error"
    assert executor_error.request_created is False
    assert executor_error.automatic_retry is False

    nested = parse_upload_response(
        {
            "status": "OK",
            "response": {
                "status": "error",
                "response": {
                    "status": "error",
                    "http_status": 406,
                    "error_code": 4002,
                    "message": "JSON Not readable",
                },
            },
        }
    )
    assert nested.write_state == "protocol_error"
    assert nested.http_status == 406
    assert nested.upstream_code == "4002"


def test_checkpoint_migration_drops_legacy_phantoms_but_keeps_explicit_pending():
    from app.agent.tool_activation_intent import activation_intent_from_checkpoint

    legacy = activation_intent_from_checkpoint(
        {
            "requested_extra_tools": [
                "mailto:git",
                "toolset:tool",
                "grasshopper_mcp:grasshopper_plan_profile_upload",
            ],
            "activation_request_revision": 7,
        },
        registry_index=REGISTRY,
    )
    assert legacy.requested_exact_tools == (
        "grasshopper_mcp:grasshopper_plan_profile_upload",
    )

    explicit = activation_intent_from_checkpoint(
        {
            "requested_extra_tools": ["newserver_mcp:some_tool"],
            "pending_registry_requests": ["newserver_mcp:some_tool"],
            "activation_request_revision": 8,
        },
        registry_index=REGISTRY,
    )
    assert explicit.requested_exact_tools == ("newserver_mcp:some_tool",)
    assert explicit.pending_registry_requests == ("newserver_mcp:some_tool",)


def test_agentic_completion_repair_uses_nonpersisted_human_followup():
    from pathlib import Path

    source = Path("app/agent/agents/agentic_rag.py").read_text()
    assert "repair_prompt = HumanMessage" in source
    assert "render_incomplete_contract(still_missing)" in source


def test_parent_only_constraint_is_copied_to_child_audit_scope():
    from app.agent.tool_execution_audit import (
        AuditRequestState,
        derive_child_audit_request_state,
    )

    parent = AuditRequestState(
        execution_constraints={"mcop_children_forbidden": True}
    )
    child = derive_child_audit_request_state(parent, child_run_id="child-1")
    assert child.execution_constraints == {"mcop_children_forbidden": True}


def test_legacy_log_type_normalization_is_depth_and_size_bounded():
    from apps.nightly_rca.state import normalize_requested_log_types

    deep = [[[[[["nal"]]]]]]
    assert normalize_requested_log_types(deep, max_depth=2) == []
    many = [f"type_{index}" for index in range(100)]
    normalized = normalize_requested_log_types(many, max_items=8)
    assert normalized == [f"type_{index}" for index in range(8)]
    assert normalize_requested_log_types({"nal", "qt_gui"}) == ["nal", "qt_gui"]


def test_new_checkpoint_fields_advance_tool_policy_version():
    from app.agent.tool_policy_state import TOOL_POLICY_VERSION, default_tool_policy_state

    assert TOOL_POLICY_VERSION >= 3
    state = default_tool_policy_state()
    assert "pending_registry_requests" in state
    assert "mcop_children_forbidden" in state
    assert state["tool_policy_version"] == TOOL_POLICY_VERSION


def test_pending_registry_and_unavailable_states_remain_distinct():
    from types import SimpleNamespace
    from app.agent.tool_policy_state import default_tool_policy_state
    from app.agent.tool_policy_transition import build_tool_policy_update

    pending_name = "newserver_mcp:some_tool"
    unavailable_name = "grasshopper_mcp:not_a_real_tool"
    plan = SimpleNamespace(
        methodology="generic_engineering",
        requested_toolsets=(),
        requested_extra_tools=(pending_name, unavailable_name),
        eligible_extra_tools=(),
        pending_authorization_extra_tools=(),
        pending_registry_requests=(pending_name,),
        unavailable_extra_tools=(unavailable_name,),
        unhealthy_requests=(),
        candidate_toolsets=("agent_mode", "backend_facades", "search"),
        inventory_signature="sha256:test",
        activation_request_revision=1,
        continuity_task_scope="",
        continuity_environment="",
        continuity_revision=0,
        mcop_children_forbidden=False,
    )
    update = build_tool_policy_update(
        stored=default_tool_policy_state(),
        plan=plan,
        authorization_flags=_all_auth(),
        bound_tool_names=[],
        activation_request_revision=1,
    )
    assert update["pending_registry_requests"] == [pending_name]
    assert update["unavailable_extra_tools"] == [unavailable_name]
    assert update["last_activation_status"][pending_name] == "PENDING_REGISTRY"
    assert update["last_activation_status"][unavailable_name] == "UNAVAILABLE_UPSTREAM"


def test_completion_contract_ignores_fenced_and_labeled_examples():
    from app.agent.completion_contract import extract_completion_contract

    fenced = extract_completion_contract(
        """Example only:
```text
ABSOLUTE COMPLETION GATE
MR_READY=<true|false>
```
Summarize the design.
"""
    )
    assert not fenced.active

    labeled = extract_completion_contract(
        """Example:
ABSOLUTE COMPLETION GATE
MR_READY=<true|false>

Continue normal analysis.
"""
    )
    assert not labeled.active


def test_parent_only_constraint_persists_and_can_be_explicitly_released():
    from app.agent.tool_profiles import build_tool_profile

    first = build_tool_profile(
        methodology="generic_engineering",
        methodology_toolsets=("agent_mode",),
        prompt="Work in the parent thread only.",
        authorization_flags=_all_auth(),
        registry_index=REGISTRY,
    )
    assert first.mcop_children_forbidden is True

    second = build_tool_profile(
        methodology="generic_engineering",
        methodology_toolsets=("agent_mode",),
        prompt="Continue.",
        authorization_flags=_all_auth(),
        registry_index=REGISTRY,
        prior_mcop_children_forbidden=first.mcop_children_forbidden,
    )
    assert second.mcop_children_forbidden is True

    released = build_tool_profile(
        methodology="generic_engineering",
        methodology_toolsets=("agent_mode",),
        prompt="You may use child agents now.",
        authorization_flags=_all_auth(),
        registry_index=REGISTRY,
        prior_mcop_children_forbidden=second.mcop_children_forbidden,
    )
    assert released.mcop_children_forbidden is False


def test_parent_only_constraint_removes_spawn_tools_from_model_facing_tray(monkeypatch):
    import sys
    import types
    from types import SimpleNamespace
    from app.agent.tool_execution_policy import get_scoped_tools_for_prompt

    fake_tools = types.ModuleType("app.agent.agents.tools")
    inventory = {
        "agent_mode": [
            SimpleNamespace(name="agent_check_tasks", description=""),
            SimpleNamespace(name="agent_spawn_task", description=""),
            SimpleNamespace(name="agent_spawn_parallel", description=""),
        ],
        "search": [],
        "backend_facades": [],
        "dish_internal": [],
    }
    fake_tools.get_tools_set = lambda family: list(inventory.get(family, []))
    fake_tools.get_tools_set_filtered = lambda family, allowed: [
        tool for tool in inventory.get(family, []) if tool.name in set(allowed)
    ]
    monkeypatch.setitem(sys.modules, "app.agent.agents.tools", fake_tools)

    tools, plan = get_scoped_tools_for_prompt(
        "Work in the parent thread only.",
        authorization_flags=_all_auth(),
        registry_index=REGISTRY,
    )
    names = {tool.name for tool in tools}
    assert plan.mcop_children_forbidden is True
    assert "agent_check_tasks" in names
    assert "agent_spawn_task" not in names
    assert "agent_spawn_parallel" not in names


def test_canonical_authorization_is_wired_through_checkpoint_delta_path():
    from app.agent.tool_policy_state import (
        merge_authorization_state,
        parse_authorization_delta,
    )

    delta = parse_authorization_delta(
        """operator_authorized=true
heavy_tools_authorized=true
mutation_authorized=true
persistence_authorized=true"""
    )
    merged = merge_authorization_state({}, delta)
    assert merged == _all_auth()

    no_delta = parse_authorization_delta(
        """Example:
operator_authorized=false
Expected output only."""
    )
    assert no_delta.is_empty()


def test_operational_exact_tool_request_stays_bound_across_methodology_change():
    from app.agent.tool_profiles import build_tool_profile

    registry = RegistryToolIndex.from_material(
        inventory=[
            {"toolset": "agent_mode", "tool_name": "agent_run_shell"},
            {"toolset": "agent_mode", "tool_name": "agent_run_python"},
        ],
        family_status={"agent_mode": "LOCAL"},
    )
    first = build_tool_profile(
        methodology="repo_checkout_local_deploy",
        methodology_toolsets=("agent_mode",),
        prompt="Bind agent_mode:agent_run_shell.",
        authorization_flags=_all_auth(),
        registry_index=registry,
    )
    assert "agent_mode:agent_run_shell" in first.eligible_extra_tools

    second = build_tool_profile(
        methodology="repo_code_review",
        methodology_toolsets=("agent_mode",),
        prompt="Continue the review.",
        prior_active_toolsets=first.active_toolsets,
        prior_extra_tools=first.requested_extra_tools,
        prior_requested_toolsets=first.requested_toolsets,
        prior_operational_workflow=first.operational_workflow,
        authorization_flags=_all_auth(),
        registry_index=registry,
        prior_activation_request_revision=first.activation_request_revision,
    )
    assert "agent_mode:agent_run_shell" in second.requested_extra_tools
    assert "agent_mode:agent_run_shell" in second.eligible_extra_tools
    assert second.operational_workflow == first.operational_workflow


def test_legacy_submitted_tracker_is_a_bounded_reconciliation_candidate():
    from datetime import datetime, timezone
    from apps.nightly_rca.pending_selector import select_pending_investigations

    normalized = normalize_pending_identifier_provenance(
        {
            "tracker_id": "legacy-nal",
            "receiver_id": "R1955706171",
            "issue_profile": "guide_1031",
            "requested_log_types": ['["nal"]'],
            "receipt_status": "submitted",
            "workflow_status": "WAITING_FOR_LOGS",
        },
        current_run_id="run-current",
    )
    selected = select_pending_investigations(
        [normalized], batch_size=1, now=datetime(2026, 8, 7, tzinfo=timezone.utc)
    )
    assert [row["tracker_id"] for row in selected] == ["legacy-nal"]
    assert selected[0]["raw_receipt_status"] == "submitted"
    assert selected[0]["normalized_pending_state"] == "UPLOAD_REQUESTED_PENDING_RECEIPT"


@pytest.mark.parametrize(
    "text",
    [
        "Bind mailto:test.",
        "Bind toolset:tool.",
        "Bind key:value.",
        "Bind owner:name.",
    ],
)
def test_explicit_semantic_nonfamilies_do_not_enter_pending_registry(text: str):
    intent = activation_intent_from_prompt(text, registry_index=REGISTRY)
    assert intent.requested_exact_tools == ()
    assert intent.pending_registry_requests == ()


def test_current_parent_required_tools_excludes_sticky_checkpoint_only_requests():
    from app.agent.no_progress_controller import current_parent_required_tools

    required = current_parent_required_tools(
        prompt_requested=("agent_mode:agent_run_shell",),
        management_requested=(),
        operational_required=(),
        first_tool="",
    )
    assert required == {"agent_run_shell"}
    # A checkpoint-only Grasshopper request is deliberately absent because it
    # was not supplied as current-turn prompt/management/operational intent.
    assert "grasshopper_upload_profile_logs" not in required


def test_child_scope_failure_for_same_required_tool_does_not_poison_parent():
    messages = [
        HumanMessage("Continue the parent shell task."),
        ToolMessage(
            '{"ok": false, "scope": "mcop_child", '
            '"result_code": "TOOL_NOT_IN_LAST_BINDING", '
            '"tool_name": "agent_run_shell"}',
            name="agent_run_shell",
        ),
    ]
    decision = evaluate_no_progress(
        messages, required_tools={"agent_mode:agent_run_shell"}
    )
    assert decision.stop is False
    assert decision.no_progress_attempts == 0


def test_parent_only_gate_has_distinct_audit_result_code():
    from app.agent.tool_execution_audit import (
        DECISION_BLOCKED,
        RESULT_BLOCKED_PARENT_ONLY_TURN,
        classify_gate_decision,
    )

    decision, result = classify_gate_decision(
        gate_result_code="BLOCKED_PARENT_ONLY_TURN",
        allowed=False,
    )
    assert decision == DECISION_BLOCKED
    assert result == RESULT_BLOCKED_PARENT_ONLY_TURN


def test_completion_contract_heading_accepts_colon():
    from app.agent.completion_contract import extract_completion_contract

    contract = extract_completion_contract(
        "FINAL REQUIRED REPORT:\nPATCH_STATUS=<complete|blocked>\n"
    )
    assert contract.required_fields == ("PATCH_STATUS",)


def test_corrupt_pending_ledger_is_not_authoritative_empty_inventory(tmp_path):
    from apps.nightly_rca.state import RunStore

    store = RunStore(tmp_path, "run-corrupt-ledger")
    store.pending_path.write_text("{not-json", encoding="utf-8")
    assert store.load_pending_investigations() == []
    assert store.pending_inventory_available is False
    assert store.pending_load_status == "ERROR"
    assert store.pending_load_error_class == "JSONDecodeError"


def test_missing_pending_ledger_is_valid_empty_local_inventory(tmp_path):
    from apps.nightly_rca.state import RunStore

    store = RunStore(tmp_path, "run-no-ledger")
    assert store.load_pending_investigations() == []
    assert store.pending_inventory_available is True
    assert store.pending_load_status == "NOT_PRESENT"



def test_executor_failure_is_acceptance_unknown_not_business_rejection():
    upload = parse_upload_response(
        {"status": "STEP_FAILED", "error": "TimeoutError: upstream transport closed"}
    )
    assert upload.write_state == "executor_failed"
    assert upload.request_created is None
    assert determine_acquisition_state(
        parse_plan_response(
            {
                "status": "OK",
                "response": {
                    "status": "success",
                    "requested_profile": "nal",
                    "resolved_profile": "nal",
                    "plan": {"selected_file_count": 1, "selected_file_ids": [408]},
                },
            },
            profile_metadata=ProfileMetadata(issue_profile="nal", grasshopper_profile="nal"),
        ),
        upload,
        dry_run=False,
    ) == AcquisitionState.UPLOAD_ACCEPTANCE_UNKNOWN


def test_public_upload_summary_omits_arbitrary_error_body():
    upload = parse_upload_response(
        {
            "status": "OK",
            "response": {
                "status": "error",
                "http_status": 406,
                "code": 4002,
                "error": "JSON Not readable SECRET-SENTINEL-DO-NOT-PERSIST",
            },
        }
    )
    public = upload.to_public_dict()
    assert public["error_class"] == "UPLOAD_PROTOCOL_ERROR"
    assert "SECRET-SENTINEL" not in repr(public)
    assert "error" not in public


def test_public_plan_summary_is_bounded_and_nonsecret():
    plan = parse_plan_response(
        {
            "status": "OK",
            "response": {
                "status": "success",
                "requested_profile": "nal",
                "resolved_profile": "nal",
                "plan": {
                    "selected_file_count": 500,
                    "selected_file_ids": list(range(500)),
                    "warnings": [f"warning-{i}" for i in range(100)],
                },
            },
        },
        profile_metadata=ProfileMetadata(issue_profile="nal", grasshopper_profile="nal"),
    )
    public = plan.to_public_dict()
    assert len(public["selected_file_ids"]) == 200
    assert len(public["warnings"]) == 32
    assert "raw" not in public


def test_completion_response_text_handles_provider_blocks():
    from app.agent.completion_contract import completion_response_text, extract_completion_contract

    contract = extract_completion_contract(
        "FINAL REQUIRED REPORT\nPATCH_STATUS=<complete|blocked>\nCLEAN_ROOM=<PASS|FAIL>"
    )
    content = [
        {"type": "text", "text": "PATCH_STATUS=complete"},
        {"type": "text", "text": "CLEAN_ROOM=PASS"},
    ]
    assert contract.missing_from(completion_response_text(content)) == ()


def test_agentic_no_progress_uses_current_parent_intent_not_all_sticky_eligible_tools():
    from pathlib import Path

    source = Path("app/agent/agents/agentic_rag.py").read_text()
    block = source[source.index("_current_operational = detect_operational_workflow"):
                   source.index("_no_progress = evaluate_no_progress")]
    assert "current_parent_required_tools(" in block
    assert "_prompt_activation.requested_exact_tools" in block
    assert "_management_activation.requested_exact_tools" in block
    assert "tool_plan.eligible_extra_tools" not in block


# v2.2 live-canary regressions (2026-08-07)
@dataclass
class AISeparatorMessage:
    content: str = ""


def test_no_progress_accumulates_sequential_parent_failures_across_ai_messages():
    """Builds exactly `ceiling` sequential AI/Tool failure pairs.

    2026-08-18: was hardcoded to exactly 2 messages/attempts, which assumed
    the pre-tuning MAX_CONSECUTIVE_TOOL_ERRORS ceiling of 2. The ceiling is
    now a tunable config value (4 as of the 2026-08-18 tuning pass), so this
    test derives the message count from settings instead of a literal, to
    avoid going stale again on the next tuning change.
    """
    from app.config import get_settings

    ceiling = int(get_settings().MAX_CONSECUTIVE_TOOL_ERRORS)
    messages = [HumanMessage("Inspect the tracker read-only.")]
    for i in range(ceiling):
        messages.append(AISeparatorMessage(f"Trying bounded read path #{i + 1}."))
        messages.append(
            ToolMessage(
                '{"ok": false, "result_code": "TOOL_NOT_IN_LAST_BINDING", "tool_name": "internal_search"}',
                name="internal_search",
            )
        )
    decision = evaluate_no_progress(
        messages,
        required_tools={"s3_stb_logs:list_pending_investigations"},
    )
    assert decision.stop is True
    assert decision.result_code == "BLOCKED_NO_PROGRESS_LIMIT"
    assert decision.no_progress_attempts == ceiling
    assert decision.missing_tools == ("internal_search",)


def test_completion_contract_is_injected_as_server_instruction_before_initial_model_call():
    from pathlib import Path

    source = Path("app/agent/agents/agentic_rag.py").read_text(encoding="utf-8")
    extract_pos = source.index("_completion_contract = extract_completion_contract(last_human_content)")
    invoke_pos = source.index("response = await model_with_tools.ainvoke", extract_pos)
    assert extract_pos < invoke_pos
    block = source[extract_pos:invoke_pos]
    assert "render_completion_contract_system_prompt" in block
    assert "completion_policy_prompt" in block


def test_completion_contract_server_instruction_names_required_fields():
    from app.agent.completion_contract import (
        extract_completion_contract,
        render_completion_contract_system_prompt,
    )

    contract = extract_completion_contract(
        "FINAL REQUIRED REPORT:\nPHASE_A=<value>\nPHASE_B=<value>\nDONE=<true|false>\n"
    )
    text = render_completion_contract_system_prompt(contract)
    assert "SERVER_ENFORCED_COMPLETION_CONTRACT" in text
    assert "PHASE_A" in text
    assert "PHASE_B" in text
    assert "DONE" in text
