from __future__ import annotations

from dataclasses import dataclass
import json

from app.agent.tool_activation_intent import (
    SOURCE_CHECKPOINTED_STATE,
    SOURCE_CURRENT_USER_PROMPT,
    SOURCE_MANAGEMENT_FACADE_RESULT,
    RegistryToolIndex,
    ToolActivationIntent,
    activation_intent_from_management_messages,
    activation_intent_from_prompt,
    merge_activation_intents,
)
from app.agent.tool_profiles import build_tool_profile


REGISTRY = RegistryToolIndex.from_material(
    inventory=[
        {"toolset": "fake_grasshopper_mcp", "tool_name": "plan_logs", "enabled": True},
        {"toolset": "fake_grasshopper_mcp", "tool_name": "upload_logs", "enabled": True},
        {"toolset": "other_mcp", "tool_name": "plan_logs", "enabled": True},
    ],
    family_status={
        "fake_grasshopper_mcp": "HEALTHY",
        "other_mcp": "HEALTHY",
    },
)


def _all_auth() -> dict[str, bool]:
    return {
        "operator_authorized": True,
        "heavy_tools_authorized": True,
        "mutation_authorized": True,
        "persistence_authorized": True,
    }


def test_same_turn_exact_dynamic_tools_bind_when_authorized():
    profile = build_tool_profile(
        methodology="generic_engineering",
        methodology_toolsets=(),
        prompt=(
            "I authorize operator actions, heavy tools, mutation, and persistence. "
            "Bind fake_grasshopper_mcp:plan_logs and fake_grasshopper_mcp:upload_logs."
        ),
        authorization_flags=_all_auth(),
        registry_index=REGISTRY,
    )
    assert profile.requested_extra_tools == (
        "fake_grasshopper_mcp:plan_logs",
        "fake_grasshopper_mcp:upload_logs",
    )
    assert profile.eligible_extra_tools == profile.requested_extra_tools
    assert profile.curated_tools_by_toolset["fake_grasshopper_mcp"] == (
        "plan_logs",
        "upload_logs",
    )


def test_exact_tool_request_does_not_bind_entire_family():
    profile = build_tool_profile(
        methodology="generic_engineering",
        methodology_toolsets=(),
        prompt="Bind fake_grasshopper_mcp:plan_logs",
        authorization_flags=_all_auth(),
        registry_index=REGISTRY,
    )
    assert profile.requested_toolsets == ()
    assert profile.curated_tools_by_toolset["fake_grasshopper_mcp"] == ("plan_logs",)


def test_explicit_toolset_request_can_bind_bounded_family_tools():
    profile = build_tool_profile(
        methodology="generic_engineering",
        methodology_toolsets=(),
        prompt="Activate fake_grasshopper_mcp.",
        authorization_flags=_all_auth(),
        registry_index=REGISTRY,
    )
    assert profile.requested_toolsets == ("fake_grasshopper_mcp",)
    assert "fake_grasshopper_mcp" in profile.active_toolsets
    assert "fake_grasshopper_mcp" not in profile.curated_tools_by_toolset


def test_unqualified_dynamic_tool_must_be_unique():
    intent = activation_intent_from_prompt("Bind plan_logs", registry_index=REGISTRY)
    assert intent.requested_exact_tools == ()
    assert intent.ambiguous_requests == {
        "plan_logs": (
            "fake_grasshopper_mcp:plan_logs",
            "other_mcp:plan_logs",
        )
    }


def test_qualified_dynamic_tool_resolves_against_live_registry():
    intent = activation_intent_from_prompt(
        "Bind fake_grasshopper_mcp:upload_logs", registry_index=REGISTRY
    )
    assert intent.requested_exact_tools == ("fake_grasshopper_mcp:upload_logs",)
    assert intent.source == SOURCE_CURRENT_USER_PROMPT


@dataclass
class ToolMessage:
    name: str
    content: str


def test_management_activation_result_is_merged_into_checkpoint_intent():
    payload = {
        "schema": "diship_backend_tool_activation_request.v1",
        "requested_toolsets": ["fake_grasshopper_mcp"],
        "requested_extra_tools": ["fake_grasshopper_mcp:plan_logs"],
        "activation_performed": False,
        "write_performed": False,
    }
    messages = [ToolMessage("diship_backend_activate_tool_binding", json.dumps(payload))]
    intent = activation_intent_from_management_messages(messages, registry_index=REGISTRY)
    assert intent.source == SOURCE_MANAGEMENT_FACADE_RESULT
    assert intent.requested_toolsets == ("fake_grasshopper_mcp",)
    assert intent.requested_exact_tools == ("fake_grasshopper_mcp:plan_logs",)


def test_assistant_authored_json_is_not_treated_as_management_execution():
    class AIMessage:
        name = "diship_backend_activate_tool_binding"
        content = json.dumps({
            "schema": "diship_backend_tool_activation_request.v1",
            "requested_toolsets": ["fake_grasshopper_mcp"],
        })

    intent = activation_intent_from_management_messages([AIMessage()], registry_index=REGISTRY)
    assert intent.is_empty


def test_requested_toolsets_and_exact_tools_persist_to_next_turn():
    first = merge_activation_intents(
        ToolActivationIntent(
            requested_toolsets=("fake_grasshopper_mcp",),
            requested_exact_tools=("fake_grasshopper_mcp:plan_logs",),
            source=SOURCE_CURRENT_USER_PROMPT,
            request_revision=1,
        )
    )
    restored = ToolActivationIntent(
        requested_toolsets=first.requested_toolsets,
        requested_exact_tools=first.requested_exact_tools,
        source=SOURCE_CHECKPOINTED_STATE,
        request_revision=first.request_revision,
    )
    second = merge_activation_intents(restored, ToolActivationIntent())
    assert second.requested_toolsets == first.requested_toolsets
    assert second.requested_exact_tools == first.requested_exact_tools
    assert second.request_revision == first.request_revision


def test_authorization_removal_preserves_request_but_moves_upload_pending():
    first = build_tool_profile(
        methodology="generic_engineering",
        methodology_toolsets=(),
        prompt="Bind fake_grasshopper_mcp:upload_logs",
        authorization_flags=_all_auth(),
        registry_index=REGISTRY,
    )
    second = build_tool_profile(
        methodology="generic_engineering",
        methodology_toolsets=(),
        prompt="continue",
        prior_extra_tools=first.requested_extra_tools,
        authorization_flags={
            "operator_authorized": False,
            "heavy_tools_authorized": True,
            "mutation_authorized": False,
            "persistence_authorized": True,
        },
        registry_index=REGISTRY,
    )
    assert second.requested_extra_tools == first.requested_extra_tools
    assert second.eligible_extra_tools == ()
    assert second.pending_authorization_extra_tools == ("fake_grasshopper_mcp:upload_logs",)


def test_unknown_qualified_dynamic_request_is_sticky_pending_registry():
    intent = activation_intent_from_prompt(
        "Bind missing_mcp:plan_logs",
        registry_index=REGISTRY,
        request_revision=4,
    )
    assert intent.requested_exact_tools == ("missing_mcp:plan_logs",)
    assert intent.pending_registry_requests == ("missing_mcp:plan_logs",)


def test_replayed_management_result_does_not_advance_revision_again():
    checkpoint = ToolActivationIntent(
        requested_toolsets=("fake_grasshopper_mcp",),
        requested_exact_tools=("fake_grasshopper_mcp:plan_logs",),
        source=SOURCE_CHECKPOINTED_STATE,
        request_revision=7,
    )
    replay = ToolActivationIntent(
        requested_toolsets=checkpoint.requested_toolsets,
        requested_exact_tools=checkpoint.requested_exact_tools,
        source=SOURCE_MANAGEMENT_FACADE_RESULT,
        request_revision=7,
    )
    merged = merge_activation_intents(checkpoint, replay)
    assert merged.request_revision == 7


def test_new_management_request_advances_revision_once():
    checkpoint = ToolActivationIntent(
        requested_exact_tools=("fake_grasshopper_mcp:plan_logs",),
        source=SOURCE_CHECKPOINTED_STATE,
        request_revision=7,
    )
    new = ToolActivationIntent(
        requested_exact_tools=("fake_grasshopper_mcp:upload_logs",),
        source=SOURCE_MANAGEMENT_FACADE_RESULT,
        request_revision=7,
    )
    merged = merge_activation_intents(checkpoint, new)
    assert merged.request_revision == 8
    assert merged.requested_exact_tools == (
        "fake_grasshopper_mcp:plan_logs",
        "fake_grasshopper_mcp:upload_logs",
    )


def test_checkpoint_transition_persists_requests_and_recalculates_authorization():
    from types import SimpleNamespace
    from app.agent.tool_policy_state import default_tool_policy_state
    from app.agent.tool_policy_transition import build_tool_policy_update

    plan = SimpleNamespace(
        requested_toolsets=("fake_grasshopper_mcp",),
        requested_extra_tools=("fake_grasshopper_mcp:upload_logs",),
        eligible_extra_tools=("fake_grasshopper_mcp:upload_logs",),
        pending_authorization_extra_tools=(),
        unavailable_extra_tools=(),
        pending_registry_requests=(),
        unhealthy_requests=(),
        candidate_toolsets=("backend_facades", "fake_grasshopper_mcp"),
        inventory_signature="sha256:registry",
    )
    first = build_tool_policy_update(
        stored=default_tool_policy_state(),
        plan=plan,
        authorization_flags=_all_auth(),
        bound_tool_names=["upload_logs"],
        activation_request_revision=1,
    )
    assert first["requested_toolsets"] == ["fake_grasshopper_mcp"]
    assert first["requested_extra_tools"] == ["fake_grasshopper_mcp:upload_logs"]
    assert first["last_activation_status"]["fake_grasshopper_mcp:upload_logs"] == "BOUND"

    pending_plan = SimpleNamespace(
        requested_toolsets=plan.requested_toolsets,
        requested_extra_tools=plan.requested_extra_tools,
        eligible_extra_tools=(),
        pending_authorization_extra_tools=plan.requested_extra_tools,
        unavailable_extra_tools=(),
        pending_registry_requests=(),
        unhealthy_requests=(),
        candidate_toolsets=plan.candidate_toolsets,
        inventory_signature=plan.inventory_signature,
    )
    second = build_tool_policy_update(
        stored=first,
        plan=pending_plan,
        authorization_flags={
            "operator_authorized": False,
            "heavy_tools_authorized": True,
            "mutation_authorized": False,
            "persistence_authorized": True,
        },
        bound_tool_names=[],
        activation_request_revision=1,
    )
    assert second["requested_extra_tools"] == first["requested_extra_tools"]
    assert second["last_activation_status"]["fake_grasshopper_mcp:upload_logs"] == "PENDING_AUTHORIZATION"
    assert second["activation_request_revision"] == 1


def test_agentic_source_merges_all_three_activation_sources_before_binding():
    from pathlib import Path

    source = Path("app/agent/agents/agentic_rag.py").read_text()
    checkpoint_pos = source.index("activation_intent_from_checkpoint(")
    prompt_pos = source.index("activation_intent_from_prompt(", checkpoint_pos)
    management_pos = source.index("activation_intent_from_management_messages(", prompt_pos)
    merge_pos = source.index("merge_activation_intents(", management_pos)
    binding_pos = source.index("get_scoped_tools_for_prompt(", merge_pos)
    assert checkpoint_pos < prompt_pos < management_pos < merge_pos < binding_pos
    assert "prior_requested_toolsets=_stored_policy[\"requested_toolsets\"]" in source
    assert "registry_index=_registry_index" in source


def test_management_facade_is_truthful_and_registry_backed(monkeypatch):
    import importlib
    import sys
    import types
    from pathlib import Path

    fake_lc = types.ModuleType("langchain_core")
    fake_tools = types.ModuleType("langchain_core.tools")

    def fake_tool(_name):
        def decorate(fn):
            return fn
        return decorate

    fake_tools.tool = fake_tool
    monkeypatch.setitem(sys.modules, "langchain_core", fake_lc)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", fake_tools)
    spec = importlib.util.spec_from_file_location(
        "management_under_test", Path("app/agent/agents/tools/management.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.RegistryToolIndex, "live", classmethod(lambda cls: REGISTRY))
    monkeypatch.setattr(module, "_actual_activation_status", lambda extras: {"state_known": False})

    result = module.diship_backend_activate_tool_binding(
        tool_names="fake_grasshopper_mcp:upload_logs",
        toolsets="",
    )
    assert result["activation_status"] == "REQUESTED"
    assert result["request_recorded"] is False
    assert result["recording_deferred_to_orchestration_hook"] is True
    assert result["activation_performed"] is False
    assert result["write_performed"] is False
    assert result["requested_extra_tools"] == ["fake_grasshopper_mcp:upload_logs"]
    assert "PENDING_AUTHORIZATION" not in result["activation_status"]

    ambiguous = module.diship_backend_activate_tool_binding(tool_names="plan_logs")
    assert ambiguous["activation_status"] == "AMBIGUOUS_TOOL_NAME"
    assert ambiguous["ok"] is False
    assert ambiguous["ambiguous_requests"]["plan_logs"] == [
        "fake_grasshopper_mcp:plan_logs",
        "other_mcp:plan_logs",
    ]
