from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.agent_mode.task_capability_plan import (
    BLOCKED_CHILD_MISSING_CAPABILITY,
    PREFLIGHT_PASS,
    TaskCapabilityPlan,
    evaluate_child_capability_plan,
)


def _policy(*, toolsets=(), tools=()):
    return SimpleNamespace(eligible_toolsets=tuple(toolsets), eligible_extra_tools=tuple(tools))


def test_zero_declared_capability_plan_blocks_before_llm():
    decision = evaluate_child_capability_plan(
        plan=TaskCapabilityPlan.from_values(),
        parent_policy=_policy(toolsets=("agent_mode",), tools=("agent_mode:agent_run_shell",)),
        candidate_tool_names=("agent_run_shell",),
    )
    assert decision.allowed is False
    assert decision.status == BLOCKED_CHILD_MISSING_CAPABILITY
    assert decision.llm_call_permitted is False
    assert "DECLARED_CAPABILITY_PLAN" in decision.missing_capabilities


def test_file_writer_child_not_spawnable_without_write_tool():
    plan = TaskCapabilityPlan.from_values(
        required_tools=["list_dates"],
        expected_artifact_types=["markdown_report"],
    )
    decision = evaluate_child_capability_plan(
        plan=plan,
        parent_policy=_policy(toolsets=("s3_stb_logs",), tools=("s3_stb_logs:list_dates",)),
        candidate_tool_names=("list_dates",),
    )
    assert decision.allowed is False
    assert "filesystem_write" in decision.missing_capabilities
    assert decision.llm_call_permitted is False


def test_required_exact_tool_must_be_in_actual_candidate_binding():
    plan = TaskCapabilityPlan.from_values(required_tools=["agent_mode:agent_run_shell"])
    decision = evaluate_child_capability_plan(
        plan=plan,
        parent_policy=_policy(toolsets=("agent_mode",), tools=("agent_mode:agent_run_shell",)),
        candidate_tool_names=("agent_list_artifacts",),
    )
    assert decision.allowed is False
    assert decision.missing_tools == ("agent_mode:agent_run_shell",)


def test_required_toolset_is_intersected_with_parent_policy():
    plan = TaskCapabilityPlan.from_values(required_toolsets=["grasshopper_mcp"])
    decision = evaluate_child_capability_plan(
        plan=plan,
        parent_policy=_policy(toolsets=("s3_stb_logs",)),
        candidate_tool_names=("list_dates",),
    )
    assert decision.allowed is False
    assert decision.missing_toolsets == ("grasshopper_mcp",)


def test_declared_shell_repository_plan_passes_with_exact_bound_tool():
    plan = TaskCapabilityPlan.from_values(
        required_toolsets=["agent_mode"],
        required_tools=["agent_mode:agent_run_shell"],
        required_capabilities=["shell_execution"],
        write_required=True,
        network_required=True,
        repository_access_required=True,
    )
    decision = evaluate_child_capability_plan(
        plan=plan,
        parent_policy=_policy(
            toolsets=("agent_mode",),
            tools=("agent_mode:agent_run_shell",),
        ),
        candidate_tool_names=("agent_run_shell",),
    )
    assert decision.allowed is True
    assert decision.status == PREFLIGHT_PASS
    assert decision.llm_call_permitted is True
    assert not decision.missing_capabilities


def test_child_prompt_names_python_only_when_bound():
    source = Path("app/agent_mode/child_conversation.py").read_text()
    assert '"agent_run_python" if "agent_run_python" in bound' in source
    assert "No file-writing tool is bound" in source
    assert "bound_tool_names=[getattr(tool, \"name\", \"\") for tool in child_tools]" in source


def test_preflight_occurs_before_graph_and_block_returns_without_llm():
    source = Path("app/agent_mode/child_conversation.py").read_text()
    preflight = source.index("evaluate_child_capability_plan(")
    block = source.index("if not capability_decision.allowed:", preflight)
    graph = source.index("graph = _build_child_graph", block)
    invoke = source.index("await graph.ainvoke", graph)
    assert preflight < block < graph < invoke
    blocked_section = source[block:graph]
    assert "return result" in blocked_section
    assert "BLOCKED_CHILD_MISSING_CAPABILITY" in blocked_section


def test_blocked_or_partial_packet_is_not_amplified_to_completed():
    source = Path("app/agent_mode/child_conversation.py").read_text()
    assert '"blocked": "blocked"' in source
    assert '"partial": "partial"' in source
    assert 'result.status = "completed"' not in source


def test_spawn_schema_declares_capability_plan_fields_and_no_full_access_claim():
    source = Path("app/agent_mode/mcop_tools.py").read_text()
    for field in (
        "required_toolsets",
        "required_tools",
        "required_capabilities",
        "expected_artifact_types",
        "write_required",
        "network_required",
        "repository_access_required",
    ):
        assert field in source
    assert "with full tool\n    access" not in source
