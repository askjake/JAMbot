from __future__ import annotations

from dataclasses import dataclass
import json

from app.agent.evidence_status import normalize_tool_outcome
from app.agent.no_progress_controller import (
    BLOCKED_MISSING_CAPABILITY,
    NO_PROGRESS_LIMIT_REACHED,
    evaluate_no_progress,
)
from app.agent.tool_execution_gate import GateDecision, paired_result_payload


@dataclass
class Message:
    content: str
    name: str = ""


def _tool(payload: dict, name: str = "tool") -> Message:
    return Message(json.dumps(payload), name=name)


def test_execution_error_is_unknown_coverage_and_cannot_prove_absence():
    status = normalize_tool_outcome(executed=True, error_type="TimeoutError")
    assert status.result_status == "TOOL_EXECUTION_ERROR"
    assert status.coverage == "UNKNOWN"
    assert status.negative_conclusion_allowed is False


def test_paired_error_payload_carries_coverage_contract():
    decision = GateDecision(tool_call_id="c1", tool_name="list_dates", allowed=True)
    payload = paired_result_payload(decision, error="TimeoutError")
    assert payload["result_code"] == "TOOL_EXECUTION_ERROR"
    assert payload["coverage"] == "UNKNOWN"
    assert payload["negative_conclusion_allowed"] is False


def test_one_blocked_required_tool_terminates_without_unrelated_fallback():
    messages = [
        _tool(
            {
                "ok": False,
                "result_code": "TOOL_NOT_IN_LAST_BINDING",
                "tool_name": "grasshopper_plan_profile_upload",
            },
            name="grasshopper_plan_profile_upload",
        )
    ]
    decision = evaluate_no_progress(
        messages,
        required_tools={"grasshopper_plan_profile_upload"},
        exact_activation_path_available=False,
    )
    assert decision.stop is True
    assert decision.result_code == BLOCKED_MISSING_CAPABILITY
    assert decision.no_progress_attempts == 1


def test_single_exact_activation_path_is_permitted_once():
    messages = [
        _tool({"ok": False, "result_code": "TOOL_NOT_IN_LAST_BINDING", "tool_name": "x"}, name="x")
    ]
    decision = evaluate_no_progress(
        messages,
        required_tools={"x"},
        exact_activation_path_available=True,
        activation_attempted=False,
    )
    assert decision.stop is False
    assert decision.activation_allowed is True


def test_no_progress_ceiling_reached_stops():
    """Renamed 2026-08-18 from test_two_consecutive_no_progress_tool_results_stop.

    That name and its hardcoded "2" assumed the pre-tuning ceiling. The
    ceiling is a tunable config value (settings.MAX_CONSECUTIVE_TOOL_ERRORS,
    2 originally -> 4 as of the 2026-08-18 tuning), so this test now builds
    exactly `ceiling` consecutive no-progress results and asserts stop=True,
    rather than hardcoding a count that would silently go stale on the next
    tuning pass.
    """
    from app.config import get_settings

    ceiling = int(get_settings().MAX_CONSECUTIVE_TOOL_ERRORS)
    messages = [
        _tool({"ok": False, "result_code": "TOOL_EXECUTION_ERROR", "tool_name": f"t{i}"}, name=f"t{i}")
        for i in range(ceiling)
    ]
    decision = evaluate_no_progress(messages)
    assert decision.stop is True
    assert decision.result_code == NO_PROGRESS_LIMIT_REACHED
    assert decision.no_progress_attempts == ceiling


def test_one_below_ceiling_does_not_yet_stop():
    """Complement to test_no_progress_ceiling_reached_stops: ceiling-1 must not stop."""
    from app.config import get_settings

    ceiling = int(get_settings().MAX_CONSECUTIVE_TOOL_ERRORS)
    messages = [
        _tool({"ok": False, "result_code": "TOOL_EXECUTION_ERROR", "tool_name": f"t{i}"}, name=f"t{i}")
        for i in range(ceiling - 1)
    ]
    decision = evaluate_no_progress(messages)
    assert decision.stop is False


def test_successful_evidence_resets_no_progress_sequence():
    messages = [
        _tool({"ok": False, "result_code": "TOOL_EXECUTION_ERROR", "tool_name": "a"}, name="a"),
        _tool({"ok": True, "result_code": "TOOL_EXECUTION_ALLOWED", "result": {"records": [1]}}, name="b"),
    ]
    decision = evaluate_no_progress(messages)
    assert decision.stop is False
    assert decision.no_progress_attempts == 0


def test_health_status_does_not_count_as_requested_functional_success():
    messages = [
        _tool(
            {
                "ok": True,
                "result_code": "TOOL_EXECUTION_ALLOWED",
                "result": {"family": "grasshopper_mcp", "health": "HEALTHY"},
            },
            name="inventory_status",
        )
    ]
    decision = evaluate_no_progress(
        messages,
        required_tools={"grasshopper_plan_profile_upload"},
    )
    assert decision.functional_success is False


def test_terminal_report_is_bounded_and_names_missing_capability():
    messages = [
        _tool({"ok": False, "result_code": "BOUND_TOOL_NOT_EXECUTABLE", "tool_name": "agent_run_shell"}, name="agent_run_shell")
    ]
    decision = evaluate_no_progress(messages, required_tools={"agent_run_shell"})
    text = decision.render_terminal_message()
    assert len(text) < 1200
    assert "agent_run_shell" in text
    assert "BLOCKED_MISSING_CAPABILITY" in text

@dataclass
class HumanMessage:
    content: str


def test_stale_previous_turn_identifier_failure_does_not_poison_new_human_turn():
    messages = [
        _tool(
            {
                "ok": False,
                "result_code": "TOOL_IDENTIFIER_VALIDATION_FAILED",
                "tool_name": "grasshopper_get_upload_history",
            },
            name="grasshopper_get_upload_history",
        ),
        HumanMessage("Resume the duplicate preflight from the current turn only."),
    ]
    decision = evaluate_no_progress(
        messages,
        required_tools={"grasshopper_get_upload_history"},
    )
    assert decision.stop is False
    assert decision.no_progress_attempts == 0


def test_first_identifier_validation_failure_is_recoverable_not_missing_capability():
    messages = [
        _tool(
            {
                "ok": False,
                "result_code": "TOOL_IDENTIFIER_VALIDATION_FAILED",
                "tool_name": "grasshopper_get_upload_request_status",
            },
            name="grasshopper_get_upload_request_status",
        )
    ]
    decision = evaluate_no_progress(
        messages,
        required_tools={"grasshopper_get_upload_request_status"},
        exact_activation_path_available=False,
    )
    assert decision.stop is False
    assert decision.result_code == "CONTINUE"
    assert decision.no_progress_attempts == 1


def test_current_turn_tool_failure_after_human_boundary_is_still_counted():
    messages = [
        _tool(
            {
                "ok": False,
                "result_code": "TOOL_EXECUTION_ERROR",
                "tool_name": "stale_tool",
            },
            name="stale_tool",
        ),
        HumanMessage("new turn"),
        _tool(
            {
                "ok": False,
                "result_code": "TOOL_EXECUTION_ERROR",
                "tool_name": "current_tool",
            },
            name="current_tool",
        ),
    ]
    decision = evaluate_no_progress(messages, required_tools={"current_tool"})
    assert decision.stop is False
    assert decision.no_progress_attempts == 1
