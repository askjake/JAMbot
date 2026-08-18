"""Regression test for the 2026-08-18 MAX_CONSECUTIVE_TOOL_ERRORS tuning.

Follow-up to the wiring fix in
tests/test_tool_error_cap_and_token_calibration_20260818.py, which connected
settings.MAX_CONSECUTIVE_TOOL_ERRORS to the real enforcement point in
no_progress_controller.py and set it to 2 to exactly preserve prior behaviour.

This test locks in the subsequent tuning: 2 -> 4, motivated by live triage
where transient MCP ToolExceptions (get_summary, search_logs, list_dates,
list_files each threw once and recovered on the very next call) tripped
BLOCKED_NO_PROGRESS_LIMIT and abandoned the turn after a single retry.
"""

from __future__ import annotations

from app.agent import no_progress_controller as npc
from app.config import get_settings


def test_ceiling_is_tuned_to_four():
    value = int(get_settings().MAX_CONSECUTIVE_TOOL_ERRORS)
    assert value == 4


def test_ceiling_still_within_the_plausible_band():
    """Guards against a future typo reintroducing something like 1555550."""
    value = int(get_settings().MAX_CONSECUTIVE_TOOL_ERRORS)
    assert 1 <= value <= 100


def test_resolved_ceiling_matches_the_tuned_config_value():
    assert npc._resolve_max_no_progress_attempts() == 4


def _failing_tool_record(name="search"):
    import json

    class _ToolMessage:
        def __init__(self, name, payload):
            self.name = name
            self.content = json.dumps(payload)

    _ToolMessage.__name__ = "ToolMessage"
    return _ToolMessage(name, {"ok": False, "result_code": "TOOL_EXECUTION_ERROR", "tool_name": name})


def test_two_transient_failures_no_longer_trip_the_guard():
    """The exact live scenario: 2 consecutive transient tool errors.

    Under the old ceiling of 2 this would stop the turn. Under 4 it must not,
    giving the agent room to retry a flaky MCP dependency.
    """
    two_failures = [_failing_tool_record(), _failing_tool_record()]
    assert npc.evaluate_no_progress(two_failures).stop is False


def test_four_consecutive_failures_still_trips_the_guard():
    """The ceiling is real, not disabled -- it is a bound, not unlimited retry."""
    four_failures = [_failing_tool_record() for _ in range(4)]
    assert npc.evaluate_no_progress(four_failures).stop is True


def test_three_consecutive_failures_do_not_yet_trip_the_guard():
    three_failures = [_failing_tool_record() for _ in range(3)]
    assert npc.evaluate_no_progress(three_failures).stop is False
