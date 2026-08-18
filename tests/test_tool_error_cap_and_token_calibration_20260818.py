"""Regression tests for the 2026-08-18 tool-error-cap and token-calibration fixes.

Two separate defects, both of the same "declared but never wired / wrong model"
family found during the agent-loop investigation:

1. settings.MAX_CONSECUTIVE_TOOL_ERRORS shipped as 1555550 with zero references
   anywhere in the codebase, while the real enforcement point in
   no_progress_controller.py was hardcoded to MAX_NO_PROGRESS_ATTEMPTS = 2.

2. token_counter stashed its pre-invocation estimate in a module-level global,
   justified by the comment "each async request runs in a single coroutine
   chain, so a simple module variable suffices". Concurrent requests interleave
   at every await on one event loop, so the calibration loop paired one
   request's estimate with another request's actual token count. Production
   showed per-sample ratios from 0.04x to ~197x in a single 50-sample window and
   a "converged" factor swinging between 1.19 and 28.5.
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent import no_progress_controller as npc
from app.message import token_counter as tc


# ─── Fix A: consecutive tool-error ceiling ────────────────────────────────────

def test_config_default_is_no_longer_the_fat_fingered_value():
    from app.config import get_settings

    value = int(get_settings().MAX_CONSECUTIVE_TOOL_ERRORS)
    assert value != 1555550, "the 1555550 typo must not come back"
    assert 1 <= value <= 100, f"implausible ceiling: {value}"


def test_setting_is_actually_wired_now():
    """It previously had zero references outside its own declaration.

    Reads the expected value from settings rather than hardcoding it, so this
    test verifies *wiring* (config -> resolver) independent of whatever the
    ceiling happens to be tuned to (2 originally, 4 as of the 2026-08-18
    tuning change -- see app/config.py comment history).
    """
    from app.config import get_settings

    assert npc._resolve_max_no_progress_attempts() == int(get_settings().MAX_CONSECUTIVE_TOOL_ERRORS)


def test_configured_value_is_honoured(monkeypatch):
    class _S:
        MAX_CONSECUTIVE_TOOL_ERRORS = 7

    monkeypatch.setattr("app.config.get_settings", lambda: _S())
    assert npc._resolve_max_no_progress_attempts() == 7


@pytest.mark.parametrize("bad", [1555550, 0, -1, 10_000])
def test_implausible_config_cannot_disable_the_guard(monkeypatch, bad):
    """A mistyped ceiling must fall back, never silently switch the guard off."""
    class _S:
        MAX_CONSECUTIVE_TOOL_ERRORS = bad

    monkeypatch.setattr("app.config.get_settings", lambda: _S())
    assert npc._resolve_max_no_progress_attempts() == npc.MAX_NO_PROGRESS_ATTEMPTS


def test_missing_settings_falls_back_safely(monkeypatch):
    def _boom():
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr("app.config.get_settings", _boom)
    assert npc._resolve_max_no_progress_attempts() == npc.MAX_NO_PROGRESS_ATTEMPTS


def _failing_tool_record(name="search", code="TOOL_EXECUTION_ERROR"):
    import json

    class _ToolMessage:
        def __init__(self, name, payload):
            self.name = name
            self.content = json.dumps(payload)

    _ToolMessage.__name__ = "ToolMessage"
    return _ToolMessage(name, {"ok": False, "result_code": code, "tool_name": name})


def test_stop_threshold_follows_the_configured_ceiling(monkeypatch):
    """End-to-end: the ceiling actually changes when the setting changes."""
    two_failures = [_failing_tool_record(), _failing_tool_record()]

    class _S2:
        MAX_CONSECUTIVE_TOOL_ERRORS = 2

    monkeypatch.setattr("app.config.get_settings", lambda: _S2())
    assert npc.evaluate_no_progress(two_failures).stop is True

    class _S5:
        MAX_CONSECUTIVE_TOOL_ERRORS = 5

    monkeypatch.setattr("app.config.get_settings", lambda: _S5())
    assert npc.evaluate_no_progress(two_failures).stop is False, (
        "with a ceiling of 5, two failures must not terminate the turn"
    )


# ─── Fix B: calibration estimate isolation ────────────────────────────────────

def test_estimate_roundtrips_within_one_context():
    tc.stash_pre_invocation_estimate(4242)
    assert tc.get_last_estimate() == 4242


def test_concurrent_requests_do_not_clobber_each_others_estimate():
    """The actual production bug: a module global shared across requests.

    Each task stashes its own estimate, yields control (forcing interleaving),
    then reads it back. With the old module-level global the last writer won and
    both tasks observed the same value.
    """
    observed: dict[str, int] = {}

    async def one_request(tag: str, estimate: int) -> None:
        tc.stash_pre_invocation_estimate(estimate)
        await asyncio.sleep(0.01)  # let the other request run and stash
        observed[tag] = tc.get_last_estimate()

    async def main() -> None:
        await asyncio.gather(
            one_request("a", 1_000),
            one_request("b", 250_000),
        )

    asyncio.run(main())

    assert observed["a"] == 1_000, f"request A saw {observed['a']}, expected 1000"
    assert observed["b"] == 250_000, f"request B saw {observed['b']}, expected 250000"


def test_reset_clears_only_current_context():
    tc.stash_pre_invocation_estimate(999)
    tc.reset_pre_invocation_estimate()
    assert tc.get_last_estimate() == 0


# ─── Fix B: outlier rejection / factor clamping ───────────────────────────────

def _fresh_state(factor=1.0):
    return tc._CalibrationState(initial_factor=factor)


def test_plausible_ratio_is_accepted():
    state = _fresh_state()
    state.update(estimated=1000, actual=1150)
    assert state.sample_count == 1
    assert 1.0 < state.factor <= 2.0


@pytest.mark.parametrize(
    "estimated,actual",
    [
        (94, 18519),      # ~197x, observed in production
        (18, 18481),      # ~1027x, observed in production
        (131279, 5610),   # ~0.04x, observed in production
        (78040, 567946),  # ~7.3x, the reported 6.2x-class sample
    ],
)
def test_production_outliers_are_rejected(estimated, actual):
    """These exact shapes came from the production calibration log."""
    state = _fresh_state()
    state.update(estimated=estimated, actual=actual)
    assert state.sample_count == 0, "outlier must not enter the history"
    assert state.factor == 1.0, "factor must be untouched by an outlier"
    assert state.get_stats()["rejected_outliers"] == 1


def test_factor_stays_in_band_under_a_stream_of_outliers():
    """The reported 6.2199 factor must no longer be reachable."""
    state = _fresh_state()
    for _ in range(50):
        state.update(estimated=78040, actual=567946)
    assert 0.5 <= state.factor <= 2.0
    assert state.factor == 1.0
    assert state.get_stats()["rejected_outliers"] == 50


def test_calibrated_count_cannot_be_scaled_by_a_pathological_factor():
    """count_tokens() drives compression; a 6x multiplier destroys history."""
    text = "some representative message content " * 50
    raw = tc.count_tokens_raw(text)
    calibrated = tc.count_tokens(text)
    assert raw > 0
    assert calibrated <= raw * 2.0 + 1, (
        f"calibrated {calibrated} exceeds the plausible band vs raw {raw}"
    )
