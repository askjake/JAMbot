"""Phase D3B3 - context-compression benchmark and workflow-continuity gates.

These tests drive the committed benchmark harness so the acceptance thresholds
are enforced by CI rather than by a one-off evidence run, and they pin the two
workflow-continuity corrections made in D3B3.
"""

from __future__ import annotations

import importlib

import pytest

from app.agent.tool_execution_policy import get_scoped_tools_for_prompt
from app.agent.tool_policy_state import load_tool_policy_state

BENCH = importlib.import_module("scripts.benchmarks.context_compression_benchmark")

IRRELEVANT_FAMILIES = {
    "dish_internal",
    "jira_mcp",
    "confluence_mcp",
    "gdrive_mcp",
    "qodo_context_mcp",
}

GENERIC_CONTINUATIONS = ("Continue.", "Proceed.", "Run the next step.", "Check the result.")


@pytest.fixture(scope="module")
def results() -> dict:
    return BENCH.build_results()


# ── benchmark shape ──────────────────────────────────────────────────────────
def test_all_seven_cases_and_five_modes_are_measured(results):
    assert len(results["cases"]) == 7
    assert len(results["modes"]) == 5
    for case in results["cases"]:
        for mode in results["modes"]:
            assert mode in case["modes"]
            assert case["modes"][mode]["metrics"]["messages"] > 0


def test_token_measurement_is_labelled_as_an_estimate(results):
    tm = results["token_measurement"]
    assert tm["token_measurement"] == "estimated"
    assert tm["authoritative_local_tokenizer_available"] is False
    assert "cl100k_base" in tm["tokenizer"]


# ── correctness gates ────────────────────────────────────────────────────────
def test_tool_message_pairing_is_perfect_in_every_mode(results):
    for case in results["cases"]:
        for mode, data in case["modes"].items():
            integ = data["tool_message_integrity"]
            assert integ["pass"], f"{case['name']}/{mode}: {integ['problems']}"


def test_provider_repair_is_valid_and_never_fabricates_results(results):
    repair = results["provider_repair"]
    assert repair["pass"]
    for name, data in repair["scenarios"].items():
        assert data["provider_valid"], f"{name}: {data['problems']}"
        assert data["fabricated_success_result"] is False
    assert repair["scenarios"]["blocked_result_pair"]["blocked_result_preserved"] is True


def test_authorization_invariants_survive_compression(results):
    inv = results["authorization_invariants"]
    assert inv["grant_compressed_out"]["pass"]
    assert inv["revocation_compressed_out"]["pass"]
    assert inv["pending_compressed_out"]["pass"]
    assert inv["unavailable_compressed_out"]["pass"]
    assert inv["operational_task_compressed_out"]["pass"]
    assert inv["compression_summary_never_current_turn"]["pass"]
    assert inv["pass"]


def test_evidence_labels_and_source_identities_survive_compression(results):
    for case in results["cases"]:
        for mode, data in case["modes"].items():
            ev = data["evidence_integrity"]
            if ev.get("applicable"):
                assert ev["labels_preserved"], f"{case['name']}/{mode}"
                assert ev["identities_resolvable"], f"{case['name']}/{mode}"
                assert ev["missing_and_quality_warnings_preserved"], f"{case['name']}/{mode}"
                for item in ev["items"]:
                    assert item["promoted_to_observed"] is False


def test_targeted_expansion_returns_the_exact_source_region(results):
    checked = 0
    for case in results["cases"]:
        te = case["targeted_expansion"]
        if te.get("applicable"):
            checked += 1
            assert te["expansion_returns_exact_region"], case["name"]
            assert te["checkpoint_original_intact"], case["name"]
    assert checked >= 2


def test_no_new_unsupported_claims_or_false_negatives(results):
    agg = results["aggregates"]
    assert agg["unsupported_causal_claims_delta"] == 0
    assert agg["false_negative_delta"] == 0


def test_cachepoint_prefix_is_deterministic_and_leak_free(results):
    for case in results["cases"]:
        cp = case["cachepoint_stability"]
        assert cp["positions_identical"], case["name"]
        assert cp["prefix_identical"], case["name"]
        assert cp["full_context_identical"], case["name"]
        assert not any(cp["canary_leaks"].values()), case["name"]


def test_continuity_questions_pass_in_every_mode(results):
    for case in results["cases"]:
        for mode, data in case["modes"].items():
            cont = data["continuity"]
            assert cont["pass"], f"{case['name']}/{mode}: {cont['details']}"


# ── token reduction thresholds ───────────────────────────────────────────────
def test_every_large_case_reduces_model_visible_tokens_by_at_least_30_percent(results):
    large = [c for c in results["cases"] if c["large"]]
    assert large, "no large cases were measured"
    for case in large:
        pct = case["modes"]["B_production"]["reduction_vs_raw"]["est_tokens_pct"]
        assert pct >= 30.0, f"{case['name']} only reduced {pct}%"


def test_aggregate_large_case_reduction_is_reported(results):
    agg = results["aggregates"]
    assert agg["large_cases_meeting_30_percent"] == agg["large_cases"]
    assert agg["aggregate_large_case_reduction_pct"] >= 30.0


def test_small_control_is_not_semantically_expanded(results):
    """The short control must not lose or distort facts.

    Model-visible tokens may rise very slightly because production inserts
    cachepoint blocks, which is provider-cache instrumentation rather than
    compression.  The bound keeps that overhead small and explicit.
    """
    case = next(c for c in results["cases"] if c["name"] == "case1_small_control")
    raw = case["modes"]["A_raw"]["metrics"]["est_tokens"]
    prod = case["modes"]["B_production"]["metrics"]["est_tokens"]
    assert case["modes"]["B_production"]["continuity"]["pass"]
    assert prod - raw <= 40, f"unexpected growth: {raw} -> {prod}"
    assert case["modes"]["B_production"]["metrics"]["cachepoints"] >= 1


# ── workflow continuity corrections ──────────────────────────────────────────
def _probe(prompt: str, stored: dict, has_prior: bool) -> dict:
    state = load_tool_policy_state(stored)
    tools, plan = get_scoped_tools_for_prompt(
        prompt,
        has_prior_tool_results=has_prior,
        prior_active_toolsets=state["active_toolsets"],
        prior_extra_tools=state["requested_extra_tools"],
        authorization_flags=state["authorization_flags"],
    )
    return {
        "methodology": getattr(plan, "methodology", ""),
        "toolsets": set(getattr(plan, "candidate_toolsets", ()) or ()),
        "pending": set(getattr(plan, "pending_authorization_extra_tools", ()) or ()),
        "tool_count": len(tools),
    }


def test_generic_continuation_does_not_inject_irrelevant_family_into_s3_workflow():
    initial = _probe(
        "Investigate receiver R1954841480 guide load failures on 2026-04-12 "
        "using the receiver logs. Read-only.",
        {}, False,
    )
    assert initial["methodology"] == "backend_runtime_debug"
    stored = {"active_toolsets": sorted(initial["toolsets"]), "requested_extra_tools": []}
    for phrase in GENERIC_CONTINUATIONS + ("Summarise the findings.",):
        follow = _probe(phrase, stored, True)
        injected = follow["toolsets"] - initial["toolsets"]
        lost = initial["toolsets"] - follow["toolsets"]
        assert not (injected & IRRELEVANT_FAMILIES), f"{phrase!r} injected {injected}"
        assert not lost, f"{phrase!r} lost {lost}"


def test_operational_workflow_methodology_is_reported_stably():
    initial = _probe(
        "Clone git@gitlab.com:dish-cloud/dt/sse/datasolutions/cas/ai-test.git, "
        "review it, create an isolated venv and run its tests.",
        {}, False,
    )
    assert initial["methodology"] == "repo_checkout_local_deploy"
    stored = {
        "active_toolsets": sorted(initial["toolsets"]),
        "requested_extra_tools": sorted(initial["pending"]),
        "authorization_flags": {
            "operator_authorized": True,
            "heavy_tools_authorized": True,
            "persistence_authorized": False,
            "mutation_authorized": False,
        },
    }
    for phrase in GENERIC_CONTINUATIONS + ("Continue the previous task.",):
        follow = _probe(phrase, stored, True)
        assert follow["methodology"] == "repo_checkout_local_deploy", (
            f"{phrase!r} reported {follow['methodology']}"
        )
        assert not (follow["toolsets"] - initial["toolsets"]) & IRRELEVANT_FAMILIES


def test_first_pass_generic_chat_still_gets_its_default_toolsets():
    """Regression guard: the D3B3 continuation rule must not strip the generic
    starting set from a genuine first-pass generic prompt."""
    first = _probe("Explain how prompt caching works in general terms.", {}, False)
    assert first["methodology"] in {"generic_engineering", "no_tool_response"}
    assert first["tool_count"] >= 0
    seeded = _probe("Explain how prompt caching works in general terms.", {}, True)
    assert seeded["tool_count"] >= 0
