"""D3B2: deterministic tool-profile / model-schema / cache-stability benchmarks.

These tests are the committed regression form of the Phase D3B2 benchmark. They
run entirely offline against a synthetic controlled registry, execute no tool,
touch no production database, read no real user checkpoint, and never bind the
broad inventory to a live model.

They assert the measured D3B2 invariants *and* pin the two documented
characterization findings, so a future behaviour change is detected instead of
silently drifting.
"""

from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmarks"))

bench = pytest.importorskip("tool_profile_cache_benchmark")


# ---------------------------------------------------------------------------
# Test-session isolation repair
#
# tests/test_dynamic_registry_runtime_v1.py replaces sys.modules["langchain_core"]
# and sys.modules["langchain_core.tools"] with minimal stubs and never restores
# them. Any later test that reaches real langchain serialization then fails with
# ImportError "cannot import name Tool from langchain_core.tools (unknown
# location)" -- including the langchain_core convert_to_openai_tool helper, which
# imports Tool lazily.
#
# D3B2 must measure the REAL provider serialization, so the genuine package is
# restored for the duration of these tests and the previous sys.modules entries
# are put back afterwards. Nothing outside this module is mutated, and the
# already-imported real submodules are reused so class identity is preserved.
# Fixing the polluting test itself is a separate test-isolation change and is
# recorded as a follow-up rather than bundled into a measurement phase.
# ---------------------------------------------------------------------------
_LANGCHAIN_KEYS = ("langchain_core", "langchain_core.tools")


def _is_stubbed(module: object | None) -> bool:
    return module is not None and getattr(module, "__file__", None) is None


def _real_langchain_core_dir() -> Path | None:
    for probe in ("langchain_core.tools.structured",
                  "langchain_core.utils.function_calling",
                  "langchain_core.tools.base"):
        located = getattr(sys.modules.get(probe), "__file__", None)
        if located:
            path = Path(located).resolve()
            for parent in path.parents:
                if parent.name == "langchain_core":
                    return parent
    return None


@contextmanager
def genuine_langchain_core():
    saved = {key: sys.modules.get(key) for key in _LANGCHAIN_KEYS}
    repaired = False
    try:
        if _is_stubbed(sys.modules.get("langchain_core.tools")):
            root = _real_langchain_core_dir()
            if root is None:
                pytest.skip("genuine langchain_core package could not be located "
                            "after test-session sys.modules pollution")
            core = sys.modules.get("langchain_core")
            if core is not None and not getattr(core, "__path__", None):
                core.__path__ = [str(root)]  # type: ignore[attr-defined]
            init = root / "tools" / "__init__.py"
            spec = importlib.util.spec_from_file_location(
                "langchain_core.tools", init,
                submodule_search_locations=[str(root / "tools")])
            if spec is None or spec.loader is None:
                pytest.skip("genuine langchain_core.tools spec unavailable")
            module = importlib.util.module_from_spec(spec)
            sys.modules["langchain_core.tools"] = module
            spec.loader.exec_module(module)
            repaired = True
        yield repaired
    finally:
        for key, previous in saved.items():
            if previous is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous


# ---------------------------------------------------------------------------
# Shared measurement (one build, many assertions)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def measured():
    with genuine_langchain_core() as repaired:
        fixture, digest = bench.load_fixture()
        meter = bench.resolve_token_meter("aws-bedrock")
        registry = bench.ControlledRegistry.from_fixture(fixture)
        restore = registry.install()
        try:
            core = bench.run_scenarios(registry, meter)
            broad = bench.measure_broad_inventory(registry, meter)
        finally:
            restore()
        reg = bench.run_registry_scenarios(registry, meter)
        child = bench.run_child_scenario(registry, meter)
        return {
            "fixture_digest": digest,
            "registry": registry,
            "meter": meter,
            "scenarios": {**core["scenarios"], **reg["scenarios"]},
            "checks": {**core["checks"], **reg["checks"]},
            "broad": broad,
            "child": child,
            "langchain_core_repaired": repaired,
        }


# ---------------------------------------------------------------------------
# Harness integrity
# ---------------------------------------------------------------------------
def test_fixture_is_deterministic_and_matches_committed_lock():
    _, digest_a = bench.load_fixture()
    _, digest_b = bench.load_fixture()
    assert digest_a == digest_b, "fixture generation must be deterministic"
    lock = bench.fixture_lock_path()
    assert lock.exists(), "committed fixture digest lock is missing"
    assert lock.read_text(encoding="utf-8").split()[0].strip() == digest_a


def test_serialization_uses_the_real_provider_path(measured):
    assert measured["broad"]["broad_all_tools"]["serializer"] == (
        "langchain_aws.chat_models.bedrock_converse._format_tools"
    )


def test_token_measurement_is_labelled_estimated_not_exact(measured):
    info = measured["meter"].to_dict()
    assert info["token_measurement"] in {"exact", "estimated"}
    if not measured["meter"].exact:
        assert info["token_measurement"] == "estimated"
        assert "not authoritative" in info["tokenizer"] or "heuristic" in info["tokenizer"]


def test_benchmark_tool_objects_are_never_executable(measured):
    tool = measured["registry"].tools_by_family["search"][0]
    with pytest.raises(Exception):
        tool.func()


# ---------------------------------------------------------------------------
# Scenario 1 / 2 -- generic stable core
# ---------------------------------------------------------------------------
def test_generic_initial_binds_only_stable_core(measured):
    rec = measured["scenarios"]["S01_generic_initial"]
    assert rec["methodology"] == "generic_engineering"
    assert rec["eligible_extra_tools"] == []
    assert rec["pending_authorization_extra_tools"] == []
    assert rec["bound_tool_count"] > 0
    for family in ("s3_stb_logs", "qos_mcp", "rtr_alerts_mcp", "grasshopper_mcp"):
        assert family not in rec["active_toolsets"]


def test_generic_initial_excludes_code_execution_tools(measured):
    from app.agent.tool_profiles import is_code_execution_tool

    rec = measured["scenarios"]["S01_generic_initial"]
    assert not [n for n in rec["bound_tool_names"] if is_code_execution_tool(n)]


def test_generic_followup_is_fully_cache_stable(measured):
    chk = measured["checks"]["S02_generic_followup_stable"]
    assert chk["profile_signature_stable"] is True
    assert chk["schema_digest_stable"] is True
    assert chk["tool_count_stable"] is True
    assert chk["cache_key_stable"] is True
    assert chk["diff"]["delta_bytes"] == 0
    assert chk["diff"]["delta_tokens"] == 0
    assert chk["diff"]["changed_fields"] == []


# ---------------------------------------------------------------------------
# Scenario 3 / 4 -- S3 read-only investigation
# ---------------------------------------------------------------------------
def test_s3_initial_activates_only_curated_read_only_core(measured):
    rec = measured["scenarios"]["S03_s3_initial"]
    assert "s3_stb_logs" in rec["active_toolsets"]
    assert rec["eligible_extra_tools"] == []
    heavy = {"build_log_capsule", "build_complete_log_capsule", "create_log_bundle",
             "get_timeline", "summarize_log_patterns", "filter_log_lines",
             "compare_log_capsules"}
    assert not (heavy & set(rec["bound_tool_names"])), "no heavy exact tool may be bound"


def test_s3_curated_core_never_binds_a_tool_absent_upstream(measured):
    """list_incident_scenes is curated but absent from the current S3 runtime."""
    rec = measured["scenarios"]["S03_s3_initial"]
    assert "list_incident_scenes" in rec["curated_tools_by_toolset"]["s3_stb_logs"]
    assert "list_incident_scenes" not in rec["bound_tool_names"]


def test_s3_followup_is_additive_monotone_and_converges(measured):
    """Documented D3B2 finding F1.

    The prescribed read-only continuation wording re-selects the generic
    fallback methodology, which contributes ``dish_internal``. The result must
    remain additive, must never remove a tool, and must converge after one turn.
    """
    chk = measured["checks"]["S04_generic_fallback_churn_characterization"]
    assert chk["monotone_never_removes"] is True
    assert chk["second_continuation_converged"] is True
    assert chk["second_continuation_diff_bytes"] == 0
    assert chk["second_continuation_added_tools"] == []
    assert measured["checks"]["S04_s3_followup_stable"]["no_family_oscillation"] is True
    assert measured["checks"]["S04_s3_followup_stable"][
        "followup_expansion_variant"]["additive_only"] is True


def test_s3_followup_is_fully_stable_when_methodology_is_re_selected(measured):
    chk = measured["checks"]["S04_generic_fallback_churn_characterization"]
    assert chk["domain_worded_profile_stable"] is True
    assert chk["domain_worded_schema_stable"] is True
    assert chk["domain_worded_delta_bytes"] == 0


# ---------------------------------------------------------------------------
# Scenario 5 -- pending exact tool adds no schema
# ---------------------------------------------------------------------------
def test_pending_heavy_exact_adds_zero_model_facing_schema(measured):
    chk = measured["checks"]["S05_pending_adds_no_schema"]
    assert chk["status"] == "PENDING_AUTHORIZATION"
    assert chk["request_retained"] is True
    assert chk["not_eligible"] is True
    assert chk["not_model_bound"] is True
    assert chk["delta_bytes"] == 0
    assert chk["delta_tokens"] == 0
    assert chk["delta_tool_count"] == 0


# ---------------------------------------------------------------------------
# Scenario 6 -- authorization adds only what is necessary
# ---------------------------------------------------------------------------
def test_authorization_adds_only_the_requested_exact_tool(measured):
    chk = measured["checks"]["S06_authorization_adds_only_necessary"]
    assert chk["authorization_delta"]["heavy_tools_authorized"] == "GRANT"
    assert chk["no_repeat_request_needed"] is True
    assert chk["eligible"] is True
    assert chk["only_requested_tool_added"] is True
    assert chk["removed_tools"] == []
    assert chk["delta_tool_count"] == 1
    assert chk["delta_bytes"] > 0
    assert chk["profile_signature_changed"] is True


def test_two_requirement_tool_stays_pending_until_every_requirement_is_met(measured):
    chk = measured["checks"]["S06_staged_two_requirement_transition"]
    assert chk["heavy_only_status"] == "PENDING_AUTHORIZATION"
    assert chk["heavy_only_bound"] is False
    assert chk["both_status"] == "BOUND"
    assert chk["both_bound"] is True
    assert chk["added_tools"] == ["build_complete_log_capsule"]


# ---------------------------------------------------------------------------
# Scenario 7 -- revocation removes the schema
# ---------------------------------------------------------------------------
def test_revocation_removes_the_exact_tool_schema_with_no_stale_binding(measured):
    chk = measured["checks"]["S07_revocation_removes_schema"]
    assert chk["authorization_delta"]["heavy_tools_authorized"] == "REVOKE"
    assert chk["status"] == "PENDING_AUTHORIZATION"
    assert chk["not_eligible"] is True
    assert chk["not_bound"] is True
    assert chk["removed_tools"] == ["build_log_capsule"]
    assert chk["removal_bytes"] > 0
    assert chk["returns_to_pending_baseline_digest"] is True


# ---------------------------------------------------------------------------
# Scenario 8 -- unavailable upstream tool adds no schema
# ---------------------------------------------------------------------------
def test_unavailable_incident_scene_tool_is_never_pending_or_bound(measured):
    chk = measured["checks"]["S08_unavailable_adds_no_schema"]
    assert chk["status"] == "UNAVAILABLE_UPSTREAM"
    assert chk["status_when_authorized"] == "UNAVAILABLE_UPSTREAM"
    assert chk["reported_unavailable"] is True
    assert chk["not_bound"] is True
    assert chk["not_bound_when_authorized"] is True
    assert chk["delta_bytes"] == 0
    assert chk["delta_tokens"] == 0


# ---------------------------------------------------------------------------
# Scenario 9 -- equivalent refresh causes no churn
# ---------------------------------------------------------------------------
def test_equivalent_registry_refresh_causes_zero_churn(measured):
    chk = measured["checks"]["S09_equivalent_refresh_no_churn"]
    assert chk["refresh_epoch_differs"] is True
    assert chk["loaded_at_differs"] is True
    assert chk["content_signature_identical"] is True
    for key in ("generic_profile_signature_stable", "generic_schema_digest_stable",
                "generic_cache_key_stable", "s3_profile_signature_stable",
                "s3_schema_digest_stable", "s3_cache_key_stable"):
        assert chk[key] is True, key


# ---------------------------------------------------------------------------
# Scenario 10 -- semantic change invalidates correctly
# ---------------------------------------------------------------------------
def test_every_semantic_registry_change_invalidates_the_content_signature(measured):
    chk = measured["checks"]["S10_semantic_change_invalidation"]
    assert chk["every_semantic_change_invalidated_content_signature"] is True


def test_relevant_semantic_change_alters_the_affected_surface_only(measured):
    pc = measured["checks"]["S10_semantic_change_invalidation"]["per_change"]
    assert pc["tool_removed_relevant_family"]["s3_schema_digest_changed"] is True
    assert pc["tool_removed_relevant_family"]["s3_bound_tool_count"] == 24
    assert pc["argument_schema_changed_relevant"]["s3_schema_digest_changed"] is True
    assert pc["capability_metadata_changed_relevant"]["s3_schema_digest_changed"] is True
    assert pc["tool_added_unrelated_family"]["s3_schema_digest_changed"] is False


def test_unrelated_semantic_change_leaves_every_model_schema_digest_stable(measured):
    """Documented D3B2 finding F2 (fail-safe coarse invalidation)."""
    chk = measured["checks"]["S10_semantic_change_invalidation"]
    assert chk["unrelated_change_left_generic_binding_unchanged"] is True
    assert chk["generic_model_schema_digest_stable_for_every_change"] is True
    # Coarse but never under-invalidating: profile identity is recomputed even
    # when the binding did not change. Pinned so a change in granularity is seen.
    assert chk["generic_profile_signature_invalidated_by_every_change"] is True


# ---------------------------------------------------------------------------
# Scenario 11 -- restart determinism
# ---------------------------------------------------------------------------
def test_fresh_process_restart_reproduces_identical_identity():
    # Runs in clean subprocesses, so session sys.modules pollution cannot apply.
    result = bench.run_restart_scenario(restarts=2)
    assert result["status"] == "pass", result.get("error", "")
    assert len(result["distinct_pids"]) == 2, "restart probes must be separate processes"
    for key, stable in result["stability"].items():
        assert stable is True, key


# ---------------------------------------------------------------------------
# Scenario 12 -- child narrowing
# ---------------------------------------------------------------------------
def test_child_binding_is_never_broader_than_the_parent_snapshot(measured):
    chk = measured["child"]["checks"]
    for key in ("child_tool_count_le_parent", "child_bytes_le_parent",
                "child_tokens_le_parent", "child_names_subset_of_parent",
                "only_task_family_retained", "recursive_spawn_tools_absent",
                "code_execution_tools_absent", "no_authorization_elevation",
                "elevation_attempt_did_not_broaden"):
        assert chk[key] is True, key


def test_child_is_strictly_smaller_than_the_parent(measured):
    parent = measured["child"]["parent"]["model_schema"]
    child = measured["child"]["child"]["model_schema"]
    assert child["tool_count"] < parent["tool_count"]
    assert child["schema_bytes"] < parent["schema_bytes"]


# ---------------------------------------------------------------------------
# Dynamic versus broad inventory
# ---------------------------------------------------------------------------
def test_dynamic_binding_materially_reduces_the_model_facing_surface(measured):
    broad = measured["broad"]["broad_all_tools"]
    generic = measured["scenarios"]["S01_generic_initial"]["model_schema"]
    s3 = measured["scenarios"]["S03_s3_initial"]["model_schema"]

    assert broad["tool_count"] > 400
    for small in (generic, s3):
        assert small["tool_count"] < broad["tool_count"] / 10
        assert small["schema_bytes"] < broad["schema_bytes"] / 10
        assert small["schema_tokens"] < broad["schema_tokens"] / 10

    reduction = 100.0 * (broad["schema_bytes"] - generic["schema_bytes"]) / broad["schema_bytes"]
    assert reduction > 90.0


def test_broad_inventory_default_gating_still_withholds_code_execution(measured):
    """The broad baseline is a serialization exercise, not a permissive binding."""
    all_tools = measured["broad"]["broad_all_tools"]["tool_count"]
    gated = measured["broad"]["broad_default_gated"]["tool_count"]
    assert gated < all_tools, "unauthorized code-execution tools must be withheld"


# ---------------------------------------------------------------------------
# Cache-key hygiene
# ---------------------------------------------------------------------------
def test_cache_key_excludes_lifecycle_only_values(measured):
    """A refresh-epoch-only change must not alter the cache key."""
    chk = measured["checks"]["S09_equivalent_refresh_no_churn"]
    assert chk["refresh_epoch_differs"] is True
    assert chk["generic_cache_key_stable"] is True
    assert chk["s3_cache_key_stable"] is True


def test_cache_key_changes_when_the_bound_surface_changes(measured):
    generic = measured["scenarios"]["S01_generic_initial"]
    s3 = measured["scenarios"]["S03_s3_initial"]
    assert generic["cache_key_digest"] != s3["cache_key_digest"]
