"""D3B1 health-qualified registry, persistent LKG, gate, D1 and D2 tests.

Covers requirement items 8-22 and 26-35 of the D3B1 test matrix.  Every failure
outcome is injected through a fake async factory; no production MCP networking is
disturbed and no test writes to the production cache directory.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import subprocess
import sys

import pytest

from app.agent import mcp_registry_health as rh
from app.agent import registry_canonical as canonical
from app.agent.agents.tools import registry as reg
# Bound at import time: another test in the same session replaces this
# module in sys.modules, so a late in-test import can resolve to a stub.
from app.agent.agents.tools.management import diship_backend_tool_inventory_status

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

FAMILY = "faketools"
OTHER_FAMILY = "otherfam"


class FakeTool:
    def __init__(self, name, description="d", args_schema=None):
        self.name = name
        self.description = description
        self.args_schema = args_schema if args_schema is not None else {"type": "object", "properties": {}}


def _tools(*names):
    return [FakeTool(name) for name in names]


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Isolate the real registry module state and the persistent cache."""
    cache = tmp_path / "registry_cache"
    monkeypatch.setenv("MCP_REGISTRY_CACHE_DIR", str(cache))
    monkeypatch.setattr(reg, "_ASYNC_TOOL_FACTORIES", {}, raising=True)
    monkeypatch.setattr(reg, "_ASYNC_TOOL_CACHE", {}, raising=True)
    monkeypatch.setattr(reg, "_TOOL_FACTORIES", dict(reg._TOOL_FACTORIES), raising=True)
    monkeypatch.setattr(reg, "_MCP_TOOLSET_STATUS", {}, raising=True)
    monkeypatch.setattr(reg, "_FAMILY_CONTENT_SIGNATURE", {}, raising=True)
    monkeypatch.setattr(reg, "_FAMILY_LKG_MATERIALS", {}, raising=True)
    monkeypatch.setattr(reg, "_LKG_PRELOADED", False, raising=True)
    monkeypatch.setattr(reg, "_MCP_REGISTRY_REFRESH_EPOCH", 0, raising=True)
    monkeypatch.setattr(reg, "_MCP_REGISTRY_LOCK", asyncio.Lock(), raising=True)
    rh.reset_family_states()
    yield cache
    rh.reset_family_states()


def _install(family, behaviour):
    async def factory():
        return behaviour()

    reg._ASYNC_TOOL_FACTORIES[family] = factory


def _ok(*names):
    return lambda: _tools(*names)


def _timeout():
    def raise_timeout():
        raise asyncio.TimeoutError()

    return raise_timeout


def _cancel():
    def raise_cancel():
        raise asyncio.CancelledError()

    return raise_cancel


def _transport():
    def raise_transport():
        raise RuntimeError("connect failed https://internal.example/mcp?token=SUPERSECRET")

    return raise_transport


def _invalid_schema():
    def build():
        return [FakeTool("bad_tool", args_schema={"nested": object()})]

    return build


def _init(force=True, families=None):
    asyncio.run(reg.initialize_mcp_tools(force=force, toolsets=families))


# ==========================================================================
# 8-12: persistent last-known-good cache
# ==========================================================================
def test_snapshot_writes_atomically_with_correct_modes(isolated):
    materials = rh.family_materials(_tools("a_tool", "b_tool"), FAMILY)
    snapshot, error = rh.write_snapshot(FAMILY, materials)
    assert error == ""
    assert snapshot is not None
    path = rh.snapshot_path(FAMILY)
    assert path.exists()
    modes = rh.cache_modes()
    assert modes["directory"] == "0o700"
    assert modes["files"] == "0o600"
    leftovers = list(path.parent.glob(".tmp-*"))
    assert leftovers == [], "atomic write left a temporary file behind"


def test_snapshot_round_trip_preserves_content_signature(isolated):
    materials = rh.family_materials(_tools("a_tool", "b_tool"), FAMILY)
    written, _ = rh.write_snapshot(FAMILY, materials)
    loaded, error = rh.load_snapshot(FAMILY)
    assert error == ""
    assert loaded["content_signature"] == written["content_signature"]
    assert loaded["schema_version"] == rh.LKG_SCHEMA_VERSION
    assert loaded["canonicalization_version"] == canonical.CANONICALIZATION_VERSION


def test_snapshot_loads_in_a_separate_process(isolated):
    materials = rh.family_materials(_tools("a_tool", "b_tool"), FAMILY)
    written, _ = rh.write_snapshot(FAMILY, materials)
    code = (
        "import json;"
        "from app.agent import mcp_registry_health as rh;"
        "s, e = rh.load_snapshot('" + FAMILY + "');"
        "print(json.dumps({'sig': (s or {}).get('content_signature'), 'err': e}))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["MCP_REGISTRY_CACHE_DIR"] = str(isolated)
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=180
    )
    assert out.returncode == 0, out.stderr[-2000:]
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    assert payload["err"] == ""
    assert payload["sig"] == written["content_signature"]


def test_corrupted_cache_fails_safely(isolated):
    rh.write_snapshot(FAMILY, rh.family_materials(_tools("a_tool"), FAMILY))
    path = rh.snapshot_path(FAMILY)
    path.write_text("{ this is not json", encoding="utf-8")
    loaded, error = rh.load_snapshot(FAMILY)
    assert loaded is None
    assert error == rh.ERROR_CACHE_CORRUPT


def test_signature_tampered_cache_fails_safely(isolated):
    rh.write_snapshot(FAMILY, rh.family_materials(_tools("a_tool"), FAMILY))
    path = rh.snapshot_path(FAMILY)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tools"][0]["description"] = "tampered upstream description"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, error = rh.load_snapshot(FAMILY)
    assert loaded is None
    assert error == rh.ERROR_CACHE_CORRUPT


def test_wrong_version_cache_fails_safely(isolated):
    rh.write_snapshot(FAMILY, rh.family_materials(_tools("a_tool"), FAMILY))
    path = rh.snapshot_path(FAMILY)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "mcp_registry_lkg.v0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, error = rh.load_snapshot(FAMILY)
    assert loaded is None
    assert error == rh.ERROR_CACHE_CORRUPT


def test_oversized_cache_fails_safely(isolated):
    rh.write_snapshot(FAMILY, rh.family_materials(_tools("a_tool"), FAMILY))
    path = rh.snapshot_path(FAMILY)
    path.write_text("x" * (rh.MAX_SNAPSHOT_BYTES + 10), encoding="utf-8")
    loaded, error = rh.load_snapshot(FAMILY)
    assert loaded is None
    assert error == rh.ERROR_CACHE_CORRUPT


def test_invalid_material_is_never_written(isolated):
    snapshot, error = rh.write_snapshot(FAMILY, [{"name": "x", "args_schema": {"bad": object()}}])
    assert snapshot is None
    assert error == rh.ERROR_INVALID_SCHEMA
    assert not rh.snapshot_path(FAMILY).exists()


def test_load_all_skips_corrupt_family_without_raising(isolated):
    rh.write_snapshot(FAMILY, rh.family_materials(_tools("a_tool"), FAMILY))
    rh.write_snapshot(OTHER_FAMILY, rh.family_materials(_tools("b_tool"), OTHER_FAMILY))
    rh.snapshot_path(OTHER_FAMILY).write_text("broken", encoding="utf-8")
    loaded = rh.load_all_snapshots()
    assert FAMILY in loaded
    assert OTHER_FAMILY not in loaded


# ==========================================================================
# 13-20: discovery-state transitions
# ==========================================================================
def test_successful_first_discovery_establishes_baseline(isolated):
    _install(FAMILY, _ok("t_one", "t_two"))
    _init()
    assert rh.family_health(FAMILY) == rh.FAMILY_HEALTHY
    state = rh.get_family_state(FAMILY)
    assert state["source"] == rh.SOURCE_LIVE_DISCOVERY
    assert state["content_signature"].startswith("sha256:")
    assert state["tool_count"] == 2
    assert rh.snapshot_path(FAMILY).exists()


def test_identical_refresh_preserves_content_signature_and_bumps_epoch(isolated):
    _install(FAMILY, _ok("t_one", "t_two"))
    _init()
    first = reg.get_registry_content_signature()
    first_epoch = reg.get_registry_refresh_epoch()
    _init()
    assert reg.get_registry_content_signature() == first
    assert reg.get_registry_refresh_epoch() == first_epoch + 1
    assert rh.family_health(FAMILY) == rh.FAMILY_HEALTHY


def test_semantic_change_changes_content_signature(isolated):
    _install(FAMILY, _ok("t_one"))
    _init()
    before = reg.get_registry_content_signature()
    _install(FAMILY, _ok("t_one", "t_three"))
    _init()
    after = reg.get_registry_content_signature()
    assert after != before
    assert rh.family_health(FAMILY) == rh.FAMILY_HEALTHY
    loaded, error = rh.load_snapshot(FAMILY)
    assert error == ""
    assert loaded["tool_count"] == 2


def test_timeout_with_baseline_retains_content_and_marks_degraded(isolated):
    _install(FAMILY, _ok("t_one", "t_two"))
    _init()
    healthy_signature = reg.get_registry_content_signature()
    family_signature = rh.family_content_signature(FAMILY)
    _install(FAMILY, _timeout())
    _init()
    assert reg.get_registry_content_signature() == healthy_signature
    assert rh.family_content_signature(FAMILY) == family_signature
    state = rh.get_family_state(FAMILY)
    assert state["health"] == rh.FAMILY_DEGRADED_LAST_KNOWN_GOOD
    assert state["error_class"] == rh.ERROR_TIMEOUT


def test_transport_failure_with_baseline_retains_content(isolated):
    _install(FAMILY, _ok("t_one"))
    _init()
    before = reg.get_registry_content_signature()
    _install(FAMILY, _transport())
    _init()
    assert reg.get_registry_content_signature() == before
    state = rh.get_family_state(FAMILY)
    assert state["health"] == rh.FAMILY_DEGRADED_LAST_KNOWN_GOOD
    assert state["error_class"] == rh.ERROR_TRANSPORT


def test_cancellation_with_baseline_retains_content_and_keeps_family(isolated):
    _install(FAMILY, _ok("t_one", "t_two"))
    _install(OTHER_FAMILY, _ok("o_one"))
    _init()
    before = reg.get_registry_content_signature()
    _install(FAMILY, _cancel())
    with pytest.raises(asyncio.CancelledError):
        _init()
    # The family must still exist in registry structures and keep its content.
    assert FAMILY in rh.effective_registry_model()
    assert rh.family_health(FAMILY) == rh.FAMILY_DEGRADED_LAST_KNOWN_GOOD
    assert rh.get_family_state(FAMILY)["error_class"] == rh.ERROR_CANCELLED
    assert reg.get_registry_content_signature() == before


def test_timeout_without_baseline_binds_no_tools(isolated):
    _install(FAMILY, _timeout())
    _init()
    assert rh.family_health(FAMILY) == rh.FAMILY_UNAVAILABLE_NO_BASELINE
    assert reg.get_tools_set(FAMILY) == []
    assert not rh.snapshot_path(FAMILY).exists()


def test_cancellation_without_baseline_is_not_a_healthy_empty_family(isolated):
    _install(FAMILY, _cancel())
    with pytest.raises(asyncio.CancelledError):
        _init()
    state = rh.get_family_state(FAMILY)
    assert state["health"] == rh.FAMILY_UNAVAILABLE_NO_BASELINE
    assert state["error_class"] == rh.ERROR_CANCELLED
    assert state["health"] != rh.FAMILY_HEALTHY
    assert reg.get_tools_set(FAMILY) == []


def test_no_baseline_is_distinguishable_from_healthy_empty_family(isolated):
    _install(FAMILY, _timeout())
    _install(OTHER_FAMILY, _ok())
    _init()
    assert rh.family_health(FAMILY) == rh.FAMILY_UNAVAILABLE_NO_BASELINE
    assert rh.family_health(OTHER_FAMILY) == rh.FAMILY_HEALTHY
    status = reg.get_mcp_registry_status()
    assert FAMILY in status["unavailable_no_baseline_families"]
    assert OTHER_FAMILY in status["healthy_families"]
    material_missing = reg._family_content_material(FAMILY)
    material_empty = reg._family_content_material(OTHER_FAMILY)
    assert material_missing["content"] != material_empty["content"]


def test_invalid_new_schema_does_not_overwrite_baseline(isolated):
    _install(FAMILY, _ok("t_one", "t_two"))
    _init()
    good_signature = rh.family_content_signature(FAMILY)
    good_disk = json.loads(rh.snapshot_path(FAMILY).read_text(encoding="utf-8"))
    _install(FAMILY, _invalid_schema())
    _init()
    assert rh.family_content_signature(FAMILY) == good_signature
    assert rh.family_health(FAMILY) == rh.FAMILY_DEGRADED_LAST_KNOWN_GOOD
    assert rh.get_family_state(FAMILY)["error_class"] == rh.ERROR_INVALID_SCHEMA
    on_disk = json.loads(rh.snapshot_path(FAMILY).read_text(encoding="utf-8"))
    assert on_disk["content_signature"] == good_disk["content_signature"]


def test_recovery_from_degraded_restores_healthy_with_same_content(isolated):
    _install(FAMILY, _ok("t_one", "t_two"))
    _init()
    original = reg.get_registry_content_signature()
    _install(FAMILY, _timeout())
    _init()
    assert rh.family_health(FAMILY) == rh.FAMILY_DEGRADED_LAST_KNOWN_GOOD
    _install(FAMILY, _ok("t_one", "t_two"))
    _init()
    assert rh.family_health(FAMILY) == rh.FAMILY_HEALTHY
    assert reg.get_registry_content_signature() == original


def test_optional_family_failure_does_not_block_other_families(isolated):
    _install(FAMILY, _transport())
    _install(OTHER_FAMILY, _ok("o_one", "o_two"))
    _init()
    assert rh.family_health(OTHER_FAMILY) == rh.FAMILY_HEALTHY
    assert len(reg.get_tools_set(OTHER_FAMILY)) == 2
    assert rh.family_health(FAMILY) == rh.FAMILY_UNAVAILABLE_NO_BASELINE


def test_local_core_toolsets_remain_available(isolated):
    _install(FAMILY, _timeout())
    _init()
    assert reg.get_tools_set("backend_facades"), "core local facades must remain bound"
    assert reg.get_registry_content_signature().startswith("sha256:")


def test_persisted_baseline_survives_a_fresh_process(isolated):
    _install(FAMILY, _ok("t_one", "t_two"))
    _init()
    expected = rh.family_content_signature(FAMILY)
    code = (
        "import json;"
        "from app.agent.agents.tools import registry as reg;"
        "from app.agent import mcp_registry_health as rh;"
        "out = reg.preload_persistent_lkg(['" + FAMILY + "']);"
        "print(json.dumps({'source': out.get('" + FAMILY + "'),"
        " 'sig': rh.family_content_signature('" + FAMILY + "'),"
        " 'health': rh.family_health('" + FAMILY + "')}))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["MCP_REGISTRY_CACHE_DIR"] = str(isolated)
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=240
    )
    assert out.returncode == 0, out.stderr[-3000:]
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    assert payload["source"] == rh.SOURCE_PERSISTED_LKG
    assert payload["sig"] == expected
    assert payload["health"] == rh.FAMILY_DEGRADED_LAST_KNOWN_GOOD


# ==========================================================================
# 21-22: degraded-family execution gate
# ==========================================================================
def test_degraded_family_call_is_blocked_and_paired(isolated):
    from app.agent.tool_execution_gate import (
        UPSTREAM_UNAVAILABLE_RESULT_CODE,
        evaluate_tool_call,
        make_tool_message,
        paired_result_payload,
    )

    _install(FAMILY, _ok("t_one"))
    _init()
    _install(FAMILY, _transport())
    _init()
    assert rh.family_health(FAMILY) == rh.FAMILY_DEGRADED_LAST_KNOWN_GOOD

    decision = evaluate_tool_call(
        {"id": "call-1", "name": "t_one", "args": {}},
        last_bound_tool_names=["t_one"],
        authorization_flags={},
    )
    assert decision.allowed is False
    assert decision.result_code == UPSTREAM_UNAVAILABLE_RESULT_CODE == "BLOCKED_UPSTREAM_UNAVAILABLE"
    payload = paired_result_payload(decision)
    assert payload["ok"] is False
    assert payload["result_code"] == "BLOCKED_UPSTREAM_UNAVAILABLE"
    assert payload["write_performed"] is False
    message = make_tool_message(payload)
    identifier = getattr(message, "tool_call_id", None) or message.get("tool_call_id")
    assert identifier == "call-1", "exactly one paired result must carry the call id"


def test_healthy_family_call_is_not_blocked(isolated):
    from app.agent.tool_execution_gate import evaluate_tool_call

    _install(FAMILY, _ok("t_one"))
    _init()
    decision = evaluate_tool_call(
        {"id": "call-2", "name": "t_one", "args": {}},
        last_bound_tool_names=["t_one"],
        authorization_flags={},
    )
    assert decision.allowed is True
    assert decision.audit["upstream_health"] == rh.FAMILY_HEALTHY
    assert decision.audit["upstream_execution_blocked"] is False


def test_degraded_audit_contains_no_error_body_or_credentials(isolated):
    from app.agent.tool_execution_gate import evaluate_tool_call

    _install(FAMILY, _ok("t_one"))
    _init()
    _install(FAMILY, _transport())
    _init()
    decision = evaluate_tool_call(
        {"id": "call-3", "name": "t_one", "args": {}},
        last_bound_tool_names=["t_one"],
        authorization_flags={},
    )
    blob = json.dumps(decision.audit)
    assert "SUPERSECRET" not in blob
    assert "https://" not in blob
    assert "token" not in blob.lower().replace("heavy_auth_token", "")
    assert decision.audit["upstream_error_class"] in rh.SAFE_ERROR_CLASSES
    assert decision.audit["upstream_error_class"] == rh.ERROR_TRANSPORT
    assert decision.audit["argument_values_recorded"] is False


def test_unavailable_family_call_is_blocked(isolated):
    from app.agent.tool_execution_gate import evaluate_tool_call

    _install(FAMILY, _ok("t_one"))
    _init()
    # Force the no-baseline state while the tool name is still known.
    rh.set_family_state(
        FAMILY,
        health=rh.FAMILY_UNAVAILABLE_NO_BASELINE,
        source=rh.SOURCE_NONE,
        error_class=rh.ERROR_TIMEOUT,
    )
    decision = evaluate_tool_call(
        {"id": "call-4", "name": "t_one", "args": {}},
        last_bound_tool_names=["t_one"],
        authorization_flags={},
    )
    assert decision.allowed is False
    assert decision.result_code == "BLOCKED_UPSTREAM_UNAVAILABLE"


def test_local_tool_is_unaffected_by_family_health(isolated):
    from app.agent.tool_execution_gate import evaluate_tool_call

    decision = evaluate_tool_call(
        {"id": "call-5", "name": "diship_backend_tool_inventory_status", "args": {"scope": "registry"}},
        last_bound_tool_names=["diship_backend_tool_inventory_status"],
        authorization_flags={},
    )
    assert decision.allowed is True
    assert decision.audit["upstream_execution_blocked"] is False


# ==========================================================================
# 23-25: profile signature depends on content only
# ==========================================================================
def _policy_signature(content_signature):
    from app.agent.tool_policy_state import compute_policy_signature

    return compute_policy_signature(
        active_toolsets=[FAMILY],
        eligible_extra_tools=[],
        registry_generation=content_signature,
        authorization_flags={},
    )


def test_refresh_epoch_change_does_not_alter_profile_signature(isolated):
    _install(FAMILY, _ok("t_one", "t_two"))
    _init()
    first_content = reg.get_registry_content_signature()
    first_profile = _policy_signature(first_content)
    first_epoch = reg.get_registry_refresh_epoch()
    _init()
    _init()
    assert reg.get_registry_refresh_epoch() > first_epoch
    assert reg.get_registry_content_signature() == first_content
    assert _policy_signature(reg.get_registry_content_signature()) == first_profile


def test_health_change_with_baseline_does_not_alter_profile_signature(isolated):
    _install(FAMILY, _ok("t_one", "t_two"))
    _init()
    content = reg.get_registry_content_signature()
    profile = _policy_signature(content)
    health_before = reg.get_registry_health_signature()
    _install(FAMILY, _timeout())
    _init()
    assert rh.family_health(FAMILY) == rh.FAMILY_DEGRADED_LAST_KNOWN_GOOD
    assert reg.get_registry_health_signature() != health_before, "health signature must move"
    assert reg.get_registry_content_signature() == content, "content identity must not move"
    assert _policy_signature(reg.get_registry_content_signature()) == profile


def test_content_change_alters_profile_signature(isolated):
    _install(FAMILY, _ok("t_one"))
    _init()
    profile_before = _policy_signature(reg.get_registry_content_signature())
    _install(FAMILY, _ok("t_one", "t_two"))
    _init()
    assert _policy_signature(reg.get_registry_content_signature()) != profile_before


# ==========================================================================
# 26-27: D1 checkpoint compatibility
# ==========================================================================
def test_old_v1_checkpoint_loads_safely():
    from app.agent.tool_policy_state import load_tool_policy_state

    old = {
        "active_toolsets": ["s3_stb_logs"],
        "eligible_extra_tools": ["s3_stb_logs:build_log_capsule"],
        "authorization_flags": {"operator_authorized": True},
        "tool_profile_signature": "sha256:oldprofile",
        "tool_registry_generation": "sha256:oldcontent",
        "tool_policy_version": 1,
    }
    loaded = load_tool_policy_state(old)
    # The v1 value held content identity, so it migrates to the content field.
    assert loaded["tool_registry_content_signature"] == "sha256:oldcontent"
    # It is never treated as a trusted refresh epoch.
    assert loaded["tool_registry_refresh_epoch"] == 0
    # Authorization and requested tools are untouched.
    assert loaded["authorization_flags"]["operator_authorized"] is True
    assert loaded["active_toolsets"] == ["s3_stb_logs"]
    assert loaded["eligible_extra_tools"] == ["s3_stb_logs:build_log_capsule"]


def test_v2_fields_are_part_of_the_policy_state_contract():
    from app.agent.tool_policy_state import POLICY_STATE_KEYS, default_tool_policy_state

    for key in (
        "tool_registry_content_signature",
        "tool_registry_refresh_epoch",
        "tool_registry_health_signature",
    ):
        assert key in POLICY_STATE_KEYS
        assert key in default_tool_policy_state()
    defaults = default_tool_policy_state()
    assert defaults["tool_registry_refresh_epoch"] == 0
    assert defaults["tool_registry_content_signature"] == ""


def test_next_turn_writes_v2_fields(isolated):
    from app.agent.tool_policy_state import (
        current_registry_content_signature,
        current_registry_health_signature,
        current_registry_refresh_epoch,
        load_tool_policy_state,
    )

    _install(FAMILY, _ok("t_one"))
    _init()
    resolved = {
        "tool_registry_content_signature": current_registry_content_signature(),
        "tool_registry_refresh_epoch": current_registry_refresh_epoch(),
        "tool_registry_health_signature": current_registry_health_signature(),
    }
    loaded = load_tool_policy_state(resolved)
    assert loaded["tool_registry_content_signature"] == reg.get_registry_content_signature()
    assert loaded["tool_registry_refresh_epoch"] == reg.get_registry_refresh_epoch()
    assert loaded["tool_registry_health_signature"] == reg.get_registry_health_signature()
    assert loaded["tool_registry_refresh_epoch"] >= 1


def test_bad_epoch_value_degrades_to_zero():
    from app.agent.tool_policy_state import load_tool_policy_state

    assert load_tool_policy_state({"tool_registry_refresh_epoch": "not-a-number"})[
        "tool_registry_refresh_epoch"
    ] == 0
    assert load_tool_policy_state({"tool_registry_refresh_epoch": -5})["tool_registry_refresh_epoch"] == 0


# ==========================================================================
# 28-32: D2 child snapshot semantics
# ==========================================================================
def _snapshot(**kw):
    from app.agent_mode.child_tool_policy import ChildToolPolicy

    base = dict(
        authorization_flags=(("operator_authorized", True),),
        eligible_toolsets=(FAMILY,),
        eligible_extra_tools=("faketools:t_one",),
        registry_generation="sha256:content-a",
        registry_content_signature="sha256:content-a",
        registry_refresh_epoch=3,
        registry_health_signature="sha256:health-a",
        child_run_id="child-1",
    )
    base.update(kw)
    return ChildToolPolicy(**base)


def test_epoch_only_mismatch_does_not_narrow():
    from app.agent_mode.child_tool_policy import reconcile_snapshot_with_registry

    policy = _snapshot()
    result, match = reconcile_snapshot_with_registry(
        policy,
        current_content_signature="sha256:content-a",
        current_refresh_epoch=policy.registry_refresh_epoch + 9,
    )
    assert match is True
    assert result.eligible_extra_tools == policy.eligible_extra_tools
    assert result.eligible_toolsets == policy.eligible_toolsets
    assert result.snapshot_status == policy.snapshot_status


def test_health_only_mismatch_does_not_broaden():
    from app.agent_mode.child_tool_policy import reconcile_snapshot_with_registry

    policy = _snapshot()
    result, match = reconcile_snapshot_with_registry(
        policy,
        current_content_signature="sha256:content-a",
        current_health_signature="sha256:health-CHANGED",
    )
    assert match is True
    assert set(result.eligible_extra_tools) <= set(policy.eligible_extra_tools)
    assert set(result.eligible_toolsets) <= set(policy.eligible_toolsets)
    assert result.flags == policy.flags


def test_content_mismatch_narrows_and_removes_absent_tools(isolated):
    from app.agent_mode.child_tool_policy import reconcile_snapshot_with_registry

    policy = _snapshot()
    result, match = reconcile_snapshot_with_registry(
        policy, current_content_signature="sha256:content-DIFFERENT"
    )
    assert match is False
    # Nothing upstream provides the snapshot tool, so it is removed.
    assert result.eligible_extra_tools == ()
    assert set(result.eligible_extra_tools) <= set(policy.eligible_extra_tools)


def test_newly_appeared_tool_is_never_added_to_a_snapshot(isolated):
    from app.agent_mode.child_tool_policy import reconcile_snapshot_with_registry

    _install(FAMILY, _ok("t_one", "t_brand_new"))
    _init()
    policy = _snapshot(eligible_extra_tools=("faketools:t_one",))
    result, _match = reconcile_snapshot_with_registry(
        policy, current_content_signature="sha256:content-DIFFERENT"
    )
    assert "faketools:t_brand_new" not in result.eligible_extra_tools
    assert set(result.eligible_extra_tools) <= {"faketools:t_one"}


def test_v1_snapshot_without_content_field_still_compares(isolated):
    from app.agent_mode.child_tool_policy import ChildToolPolicy, reconcile_snapshot_with_registry

    legacy = ChildToolPolicy(
        eligible_toolsets=(FAMILY,),
        eligible_extra_tools=(),
        registry_generation="sha256:legacy",
        child_run_id="child-legacy",
    )
    assert legacy.content_signature == "sha256:legacy"
    _result, match = reconcile_snapshot_with_registry(legacy, current_content_signature="sha256:legacy")
    assert match is True


def test_snapshot_safe_dict_separates_content_epoch_and_health():
    policy = _snapshot()
    safe = policy.as_safe_dict()
    assert safe["registry_content_signature"] == "sha256:content-a"
    assert safe["registry_refresh_epoch"] == 3
    assert safe["registry_health_signature"] == "sha256:health-a"


def test_narrow_preserves_registry_identity_fields():
    policy = _snapshot()
    narrowed = policy.narrow(eligible_extra_tools=())
    assert narrowed.registry_content_signature == policy.registry_content_signature
    assert narrowed.registry_refresh_epoch == policy.registry_refresh_epoch
    assert narrowed.registry_health_signature == policy.registry_health_signature


# ==========================================================================
# 33: management facade separation
# ==========================================================================
def test_management_facade_reports_content_and_health_separately(isolated):
    _install(FAMILY, _ok("t_one"))
    _init()
    _install(OTHER_FAMILY, _timeout())
    _init(families=[OTHER_FAMILY])

    report = diship_backend_tool_inventory_status.func(scope="registry")
    assert report["ok"] is True
    assert report["registry_content_signature"] == reg.get_registry_content_signature()
    assert report["registry_refresh_epoch"] == reg.get_registry_refresh_epoch()
    assert report["registry_health_signature"] == reg.get_registry_health_signature()
    assert report["canonicalization_version"] == canonical.CANONICALIZATION_VERSION
    assert report["family_health"][FAMILY] == rh.FAMILY_HEALTHY
    assert report["family_health"][OTHER_FAMILY] == rh.FAMILY_UNAVAILABLE_NO_BASELINE
    assert report["family_inventory_source"][FAMILY] == rh.SOURCE_LIVE_DISCOVERY
    assert report["write_performed"] is False
    assert report["registry_content_signature"] != str(report["registry_refresh_epoch"])


def test_registry_status_schema_is_v2_and_has_no_error_bodies(isolated):
    _install(FAMILY, _transport())
    _init()
    status = reg.get_mcp_registry_status()
    assert status["schema"] == "diship_mcp_registry_status.v2"
    assert status["lkg_schema_version"] == rh.LKG_SCHEMA_VERSION
    families = status["families"]
    assert FAMILY in families
    blob = json.dumps(families)
    assert "SUPERSECRET" not in blob
    assert "https://" not in blob
    for state in families.values():
        assert state["error_class"] in rh.SAFE_ERROR_CLASSES
