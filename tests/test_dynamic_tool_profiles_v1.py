from __future__ import annotations

from pathlib import Path
import sys
import types
from types import SimpleNamespace

from app.agent.tool_execution_policy import build_profile_for_prompt, get_scoped_tools_for_prompt
from app.agent.tool_profiles import (
    S3_CURATED_CORE_TOOLS,
    STABLE_CORE_TOOLSETS,
    build_tool_profile,
    extract_requested_extra_tools,
    tool_profile_selftest,
)


def test_s3_methodology_binds_curated_core_not_whole_inventory():
    profile = build_tool_profile(
        methodology="receiver_reboot_dvr_playback",
        methodology_toolsets=("s3_stb_logs", "rtr_alerts_mcp"),
        prompt="Investigate receiver R1911746693 logs",
        inventory_signature="sha256:inventory",
    )
    assert set(STABLE_CORE_TOOLSETS).issubset(profile.active_toolsets)
    assert "s3_stb_logs" in profile.active_toolsets
    assert profile.curated_tools_by_toolset["s3_stb_logs"] == S3_CURATED_CORE_TOOLS
    assert "build_log_capsule" not in profile.curated_tools_by_toolset["s3_stb_logs"]
    assert "build_incident_scene" not in profile.curated_tools_by_toolset["s3_stb_logs"]


def test_read_only_scene_extra_is_immediately_eligible_and_exact():
    profile = build_tool_profile(
        methodology="receiver_reboot_dvr_playback",
        methodology_toolsets=("s3_stb_logs",),
        prompt="Compare incident scenes using s3_stb_logs:compare_incident_scenes",
    )
    assert profile.eligible_extra_tools == ("s3_stb_logs:compare_incident_scenes",)
    assert not profile.pending_authorization_extra_tools
    assert "compare_incident_scenes" in profile.curated_tools_by_toolset["s3_stb_logs"]


def test_heavy_extra_remains_pending_and_absent_until_authorized():
    first = build_tool_profile(
        methodology="receiver_reboot_dvr_playback",
        methodology_toolsets=("s3_stb_logs",),
        prompt="Build an incident scene for R1911746693",
    )
    assert first.requested_extra_tools == ("s3_stb_logs:build_incident_scene",)
    assert first.pending_authorization_extra_tools == ("s3_stb_logs:build_incident_scene",)
    assert first.missing_authorizations["s3_stb_logs:build_incident_scene"] == ("heavy_tools_authorized",)
    assert "build_incident_scene" not in first.curated_tools_by_toolset["s3_stb_logs"]

    second = build_tool_profile(
        methodology="receiver_reboot_dvr_playback",
        methodology_toolsets=("s3_stb_logs",),
        prompt="Heavy tools authorized.",
        prior_active_toolsets=first.active_toolsets,
        prior_extra_tools=first.requested_extra_tools,
        authorization_flags={"heavy_tools_authorized": True},
    )
    assert second.eligible_extra_tools == ("s3_stb_logs:build_incident_scene",)
    assert not second.pending_authorization_extra_tools
    assert "build_incident_scene" in second.curated_tools_by_toolset["s3_stb_logs"]
    assert set(first.active_toolsets).issubset(second.active_toolsets)


def test_profile_signature_is_deterministic_and_inventory_sensitive():
    kwargs = dict(
        methodology="receiver_reboot_dvr_playback",
        methodology_toolsets=("s3_stb_logs", "rtr_alerts_mcp"),
        prompt="Query the incident scene",
    )
    first = build_tool_profile(**kwargs, inventory_signature="sha256:a")
    second = build_tool_profile(**kwargs, inventory_signature="sha256:a")
    changed = build_tool_profile(**kwargs, inventory_signature="sha256:b")
    assert first.signature == second.signature
    assert first.signature != changed.signature


def test_intent_extraction_does_not_activate_unrelated_heavy_tools():
    assert extract_requested_extra_tools("Investigate receiver logs") == ()
    assert extract_requested_extra_tools("Build a log capsule") == ("s3_stb_logs:build_log_capsule",)
    assert extract_requested_extra_tools("Render the incident scene") == ("s3_stb_logs:render_incident_scene",)


def test_active_policy_adapter_uses_current_methodology_architecture():
    profile = build_profile_for_prompt("Required methodology: receiver_reboot_dvr_playback. Investigate R1911746693 logs and build an incident scene")
    assert profile.methodology == "receiver_reboot_dvr_playback"
    assert "s3_stb_logs" in profile.active_toolsets
    assert "s3_stb_logs:build_incident_scene" in profile.pending_authorization_extra_tools


def test_registry_source_contains_bounded_concurrent_refresh_and_facades():
    source = Path("app/agent/agents/tools/registry.py").read_text()
    assert "asyncio.as_completed" in source
    assert "MCP_INIT_CONCURRENCY" in source
    assert "refresh_mcp_tools" in source
    assert "invalidate_mcp_tool_cache" in source
    assert "get_tool_inventory_signature" in source
    assert '"backend_facades"' in source
    assert "error_summary" in source and "<redacted-url>" in source



def test_active_binding_filters_full_s3_inventory_and_promotes_pending_extra(monkeypatch):
    s3_tools = [SimpleNamespace(name=name, description="") for name in S3_CURATED_CORE_TOOLS]
    s3_tools.extend(SimpleNamespace(name=f"unrelated_{index}", description="") for index in range(171))
    s3_tools.append(SimpleNamespace(name="build_incident_scene", description=""))
    toolsets = {
        "s3_stb_logs": s3_tools,
        "rtr_alerts_mcp": [SimpleNamespace(name="rtr_alert_lookup", description="")],
        "search": [SimpleNamespace(name="public_web_search", description="")],
        "agent_mode": [SimpleNamespace(name="agent_run_shell", description="")],
        "backend_facades": [SimpleNamespace(name="diship_backend_tool_inventory_status", description="")],
    }
    fake = types.ModuleType("app.agent.agents.tools")
    fake.get_tools_set = lambda name: list(toolsets.get(name, []))
    fake.get_tools_set_filtered = lambda name, allowed: [tool for tool in toolsets.get(name, []) if tool.name in set(allowed)]
    monkeypatch.setitem(sys.modules, "app.agent.agents.tools", fake)

    prompt = "Required methodology: receiver_reboot_dvr_playback. Investigate R1911746693 and build an incident scene"
    first_tools, first_plan = get_scoped_tools_for_prompt(prompt)
    first_names = {tool.name for tool in first_tools}
    assert set(S3_CURATED_CORE_TOOLS).issubset(first_names)
    assert "build_incident_scene" not in first_names
    assert "filter_log_lines" not in first_names
    assert "get_heavy_auth_status" in first_names
    assert not any(name.startswith("unrelated_") for name in first_names)
    assert "s3_stb_logs:build_incident_scene" in first_plan.pending_authorization_extra_tools

    second_tools, second_plan = get_scoped_tools_for_prompt(
        "Heavy tools authorized.",
        prior_active_toolsets=first_plan.candidate_toolsets,
        prior_extra_tools=("s3_stb_logs:build_incident_scene",),
        authorization_flags={"heavy_tools_authorized": True},
    )
    second_names = {tool.name for tool in second_tools}
    assert "build_incident_scene" in second_names
    assert "s3_stb_logs:build_incident_scene" in second_plan.eligible_extra_tools


def test_always_persistent_exact_extra_requires_both_heavy_and_persistence():
    pending = build_tool_profile(
        methodology="receiver_reboot_dvr_playback",
        methodology_toolsets=("s3_stb_logs",),
        requested_extra_tools=("s3_stb_logs:build_complete_log_capsule",),
        authorization_flags={"heavy_tools_authorized": True},
    )
    assert "s3_stb_logs:build_complete_log_capsule" in pending.pending_authorization_extra_tools
    assert pending.missing_authorizations["s3_stb_logs:build_complete_log_capsule"] == ("persistence_authorized",)
    eligible = build_tool_profile(
        methodology="receiver_reboot_dvr_playback",
        methodology_toolsets=("s3_stb_logs",),
        requested_extra_tools=("s3_stb_logs:build_complete_log_capsule",),
        authorization_flags={"heavy_tools_authorized": True, "persistence_authorized": True},
    )
    assert "s3_stb_logs:build_complete_log_capsule" in eligible.eligible_extra_tools

def test_profile_selftest_passes():
    assert tool_profile_selftest()["status"] == "pass"
