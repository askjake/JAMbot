"""Cross-branch merge regressions for the 2026-08-06 Jake-Bot integration."""

from __future__ import annotations

from types import SimpleNamespace

from app.agent.tool_execution_gate import required_authorizations_for_tool
from app.agent.tool_execution_policy import get_scoped_tools_for_prompt


ALL_FALSE = {
    "operator_authorized": False,
    "heavy_tools_authorized": False,
    "mutation_authorized": False,
    "persistence_authorized": False,
}
ALL_TRUE = {key: True for key in ALL_FALSE}
PLAN = "grasshopper_mcp:grasshopper_plan_profile_upload"
UPLOAD = "grasshopper_mcp:grasshopper_upload_profile_logs"


def _install_registry(monkeypatch, *, health: str = "HEALTHY") -> None:
    from app.agent.agents import tools as tools_module
    from app.agent.agents.tools import registry

    families = {
        "search": ["public_web_search"],
        "agent_mode": ["agent_check_tasks"],
        "backend_facades": ["diship_backend_activate_tool_binding"],
        "grasshopper_mcp": [
            "grasshopper_plan_profile_upload",
            "grasshopper_upload_profile_logs",
        ],
    }

    def get_tools_set(name: str):
        return [SimpleNamespace(name=value, description="") for value in families.get(name, [])]

    def get_tools_set_filtered(name: str, allowed):
        allowed_set = set(allowed or ())
        return [tool for tool in get_tools_set(name) if tool.name in allowed_set]

    monkeypatch.setattr(tools_module, "get_tools_set", get_tools_set)
    monkeypatch.setattr(tools_module, "get_tools_set_filtered", get_tools_set_filtered)
    monkeypatch.setattr(
        registry,
        "get_mcp_registry_status",
        lambda: {
            "toolsets": sorted(families),
            "families": {
                name: {"health": health if name == "grasshopper_mcp" else "HEALTHY"}
                for name in families
            },
        },
    )
    monkeypatch.setattr(registry, "get_tool_inventory_signature", lambda: "sha256:merge-regression")


def test_grasshopper_plan_is_read_only_despite_upload_in_name() -> None:
    assert required_authorizations_for_tool(PLAN) == ()
    assert set(required_authorizations_for_tool(UPLOAD)) == {
        "operator_authorized",
        "mutation_authorized",
        "persistence_authorized",
    }


def test_checkpointed_unhealthy_family_is_withheld(monkeypatch) -> None:
    _install_registry(monkeypatch, health="INVALID_SCHEMA")
    tools, plan = get_scoped_tools_for_prompt(
        "Continue.",
        has_prior_tool_results=True,
        prior_active_toolsets=("grasshopper_mcp",),
        prior_requested_toolsets=("grasshopper_mcp",),
        prior_extra_tools=(PLAN,),
        authorization_flags=ALL_TRUE,
    )
    assert "grasshopper_mcp" not in plan.candidate_toolsets
    assert "grasshopper_mcp" in plan.unavailable_toolsets
    assert "grasshopper_plan_profile_upload" not in {
        str(getattr(tool, "name", "") or "") for tool in tools
    }


def test_family_request_exposes_read_only_plan_without_mutation_grant(monkeypatch) -> None:
    _install_registry(monkeypatch)
    tools, plan = get_scoped_tools_for_prompt(
        "Activate grasshopper_mcp",
        authorization_flags=ALL_FALSE,
    )
    names = {str(getattr(tool, "name", "") or "") for tool in tools}
    assert plan.requested_toolsets == ("grasshopper_mcp",)
    assert "grasshopper_plan_profile_upload" in names
    assert "grasshopper_upload_profile_logs" not in names
