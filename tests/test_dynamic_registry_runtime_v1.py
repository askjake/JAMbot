from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path


class FakeSchema:
    def __init__(self, value):
        self.value = value

    def model_json_schema(self):
        return self.value


class FakeTool:
    def __init__(self, name: str, description: str = "", schema=None, fn=None):
        self.name = name
        self.description = description
        self.args_schema = FakeSchema(schema or {"type": "object", "properties": {}})
        self.fn = fn


def fake_tool(name=None):
    def decorate(fn):
        return FakeTool(name or fn.__name__, fn.__doc__ or "", fn=fn)
    return decorate


def module(name: str, **attrs):
    value = types.ModuleType(name)
    for key, item in attrs.items():
        setattr(value, key, item)
    sys.modules[name] = value
    return value


def load_registry():
    module("langchain_core")
    module("langchain_core.tools", BaseTool=FakeTool, tool=fake_tool)

    class Settings:
        PLLM_PROVIDER = "ollama"
        ELLM_PROVIDER = "ollama"

        def __getattr__(self, _name):
            return None

    module("app.config", get_settings=lambda: Settings())
    module("app.agent.agents.utils", get_mcp_tools=lambda _config: [])

    named_modules = {
        "app.tools.web_search": ["public_web_search"],
        "app.tools.internal_search": ["internal_search"],
        "app.tools.cluster_inspect": ["cluster_inspect"],
        "app.tools.aws_tools": [
            "bedrock_list_models", "bedrock_invoke_model", "s3_list_buckets", "s3_list_objects",
            "s3_get_object", "athena_list_databases", "athena_list_tables", "athena_execute_query",
            "glue_get_table_schema",
        ],
        "app.agent_mode.tools": ["agent_git_clone", "agent_create_venv", "agent_run_python", "agent_list_artifacts", "agent_run_shell"],
        "app.agent_mode.mcop_tools": ["agent_spawn_task", "agent_spawn_parallel", "agent_check_tasks", "agent_read_task_result", "agent_read_packet"],
        "app.tools.log_assist_gateway": ["logassist_web_search", "logassist_get_journal_files", "logassist_append_journal", "logassist_trigger_workflow", "logassist_embed_content"],
        "app.tools.internal_tools": ["dish_internal_tool", "google_drive_search"],
        "app.tools.web_browse": ["web_browse", "web_browse_interact", "web_browse_api", "local_web_browse_manual_login", "local_web_browse_clear_session"],
    }
    for mod_name, names in named_modules.items():
        module(mod_name, **{name: FakeTool(name) for name in names})
    module(
        "app.agent.agents.tools.management",
        BACKEND_MANAGEMENT_FACADES=[FakeTool("diship_backend_tool_inventory_status"), FakeTool("diship_backend_tool_binding_status"), FakeTool("diship_backend_activate_tool_binding"), FakeTool("diship_backend_tool_refresh_status")],
    )

    path = Path("app/agent/agents/tools/registry.py")
    spec = importlib.util.spec_from_file_location("registry_under_test", path)
    assert spec and spec.loader
    registry = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = registry
    spec.loader.exec_module(registry)
    return registry


def test_bounded_concurrent_initialization_refresh_failure_isolation_and_filtering(monkeypatch):
    registry = load_registry()
    active = 0
    max_active = 0
    version = {"a": 1}

    async def family_a():
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return [FakeTool("list_dates"), FakeTool("build_incident_scene", schema={"version": version["a"]})]

    async def family_b():
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return [FakeTool("list_dates"), FakeTool("search_logs")]

    async def family_bad():
        await asyncio.sleep(0)
        raise RuntimeError("https://secret.example/path token=super-secret")

    registry._ASYNC_TOOL_FACTORIES.clear()
    registry._ASYNC_TOOL_CACHE.clear()
    registry._MCP_TOOLSET_STATUS.clear()
    registry._ASYNC_TOOL_FACTORIES.update({"s3_stb_logs": family_a, "other_mcp": family_b, "bad_mcp": family_bad})

    asyncio.run(registry.initialize_mcp_tools(force=True))
    assert max_active >= 2
    assert set(registry._ASYNC_TOOL_CACHE) == {"s3_stb_logs", "other_mcp", "bad_mcp"}
    assert registry._MCP_TOOLSET_STATUS["bad_mcp"]["status"] == "failed"
    summary = registry._MCP_TOOLSET_STATUS["bad_mcp"]["error_summary"]
    assert "secret.example" not in summary
    assert "super-secret" not in summary
    assert "<redacted-url>" in summary
    assert registry.get_mcp_registry_status()["generation"] == 1

    # Cross-family collision is deterministic and exact filtering still finds
    # original names through description metadata.
    other_names = [tool.name for tool in registry.get_tools_set("other_mcp")]
    assert any(name != "list_dates" and name.endswith("list_dates") for name in other_names)
    filtered = registry.get_tools_set_filtered("s3_stb_logs", ["build_incident_scene"])
    assert [tool.name for tool in filtered] == ["build_incident_scene"]

    before = registry._MCP_TOOLSET_STATUS["s3_stb_logs"]["schema_signature"]
    version["a"] = 2
    asyncio.run(registry.refresh_mcp_tools(["s3_stb_logs"]))
    after = registry._MCP_TOOLSET_STATUS["s3_stb_logs"]["schema_signature"]
    assert before != after
    assert registry.get_mcp_registry_status()["generation"] == 2

    # A failed refresh preserves the last-known-good schema rather than
    # replacing a working executor inventory with an empty family.
    previous_names = [tool.name for tool in registry.get_tools_set("s3_stb_logs")]

    async def family_a_refresh_failure():
        raise RuntimeError("temporary discovery outage password=do-not-log")

    registry._ASYNC_TOOL_FACTORIES["s3_stb_logs"] = family_a_refresh_failure
    asyncio.run(registry.refresh_mcp_tools(["s3_stb_logs"]))
    state = registry._MCP_TOOLSET_STATUS["s3_stb_logs"]
    assert state["status"] == "refresh_failed_using_cached"
    assert state["using_last_known_good"] is True
    assert [tool.name for tool in registry.get_tools_set("s3_stb_logs")] == previous_names
    assert "do-not-log" not in state["error_summary"]
    assert registry.get_mcp_registry_status()["degraded_toolsets"] == ["s3_stb_logs"]

    invalidated = registry.invalidate_mcp_tool_cache(["s3_stb_logs"])
    assert invalidated["invalidated_toolsets"] == ["s3_stb_logs"]
    assert "s3_stb_logs" not in registry._ASYNC_TOOL_CACHE
