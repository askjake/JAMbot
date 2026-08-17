from __future__ import annotations

import asyncio
import ast
import importlib.util
import json
import os
import sys
import types
import uuid
from pathlib import Path

import pytest

SOURCE_ROOT = Path(os.environ["JAKE_SOURCE_ROOT"])


def _install(name: str, module: types.ModuleType) -> None:
    sys.modules[name] = module


def _identity_tool(_name):
    def decorator(fn):
        return fn
    return decorator


def _load_file(path: Path, prefix: str):
    name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_child_block() -> str:
    text = (SOURCE_ROOT / "app/agent_mode/child_conversation.py").read_text()
    return text[text.index("async def run_child_conversation"):]


def test_structured_packet_is_parsed_before_parent_preview_truncation():
    block = _run_child_block()
    parse_pos = block.index("parsed_packet = try_parse_tool_evidence_packet")
    preview_pos = block.index("summary = _bounded_child_summary(full_summary)")
    assert parse_pos < preview_pos, (
        "authoritative ToolEvidencePacket is still parsed only after destructive "
        "summary truncation"
    )


def test_unparseable_child_output_is_not_promoted_to_completed():
    block = _run_child_block()
    marker = "parsed_packet = normalize_worker_packet_from_summary("
    pos = block.index(marker)
    fallback = block[pos:pos + 400]
    assert 'status="partial"' in fallback, fallback
    assert 'status="completed"' not in fallback, fallback


def test_child_token_accounting_has_state_writer_and_result_assignment():
    source = (SOURCE_ROOT / "app/agent_mode/child_conversation.py").read_text()
    state_block = source[source.index("class ChildState"):source.index("def _tool_call_signature")]
    run_block = _run_child_block()
    assert "tokens_used: int" in state_block
    assert '"tokens_used": 0' in run_block
    assert "result.tokens_used =" in run_block
    assert "_response_token_count" in source


def _extract_child_token_helper():
    path = SOURCE_ROOT / "app/agent_mode/child_conversation.py"
    tree = ast.parse(path.read_text())
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_response_token_count"
    ]
    assert len(nodes) == 1
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"Any": object}
    exec(compile(module, str(path), "exec"), ns)
    return ns["_response_token_count"]


def test_child_token_counter_reads_normalized_and_provider_metadata():
    counter = _extract_child_token_helper()
    assert counter(types.SimpleNamespace(usage_metadata={"total_tokens": 37})) == 37
    assert counter(types.SimpleNamespace(usage_metadata={"input_tokens": 11, "output_tokens": 7})) == 18
    assert counter(types.SimpleNamespace(usage_metadata=None, response_metadata={
        "token_usage": {"prompt_tokens": 13, "completion_tokens": 5}
    })) == 18
    assert counter(types.SimpleNamespace(usage_metadata=None, response_metadata={})) == 0


def _extract_registry_exception_helpers():
    path = SOURCE_ROOT / "app/agent/agents/tools/registry.py"
    tree = ast.parse(path.read_text())
    wanted = {
        "_redact_registry_error_text",
        "_exception_leaf_details",
        "_registry_exception_details",
    }
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in nodes} == wanted
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"re": __import__("re"), "Any": object, "BaseExceptionGroup": BaseExceptionGroup}
    exec(compile(module, str(path), "exec"), ns)
    return ns


def test_registry_exceptiongroup_surfaces_redacted_leaf_cause():
    ns = _extract_registry_exception_helpers()
    exc = ExceptionGroup(
        "outer",
        [
            RuntimeError("TLS handshake failed at https://secret.internal/path"),
            ValueError("authorization=supersecret invalid"),
        ],
    )
    details = ns["_registry_exception_details"](exc)
    assert details["error_code"] == "ExceptionGroup"
    assert "RuntimeError" in details["error_leaf_codes"]
    assert "ValueError" in details["error_leaf_codes"]
    assert "TLS handshake failed" in details["error_summary"]
    assert "secret.internal" not in details["error_summary"]
    assert "supersecret" not in details["error_summary"]


def test_parent_prompt_does_not_claim_full_child_tool_access():
    text = (SOURCE_ROOT / "app/agent/agents/prompts/chat_system_prompt.txt").read_text()
    line = next(line for line in text.splitlines() if "Each child gets a clean context window" in line)
    assert "full tool access" not in line.lower()
    assert "parent" in line.lower() or "declared" in line.lower() or "bound" in line.lower()


def _load_management(status_payload: dict, policy_payload: dict | None = None):
    langchain_core = types.ModuleType("langchain_core")
    tools = types.ModuleType("langchain_core.tools")
    tools.tool = _identity_tool
    _install("langchain_core", langchain_core)
    _install("langchain_core.tools", tools)

    app = types.ModuleType("app")
    agent = types.ModuleType("app.agent")
    agents = types.ModuleType("app.agent.agents")
    agents_tools = types.ModuleType("app.agent.agents.tools")
    _install("app", app)
    _install("app.agent", agent)
    _install("app.agent.agents", agents)
    _install("app.agent.agents.tools", agents_tools)

    intent_mod = types.ModuleType("app.agent.tool_activation_intent")
    intent_mod.RegistryToolIndex = type("RegistryToolIndex", (), {"live": classmethod(lambda cls: None)})
    intent_mod.activation_intent_from_request = lambda **kwargs: types.SimpleNamespace(
        requested_exact_tools=(), requested_toolsets=(), ambiguous_requests=(),
        unavailable_requests=(), unhealthy_requests=(),
    )
    _install("app.agent.tool_activation_intent", intent_mod)

    registry_mod = types.ModuleType("app.agent.agents.tools.registry")
    registry_mod.get_mcp_registry_status = lambda: status_payload
    registry_mod.get_tool_inventory_signature = lambda: "sha256:test"
    _install("app.agent.agents.tools.registry", registry_mod)

    runtime_mod = types.ModuleType("app.agent.tool_policy_runtime")
    runtime_mod.policy_runtime_context_or_unknown = lambda: policy_payload or {
        "state_known": True,
        "requested_extra_tools": [],
        "pending_authorization_extra_tools": [],
        "eligible_extra_tools": [],
        "unavailable_extra_tools": [],
        "authorization_flags": {},
        "activation_status": {},
    }
    _install("app.agent.tool_policy_runtime", runtime_mod)

    workflows_mod = types.ModuleType("app.agent.operational_workflows")
    workflows_mod.operational_tools_in = lambda values: tuple(values or ())
    workflows_mod.ssh_environment_status = lambda: {"status": "not_tested"}
    workflows_mod.workflow_from_requested_tools = lambda values: ""
    _install("app.agent.operational_workflows", workflows_mod)

    gate_mod = types.ModuleType("app.agent.tool_execution_gate")
    gate_mod.required_authorizations_for_tool = lambda name: ()
    _install("app.agent.tool_execution_gate", gate_mod)

    return _load_file(
        SOURCE_ROOT / "app/agent/agents/tools/management.py",
        "management_under_test",
    )


def _large_registry_status():
    return {
        "generation": 7,
        "content_signature": "sha256:content",
        "refresh_epoch": 3,
        "health_signature": "sha256:health",
        "canonicalization_version": "registry_signature.v2",
        "initialized": True,
        "initialized_at": "now",
        "families": {
            "healthy": {"health": "HEALTHY", "source": "LIVE_DISCOVERY"},
            "failed": {"health": "UNAVAILABLE_NO_BASELINE", "source": "NONE"},
        },
        "toolsets": {
            "healthy": {
                "status": "loaded", "tool_count": 12,
                "blob": "X" * 30000,
            },
            "failed": {
                "status": "failed", "tool_count": 0,
                "error_code": "ExceptionGroup",
                "error_summary": "leaf failure",
                "blob": "Y" * 30000,
            },
        },
    }


def test_management_inventory_summary_is_compact_and_full_detail_is_explicit():
    mod = _load_management(_large_registry_status())
    summary = mod.diship_backend_tool_inventory_status(scope="summary")
    assert summary["ok"] is True
    assert "toolsets" not in summary
    assert summary["toolset_count"] == 2
    assert summary["toolset_status_counts"] == {"failed": 1, "loaded": 1}
    assert summary["failed_toolsets"][0]["name"] == "failed"
    assert len(json.dumps(summary)) < 8000

    full = mod.diship_backend_tool_inventory_status(scope="registry")
    assert "toolsets" in full


def test_management_binding_summary_preserves_checkpoint_identity_contract():
    policy = {
        "schema": "diship_tool_policy_runtime.v1",
        "policy_state_schema": "diship_tool_policy_state.v1",
        "state_known": True,
        "methodology": "generic_engineering",
        "thread_scoped": True,
        "requested_extra_tools": ["s3_stb_logs:heavy"],
        "pending_authorization_extra_tools": ["s3_stb_logs:heavy"],
        "eligible_extra_tools": [],
        "unavailable_extra_tools": [],
        "authorization_flags": {"operator_authorized": False},
        "authorization_material_stored": False,
        "activation_status": {"s3_stb_logs:heavy": "PENDING_AUTHORIZATION"},
        "tool_profile_signature": "sha256:profile",
        "tool_registry_generation": "gen-7",
        "tool_registry_content_signature": "sha256:content",
        "tool_registry_refresh_epoch": 3,
        "tool_registry_health_signature": "sha256:health",
        "tool_policy_version": 3,
        "activation_request_revision": 4,
        "continuity_task_scope": "generic_engineering",
        "continuity_methodology": "generic_engineering",
        "continuity_environment": "",
        "continuity_revision": 2,
        "pending_registry_requests": [f"pending_{i}" for i in range(40)],
        "last_bound_tool_names": [f"tool_{i}" for i in range(200)],
        "huge_internal_blob": "P" * 40000,
    }
    mod = _load_management(_large_registry_status(), policy)
    payload = mod.diship_backend_tool_binding_status()
    published = payload["tool_policy_state"]

    assert published["schema"] == "diship_tool_policy_runtime.v1"
    assert published["policy_state_schema"] == "diship_tool_policy_state.v1"
    assert published["thread_scoped"] is True
    assert published["authorization_material_stored"] is False
    assert published["pending_authorization_extra_tools"] == ["s3_stb_logs:heavy"]
    assert published["activation_status"]["s3_stb_logs:heavy"] == "PENDING_AUTHORIZATION"
    assert published["tool_registry_generation"] == "gen-7"
    assert published["tool_profile_signature"] == "sha256:profile"
    assert published["tool_policy_version"] == 3
    assert published["tool_registry_content_signature"] == "sha256:content"
    assert published["tool_registry_refresh_epoch"] == 3
    assert published["tool_registry_health_signature"] == "sha256:health"
    assert published["pending_registry_requests_count"] == 40
    assert published["pending_registry_requests_truncated"] is True
    assert len(published["pending_registry_requests"]) == 32
    assert published["last_bound_tool_names_count"] == 200
    assert published["last_bound_tool_names_truncated"] is True
    assert len(published["last_bound_tool_names"]) == 32
    assert "huge_internal_blob" not in published


def test_management_binding_and_refresh_default_to_compact_summary():
    policy = {
        "state_known": True,
        "requested_extra_tools": [f"tool_{i}" for i in range(200)],
        "eligible_extra_tools": [f"tool_{i}" for i in range(200)],
        "pending_authorization_extra_tools": [],
        "unavailable_extra_tools": [],
        "authorization_flags": {"operator_authorized": True},
        "activation_status": {f"tool_{i}": "ELIGIBLE" for i in range(200)},
        "huge_internal_blob": "P" * 40000,
    }
    mod = _load_management(_large_registry_status(), policy)
    binding = mod.diship_backend_tool_binding_status()
    refresh = mod.diship_backend_tool_refresh_status()
    assert binding["detail"] == "summary"
    assert "huge_internal_blob" not in json.dumps(binding)
    assert len(json.dumps(binding)) < 15000
    assert refresh["detail"] == "summary"
    assert "toolsets" not in refresh
    assert refresh["toolset_count"] == 2
    assert len(json.dumps(refresh)) < 8000


def _load_web_search(fake_data: dict):
    langchain_core = types.ModuleType("langchain_core")
    lc_tools = types.ModuleType("langchain_core.tools")
    lc_tools.InjectedToolArg = object
    lc_runnables = types.ModuleType("langchain_core.runnables")
    lc_runnables.RunnableConfig = dict
    _install("langchain_core", langchain_core)
    _install("langchain_core.tools", lc_tools)
    _install("langchain_core.runnables", lc_runnables)

    langchain = types.ModuleType("langchain")
    tools = types.ModuleType("langchain.tools")
    tools.tool = _identity_tool
    _install("langchain", langchain)
    _install("langchain.tools", tools)

    app = types.ModuleType("app")
    _install("app", app)
    config_mod = types.ModuleType("app.config")
    config_mod.get_settings = lambda: types.SimpleNamespace(COVERITY_GATEWAY_URL="https://gateway")
    _install("app.config", config_mod)

    analytics = types.ModuleType("app.analytics")
    analytics_service = types.ModuleType("app.analytics.service")
    async def save_web_search(*args, **kwargs):
        return None
    analytics_service.save_web_search = save_web_search
    _install("app.analytics", analytics)
    _install("app.analytics.service", analytics_service)

    agent_mode = types.ModuleType("app.agent_mode")
    interceptor_mod = types.ModuleType("app.agent_mode.thought_interceptor")
    interceptor_mod.interceptor = types.SimpleNamespace(
        tool_call=lambda *a, **k: None,
        thought=lambda *a, **k: None,
    )
    _install("app.agent_mode", agent_mode)
    _install("app.agent_mode.thought_interceptor", interceptor_mod)

    app_tools = types.ModuleType("app.tools")
    sanitizer = types.ModuleType("app.tools.query_sanitizer")
    sanitizer.sanitize_query = lambda q: (q, False, [])
    sanitizer.should_block_query = lambda q: (False, "")
    _install("app.tools", app_tools)
    _install("app.tools.query_sanitizer", sanitizer)

    mod = _load_file(SOURCE_ROOT / "app/tools/web_search.py", "web_search_under_test")

    class FakeResponse:
        def raise_for_status(self):
            return None
        def json(self):
            return fake_data

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            return False
        async def post(self, *args, **kwargs):
            return FakeResponse()

    mod.httpx.AsyncClient = FakeClient
    return mod


def test_public_web_search_returns_bounded_documented_result_shape():
    data = {
        "results": [
            {
                "title": f"Result {i}",
                "url": f"https://example.com/{i}",
                "snippet": "S" * 10000,
                "content": "C" * 50000,
                "raw_html": "H" * 50000,
            }
            for i in range(8)
        ],
        "provider_debug": "D" * 50000,
    }
    mod = _load_web_search(data)
    raw = asyncio.run(mod.public_web_search("bounded search", max_results=3))
    payload = json.loads(raw)
    assert payload["ok"] is True
    assert payload["schema"] == "public_web_search.v2"
    assert payload["returned_count"] == 3
    assert payload["truncated"] is True
    assert len(payload["results"]) == 3
    assert all(set(row) <= {"title", "url", "snippet", "snippet_truncated"} for row in payload["results"])
    assert "provider_debug" not in raw
    assert '"content"' not in raw
    assert len(raw) < 12000


def _extract_web_tls_helper():
    path = SOURCE_ROOT / "app/tools/web_browse.py"
    tree = ast.parse(path.read_text())
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_manual_login_error_payload"
    ]
    assert len(nodes) == 1
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"json": json}
    exec(compile(module, str(path), "exec"), ns)
    return ns["_manual_login_error_payload"]


def test_manual_login_tls_error_is_typed_without_disabling_verification():
    helper = _extract_web_tls_helper()
    payload = helper(RuntimeError("Page.goto: net::ERR_CERT_AUTHORITY_INVALID at https://internal"))
    assert payload["ok"] is False
    assert payload["result_code"] == "TLS_TRUST_REQUIRED"
    assert payload["tls_verification_disabled"] is False
    assert "downgrade to plaintext" in payload["required_action"].lower()
    source = (SOURCE_ROOT / "app/tools/web_browse.py").read_text()
    assert "ignore_https_errors=True" not in source


def test_no_progress_limit_terminal_does_not_call_failures_missing_capabilities():
    mod = _load_file(SOURCE_ROOT / "app/agent/no_progress_controller.py", "no_progress_secondary")
    decision = mod.NoProgressDecision(
        stop=True,
        result_code=mod.NO_PROGRESS_LIMIT_REACHED,
        no_progress_attempts=2,
        missing_tools=("agent_check_tasks", "internal_search"),
    )
    rendered = decision.render_terminal_message()
    assert "Missing or nonfunctional capability" not in rendered
    assert "no functional progress" in rendered.lower()
    assert "agent_check_tasks" in rendered
