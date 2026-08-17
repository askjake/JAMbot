from pathlib import Path


def test_active_paths_use_roles_not_model_arn_or_use_opus():
    agent = Path("app/agent_mode/agent.py").read_text()
    graph = Path("app/agent/agents/agentic_rag.py").read_text()
    assert "get_model(role=\"complex\")" in agent
    assert "get_model(model_arn" not in agent
    assert "workflow.add_node(\"verify_final\", verifier_gate_node)" in agent
    assert "verify_final_answer_against_packets" in agent
    assert "choose_model_role_for_context" in graph
    assert "get_tool_model(role=\"tool_worker\")" in graph
    assert "use_opus" not in graph


def test_mcop_packet_contract_and_registry_inventory():
    child = Path("app/agent_mode/child_conversation.py").read_text()
    mcop = Path("app/agent_mode/mcop_tools.py").read_text()
    registry = Path("app/agent/agents/tools/registry.py").read_text()
    assert "ToolEvidencePacket" in child
    assert "tool_evidence_packet.json" in child
    assert "_has_repeated_identical_tool_call" in child
    assert "ws.rglob" not in child
    assert "agent_read_packet" in mcop
    assert "packet_path" in mcop
    assert "get_tool_inventory" in registry
    assert "ollama_preserve_stable_names" in registry


def test_context_budget_uses_model_role():
    compression = Path("app/message/compression.py").read_text()
    budget = Path("app/tools/context_budget.py").read_text()
    assert "effective_context_budget" in compression
    assert "resolve_model_role" in compression
    assert "context_budget_selftest" in budget
    assert "MODEL_CONTEXT_LIMIT = 200_000" not in budget
