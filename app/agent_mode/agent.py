"""Agent Mode LangGraph orchestration for provider-neutral local-LLM operation."""
from functools import cache
import time
from typing import Annotated, Any
import json
import os
from pathlib import Path
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.config import get_settings
from app.core.llm import get_model
from app.agent.utils import set_model_config
from app.agent.db_utils import get_checkpointer
from app.agent.agents.tools import get_tools_set
from app.agent_mode.thought_interceptor import interceptor
from app.agent_mode.adaptive_system_prompt import build_system_prompt

settings = get_settings()

class AgentModeState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    chat_id: str
    iterations: int

async def agent_mode_node(state: AgentModeState, config=None):
    chat_id = state["chat_id"]
    messages = state["messages"]
    iterations = state.get("iterations", 0)
    max_iters = getattr(settings, "AGENT_MODE_MAX_ITERS", 5)
    
    interceptor.thought(f"Iteration {iterations+1}/{max_iters}", "thinking")
    interceptor.context_update("chat_id", chat_id)
    interceptor.context_update("iteration", iterations)

    system = SystemMessage(content=build_system_prompt(
        chat_id=chat_id,
        iterations=iterations,
        max_iters=max_iters,
    ))
    
    # Agent Mode is a complex orchestration path; use provider-neutral role routing.
    model = get_model(role="complex")
    set_model_config(model, {"temperature": 0.7, "reasoning": True})
    tools = get_tools_set("agent_mode")
    llm_with_tools = model.bind_tools(tools)

    start = time.time()
    response = await llm_with_tools.ainvoke([system, *messages], config=config)
    interceptor.metric_update("llm_time_ms", int((time.time() - start) * 1000))

    if hasattr(response, "tool_calls") and response.tool_calls:
        names = [tc.get("name","?") for tc in response.tool_calls]
        interceptor.decision(f"Calling: {', '.join(names)}", options=names)
    else:
        interceptor.thought(f"Direct response: {str(response.content)[:100]}", "result")

    return {"messages": [response], "iterations": iterations + 1}


async def limit_reached_node(state: AgentModeState, config=None):
    """Graceful recovery when max iterations hit."""
    iters = state.get("iterations", 0)
    max_iters = getattr(settings, "AGENT_MODE_MAX_ITERS", 5)
    msg = AIMessage(content=(
        f"**Task Limit Reached** ({iters}/{max_iters} steps)\n\n"
        "I have completed as many steps as allowed in this pass.\n\n"
        "**Your work is saved. Options:**\n"
        "1. Ask me to **continue** - I pick up where I left off\n"
        "2. Ask me to **focus on one part** - break it down\n"
        "3. Ask **what did you accomplish?** - see partial results\n"
        "4. Run `agent_list_artifacts` to see created files"
    ))
    return {"messages": [msg]}


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content) if content is not None else ""


def _load_child_packets(chat_id: str) -> list[dict[str, Any]]:
    """Load compact MCOP child packets for the final verifier."""
    try:
        from app.agent_mode.child_conversation import _get_mcop_dir
        mcop_dir = _get_mcop_dir(chat_id)
    except Exception:
        return []
    packets: list[dict[str, Any]] = []
    for path in sorted(Path(mcop_dir).glob("task_*/tool_evidence_packet.json")):
        try:
            data = json.loads(path.read_text())
            data.setdefault("_path", str(path))
            packets.append(data)
        except Exception:
            continue
    return packets


async def verifier_gate_node(state: AgentModeState, config=None):
    """Run a deterministic evidence-only verifier before Agent Mode final output."""
    chat_id = state["chat_id"]
    messages = state.get("messages", [])
    final_draft = _message_text(messages[-1]) if messages else ""
    try:
        from app.agent.methodology import select_methodology
        from app.agent_mode.orchestration_packets import (
            init_run_state,
            verify_final_answer_against_packets,
            write_verifier_report,
        )
        root = Path(os.environ.get("AGENT_MODE_WORKDIR", "/tmp/home_agent")) / chat_id / "_mcop" / "verifier_runs"
        methodology = select_methodology(final_draft or "agent mode final verification")["name"]
        run_dir = init_run_state(root, chat_id=chat_id, goal="agent-mode final verifier", methodology=methodology)
        packets = _load_child_packets(chat_id)
        report = verify_final_answer_against_packets(final_draft, packets)
        report_path = write_verifier_report(run_dir, report)
        if report.verdict == "PASS":
            return {"messages": []}
        note = (
            "\n\n---\n"
            f"Verification: {report.verdict}. "
            "The final answer was checked against available structured evidence packets. "
            f"Verification report: {report_path}."
        )
        return {"messages": [AIMessage(content=final_draft + note)]}
    except Exception as exc:  # noqa: BLE001
        note = (
            "\n\n---\n"
            "Verification: PASS_WITH_RISKS. "
            f"The verifier gate could not complete in this environment: {type(exc).__name__}: {exc}"
        )
        return {"messages": [AIMessage(content=final_draft + note)]}


def _route_after_agent(state: AgentModeState) -> str:
    iters = state.get("iterations", 0)
    max_iters = getattr(settings, "AGENT_MODE_MAX_ITERS", 5)
    last = state["messages"][-1]
    has_tools = hasattr(last, "tool_calls") and bool(last.tool_calls)
    
    if iters >= max_iters and not has_tools:
        return "limit_reached"
    elif has_tools:
        return "tools"
    else:
        return "verify_final"


@cache
def get_agent_mode_graph(checkpointer=None):
    workflow = StateGraph(AgentModeState)
    workflow.add_node("agent", agent_mode_node)
    workflow.add_node("tools", ToolNode(tools=get_tools_set("agent_mode")))
    workflow.add_node("limit_reached", limit_reached_node)
    workflow.add_node("verify_final", verifier_gate_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", _route_after_agent,
        {"tools": "tools", "limit_reached": "limit_reached", "verify_final": "verify_final"}
    )
    workflow.add_edge("limit_reached", END)
    workflow.add_edge("verify_final", END)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile(checkpointer=checkpointer or get_checkpointer())
