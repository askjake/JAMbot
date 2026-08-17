from functools import cache
from langgraph.graph.state import CompiledStateGraph

from app.agent.agents import agentic_rag, betareport_analysis, title_gen
from app.agent_mode.agent import get_agent_mode_graph

# Map agent type to factory function
_AGENT_FACTORIES = {
    "chat": agentic_rag.get_graph,
    "beta_report": betareport_analysis.get_graph,
    "title_gen": title_gen.get_graph,
    "agent_mode": get_agent_mode_graph,
}


@cache
def get_agent_graph(agent_type: str) -> CompiledStateGraph:
    """Return a compiled LangGraph for the requested agent type.

    If an unknown type is requested, we fall back to the default chat agent.
    """
    factory = _AGENT_FACTORIES.get(agent_type)
    if not factory:
        factory = _AGENT_FACTORIES["chat"]
    return factory()
