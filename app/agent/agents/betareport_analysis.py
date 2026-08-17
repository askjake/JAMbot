from typing import Annotated, Any
from typing_extensions import TypedDict
from functools import cache

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_mcp_adapters.client import MultiServerMCPClient


from app.core.llm import get_model
from app.config import get_settings
from app.core.utils import get_datestr_now
from app.config import get_settings

from app.agent.db_utils import get_checkpointer
from app.agent.utils import (
    aggressive_cachept,
    cleanup_cachept,
    cleanup_tmp,
    construct_messages,
    set_model_config,
)
from app.agent.agents.utils import get_prompt
from app.agent.agents.tools import get_tools_set

settings = get_settings()
system_prompt = SystemMessage(
    content=get_prompt("chat_system").format(today=get_datestr_now())
)
mcp_client = MultiServerMCPClient(settings.BETAREPORT_MCP_CONFIG)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task_instruction: HumanMessage
    model_config: dict[str, Any]
    agent_params: dict[str, Any]


### Nodes
async def get_task_instruction(state: AgentState):
    """Get the task instruction from MCP server"""
    # Prepend task instruction
    if state["agent_params"]["platform"] == "AndroidTV":
        instruction = (await mcp_client.get_prompt("beta_report", "androidtv_context"))[
            0
        ]
    else:
        instruction = (await mcp_client.get_prompt("beta_report", "dishtv_context"))[0]
    instruction.additional_kwargs["tmp"] = True

    return {"task_instruction": instruction}


async def generate_response(state: AgentState, config=None):
    """Generate the response from the current message history

    Cachepoint the message history aggressively
    """
    messages = state["messages"]

    cleanup_cachept(messages)
    messages = aggressive_cachept(
        messages, settings.MAX_CACHEPOINT_CNT, exclude_last_turn=True
    )

    model = get_model()
    set_model_config(model, state["model_config"])
    tools = get_tools_set("beta_report")
    llm_with_tools = model.bind_tools(tools)

    input_messages = construct_messages(
        messages, system_prompt, task_inst=state["task_instruction"]
    )
    response = await llm_with_tools.ainvoke(input_messages, config=config)

    # Cleanup the temporary info so they are not stored persistently
    cleanup_cachept(messages)
    cleanup_tmp(messages)

    return {"messages": [response]}


### Graph
@cache
def get_graph(checkpointer: BaseCheckpointSaver | None = None):
    tool_node = ToolNode(tools=get_tools_set("beta_report"))

    workflow = StateGraph(AgentState)
    workflow.add_node("task_instruction", get_task_instruction)
    workflow.add_node("agent", generate_response)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "task_instruction")
    workflow.add_edge("task_instruction", "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    workflow.add_edge("agent", END)

    return workflow.compile(checkpointer=checkpointer or get_checkpointer())
