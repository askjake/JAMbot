import logging
import asyncio
from typing import Annotated, Any
from typing_extensions import TypedDict
from functools import cache

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolMessage
from botocore.exceptions import ClientError
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode

from app.core.llm import get_model
from app.config import get_settings
from app.core.utils import get_datestr_now

from ..db_utils import get_checkpointer
from ..utils import aggressive_cachept, cleanup_cachept, set_model_config
from .utils import get_prompt
from ..methodology_utils import inject_methodology_into_prompt
from .tools import get_tools_set

logger = logging.getLogger(__name__)
settings = get_settings()

system_prompt = SystemMessage(
    content=get_prompt("chat_system").format(
        today=get_datestr_now(),
        token_budget=getattr(settings, "MAX_OUTPUT_COUNT", 2048),
    )
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    model_config: dict[str, Any]


### Nodes
async def call_model(state: AgentState, config=None):
    """
    Call the model with tools enabled.
    The model can choose to use tools or respond directly.
    """
    messages = state["messages"]

    # Remove leading orphaned ToolMessages (can happen after truncation)
    while messages and isinstance(messages[0], ToolMessage):
        messages = messages[1:]

    # Remove orphaned tool calls/results - two-pass approach
    # Pass 1: Find all tool call IDs that have results
    tool_results_present = set()
    for msg in messages:
        if isinstance(msg, ToolMessage) and hasattr(msg, 'tool_call_id'):
            tool_results_present.add(msg.tool_call_id)

    # Pass 2: Track which tool call IDs belong to KEPT AI messages
    kept_tool_call_ids = set()
    cleaned_messages = []
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            all_results_present = all(
                tc.get('id') in tool_results_present
                for tc in msg.tool_calls
            )
            if all_results_present:
                for tc in msg.tool_calls:
                    kept_tool_call_ids.add(tc.get('id'))
                cleaned_messages.append(msg)
            # else: skip — and its ToolMessages won't be kept either
        elif isinstance(msg, ToolMessage):
            # Only keep if its parent AIMessage was kept
            if getattr(msg, 'tool_call_id', None) in kept_tool_call_ids:
                cleaned_messages.append(msg)
        else:
            cleaned_messages.append(msg)

    messages = cleaned_messages

    # Now truncate if still too long, but keep complete pairs
    if len(messages) > 1000:
        messages = messages[-1000:]

    cleanup_cachept(messages)
    messages = aggressive_cachept(messages, settings.MAX_CACHEPOINT_CNT)

    # Get the model and bind tools
    model = get_model()
    set_model_config(model, state["model_config"])

    # Get tools: search (public web + internal) + agent_mode (git, venv, python)
    tools = get_tools_set("search") + get_tools_set("agent_mode") + get_tools_set("dish_internal")
    model_with_tools = model.bind_tools(tools)

    # Debug: log message types and tool call status
    logger.info(f"Calling model with {len(messages)} messages")
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
        tool_call_id = getattr(msg, 'tool_call_id', None)
        logger.info(f"  [{i}] {msg_type} - tool_calls: {has_tool_calls}, tool_call_id: {tool_call_id}")

    # Extract the last user message to check for methodology triggers
    user_message = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            user_message = msg.content if isinstance(msg.content, str) else ""
            break

    # Create dynamic system prompt with methodology injection
    dynamic_system_prompt = SystemMessage(
        content=inject_methodology_into_prompt(system_prompt.content, user_message)
    )

    # Call the model with dynamic system prompt
    # Retry once on ExpiredTokenException by forcing a model refresh.
    response = None
    for attempt in range(2):
        try:
            if attempt > 0:
                model = get_model(force_refresh=True)
                set_model_config(model, state["model_config"])
                model_with_tools = model.bind_tools(tools)

            response = await model_with_tools.ainvoke([dynamic_system_prompt, *messages], config=config)
            break

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code != "ExpiredTokenException" or attempt == 1:
                raise

            logger.warning("Expired AWS token during ainvoke; forcing model refresh and retrying once")
            await asyncio.sleep(0.5)

    # Cleanup the cachepoints so they're not stored persistently
    cleanup_cachept(messages)

    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Determine if we should continue to tools or end.
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If the model made tool calls, route to tools
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # Otherwise, we're done
    return END


### Graph
@cache
def get_graph():
    # Get tools for the tool node
    tools = get_tools_set("search") + get_tools_set("agent_mode") + get_tools_set("dish_internal")

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=get_checkpointer())
