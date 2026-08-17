from typing import Annotated
from typing_extensions import TypedDict
from functools import cache
import operator
import logging

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.core.llm import get_model
from app.config import get_settings

from .utils import get_prompt

settings = get_settings()
logger = logging.getLogger(__name__)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retry: Annotated[int, operator.add]
    title: str


### Nodes
async def title_gen(state: State, config=None):
    """Generate the response from the current message history

    Cachepoint the message history aggressively
    """
    messages = state.get("messages", [])
    model = get_model(efficient=True)
    title_sysprompt = get_prompt("title").format(max_len=settings.MAX_TITLE_LEN)
    response = await model.ainvoke(
        [SystemMessage(title_sysprompt), *messages], config=config
    )
    response.content = response.content.strip().strip('"')

    return {"messages": [response]}


async def retry_message(state: State, config=None):
    """Let LLM re-generate a title"""

    return {
        "messages": [
            HumanMessage(
                f"The title you generated is too long at {len(state['messages'][-1].content)} characters. Retry. Answer immediately without preamble."
            )
        ],
        "retry": 1,
    }


async def output_title(state: State, config=None):
    """Format the result and save as output. Output would be truncated if no valid length title was generated."""
    try:
        messages = state.get("messages", [])
        title = messages[-1].content
        if len(title) > settings.MAX_TITLE_LEN:
            title = title[: settings.MAX_TITLE_LEN]
            logger.warning(
                f"Generated title is too long at {len(title)} chars after max retry. Truncated to {settings.MAX_TITLE_LEN} chars."
            )
    except IndexError as e:
        title = None
        logger.exception(e)

    return {"title": title}


### Edges
async def check_len(state: State):
    """End if title len is within spec. Otherwise retry until retry limit is reached."""
    if (
        len(state["messages"][-1].content) > settings.MAX_TITLE_LEN
        and state["retry"] < settings.MAX_TITLE_RETRY
    ):
        return "retry_message"
    else:
        return "output_title"


### Graph
@cache
def get_graph(checkpointer: BaseCheckpointSaver | None = None):
    graph_builder = StateGraph(State)

    graph_builder.add_node("title_gen", title_gen)
    graph_builder.add_node("retry_message", retry_message)
    graph_builder.add_node("output_title", output_title)

    graph_builder.add_edge(START, "title_gen")
    graph_builder.add_conditional_edges(
        "title_gen",
        check_len,
        {"retry_message": "retry_message", "output_title": "output_title"},
    )
    graph_builder.add_edge("retry_message", "title_gen")

    return graph_builder.compile(checkpointer=checkpointer)
