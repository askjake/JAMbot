from app.tools.web_search import public_web_search
from app.tools.internal_search import InternalSearchTool
from app.agent_mode.tools import (
    agent_git_clone,
    agent_create_venv,
    agent_run_python,
    agent_list_artifacts,
)

_TOOL_FACTORIES["agent_mode"] = lambda: [
    agent_git_clone,
    agent_create_venv,
    agent_run_python,
    agent_list_artifacts,
]
_ASYNC_TOOL_FACTORIES["agent_mode"] = _TOOL_FACTORIES["agent_mode"]

_TOOL_FACTORIES["search"] = lambda: [public_web_search, InternalSearchTool()]
_ASYNC_TOOL_FACTORIES["search"] = lambda: [public_web_search, InternalSearchTool()]
_ASYNC_TOOL_FACTORIES = {
    "beta_report": lambda: get_mcp_tools(settings.BETAREPORT_MCP_CONFIG),
    "log_assist": lambda: get_mcp_tools(settings.LOG_ASSIST_MCP_CONFIG),
    "internal_tools": lambda: get_mcp_tools(settings.INTERNAL_TOOLS_MCP_CONFIG),
    # ...
}

