import json
import logging
from typing import Optional, Any
from langchain_core.tools import InjectedToolArg
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from typing_extensions import Annotated

from app.config import get_settings
from app.analytics.service import save_web_search
from app.agent_mode.thought_interceptor import interceptor
from app.tools.query_sanitizer import sanitize_query, should_block_query
import httpx

logger = logging.getLogger(__name__)
settings = get_settings()

COVERITY_GATEWAY_URL = settings.COVERITY_GATEWAY_URL

_WEB_SEARCH_MAX_RESULTS = 10
_WEB_SEARCH_SNIPPET_MAX_CHARS = 1200


def _compact_search_response(data: Any, *, query: str, max_results: int) -> dict:
    """Normalize gateway output to the documented bounded search contract."""
    payload = data if isinstance(data, dict) else {}
    rows = payload.get("results") if isinstance(payload.get("results"), list) else []
    try:
        requested = max(1, min(int(max_results or 1), _WEB_SEARCH_MAX_RESULTS))
    except Exception:
        requested = 6
    compact = []
    any_snippet_truncated = False
    for item in rows[:requested]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("name") or "")[:500]
        url = str(item.get("url") or item.get("link") or "")[:2000]
        raw_snippet = str(
            item.get("snippet")
            or item.get("description")
            or item.get("summary")
            or item.get("content")
            or item.get("text")
            or ""
        )
        snippet_truncated = len(raw_snippet) > _WEB_SEARCH_SNIPPET_MAX_CHARS
        any_snippet_truncated = any_snippet_truncated or snippet_truncated
        compact.append({
            "title": title,
            "url": url,
            "snippet": raw_snippet[:_WEB_SEARCH_SNIPPET_MAX_CHARS],
            "snippet_truncated": snippet_truncated,
        })
    total = payload.get("total_results")
    if not isinstance(total, int):
        total = len(rows)
    return {
        "ok": True,
        "schema": "public_web_search.v2",
        "result_code": "SEARCH_RESULTS_OK",
        "query": query,
        "total_results": total,
        "returned_count": len(compact),
        "truncated": len(rows) > len(compact) or any_snippet_truncated,
        "results": compact,
        "_meta": {"content_type": "web"},
    }


@tool("public_web_search")
async def public_web_search(
    query: str,
    max_results: int = 6,
    config: Annotated[RunnableConfig, InjectedToolArg] = None,
) -> str:
    """
    Perform a privacy-friendly web search for the given query.
    Returns a JSON string with a list of results (title, url, snippet).
    
    ⚠️ SECURITY WARNING: This tool sends queries to external public services.
    NEVER include proprietary company information, internal system names,
    employee data, or any confidential information in your queries.
    
    Use this tool when you need current information from the internet about:
    - News, weather, prices, stock quotes
    - Recent events or developments
    - Current statistics or data
    - Any information that changes frequently
    
    For company-specific information, use internal_search instead.
    """
    interceptor.tool_call("public_web_search", params={"query": query[:100], "max_results": max_results})
    interceptor.thought(f"Searching web for: {query[:50]}", "tool")
    
    # Extract chat_id from the config
    chat_id = None
    if config and "configurable" in config:
        chat_id = config["configurable"].get("thread_id")
    
    # Check if query should be blocked
    should_block, block_reason = should_block_query(query)
    if should_block:
        logger.error(f"BLOCKED sensitive query: {query[:50]}... Reason: {block_reason}")
        return json.dumps({
            "error": "Query blocked for security reasons",
            "reason": "The query appears to contain sensitive information that should not be sent to public search services.",
            "suggestion": "Please rephrase your query using generic terms, or use internal_search for company-specific information.",
        })
    
    # Sanitize and check query
    sanitized_query, was_modified, violations = sanitize_query(query)
    
    if was_modified:
        logger.warning(
            f"Potentially sensitive query detected in chat {chat_id}: "
            f"'{query[:50]}...' Violations: {violations}"
        )
        # Continue but log the concern
    
    logger.info(f"Web search for query='{query}' in chat_id={chat_id}")
    
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"{COVERITY_GATEWAY_URL}/web-search",
                json={"query": sanitized_query, "max_results": max_results},
            )
        resp.raise_for_status()
        data = resp.json()
        
        # Log the search for analytics
        try:
            await save_web_search(chat_id, query, data)
        except Exception as e:
            logger.warning(f"Failed to log web search: {e}")
        
        # Log successful search to interceptor. The analytics sink may retain the
        # provider response, but the model-facing ToolMessage receives only the
        # documented bounded result contract.
        results = data.get("results", []) if isinstance(data, dict) else []
        interceptor.tool_call("public_web_search", result=f"Found {len(results)} results")
        compact = _compact_search_response(
            data,
            query=sanitized_query,
            max_results=max_results,
        )
        return json.dumps(compact, indent=2)
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return json.dumps({
            "ok": False,
            "schema": "public_web_search.v2",
            "result_code": "SEARCH_FAILED",
            "error": "Search failed",
            "message": str(e)[:1000],
        })
