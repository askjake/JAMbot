from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests
from langchain_core.tools import tool

from app.agent_mode.thought_interceptor import interceptor

from app.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()
_BASE_URL = _settings.COVERITY_GATEWAY_URL.rstrip("/")


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Helper for POSTing JSON to the Coverity Assist gateway.

    Returns the decoded JSON response or an error payload.
    """
    url = f"{_BASE_URL}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            # Fall back to plain text if server didn't send JSON
            return {"status": "ok", "text": resp.text}
    except Exception as exc:
        logger.warning("log_assist POST %s failed: %s", url, exc)
        return {"error": str(exc), "url": url, "payload": payload}


def _get(path: str) -> Dict[str, Any]:
    """Helper for GET requests to the gateway."""
    url = f"{_BASE_URL}{path}"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {"status": "ok", "text": resp.text}
    except Exception as exc:
        logger.warning("log_assist GET %s failed: %s", url, exc)
        return {"error": str(exc), "url": url}


@tool("logassist_web_search")
def logassist_web_search(
    query: str,
    max_results: int = 6,
    include_page_text: bool = True,
    mode: str = "both",
) -> Dict[str, Any]:
    """
    Search technical docs / logs using the Coverity Assist gateway.

    Args:
        query: Natural language description of what you're trying to find.
        max_results: Max number of search results to fetch (default 6).
        include_page_text: If True, fetch and return page text where possible.
        mode: "ddg", "wiki", or "both" (default) to search both sources.
    """
    interceptor.tool_call("logassist_web_search", params={"query": query[:100], "mode": mode, "max_results": max_results})
    interceptor.thought(f"Searching LogAssist for: {query[:50]}", "tool")
    
    payload: Dict[str, Any] = {
        "query": query,
        "mode": mode,
        "max_results": max_results,
        "fetch_pages": include_page_text,
    }
    return _post("/web-search", payload)


@tool("logassist_trigger_workflow")
def logassist_trigger_workflow(
    original_request: str,
    task_description: str,
    auto_iterate: bool = True,
    max_iters: int = 2,
    inference_profile_arn: Optional[str] = None,
    chat_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a multi-step Coverity / log-assist workflow via the gateway.

    Use this when you want the external system to plan and execute a task
    (e.g., "scan repo X and summarize critical Coverity issues").

    Args:
        original_request: The user's original question or goal.
        task_description: Concise description of the work to perform.
        auto_iterate: If true, the workflow may loop for up to `max_iters` steps.
        max_iters: Safety cap on iterations (default 2).
        inference_profile_arn: Optional Bedrock inference profile override.
        chat_url: Optional existing Coverity-Assist chat URL to attach results to.
    """
    payload: Dict[str, Any] = {
        "original_request": original_request,
        "task_description": task_description,
        "auto_iterate": auto_iterate,
        "max_iters": max_iters,
        "inference_profile_arn": inference_profile_arn,
        "chat_url": chat_url,
    }
    # Let the Flask gateway fill defaults for token/profile if None
    return _post("/trigger-workflow", payload)


@tool("logassist_get_journal_files")
def logassist_get_journal_files() -> Dict[str, Any]:
    """
    List available journal and embedding files known to the gateway.

    Returns a dict with "journals" and "embed_files" entries.
    """
    interceptor.tool_call("logassist_get_journal_files", params={})
    interceptor.thought("Listing LogAssist journal files", "tool")
    return _get("/get-journal-files")


@tool("logassist_append_journal")
def logassist_append_journal(entry: str) -> Dict[str, Any]:
    """Append a free-form note to the shared Gabriel journal file."""
    interceptor.tool_call("logassist_append_journal", params={"entry": entry[:100]})
    interceptor.thought("Appending to LogAssist journal", "tool")
    payload = {"entry": entry}
    return _post("/journal", payload)


@tool("logassist_embed_content")
def logassist_embed_content(
    text: str,
    filename: str = "snippet.txt",
) -> Dict[str, Any]:
    """
    Store arbitrary text as an embedding candidate via the gateway.

    Args:
        text: Raw text to persist.
        filename: Logical filename to store under (only basename is used).
    """
    interceptor.tool_call("logassist_embed_content", params={"filename": filename, "text_length": len(text)})
    interceptor.thought(f"Embedding content: {filename}", "tool")
    
    payload = {
        "text": text,
        "filename": filename,
    }
    return _post("/embed-content", payload)

