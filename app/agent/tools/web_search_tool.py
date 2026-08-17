# app/agent/tools/web_search_tool.py
from __future__ import annotations

import logging
from typing import Dict, Any

import httpx

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


async def web_search(query: str) -> str:
    """
    Search the public web via the local gateway's /web-search endpoint.

    Use this when the user asks about current events, weather, news,
    or anything that changes over time or that you're unsure about.
    Returns a concise, text-only summary of the top results.
    """
    try:
        async with httpx.AsyncClient(
            base_url=settings.COVERITY_GATEWAY_URL,  # e.g. http://localhost:5000
            timeout=20.0,
        ) as client:
            resp = await client.post("/web-search", json={"query": query})
            resp.raise_for_status()
            data: Dict[str, Any] = resp.json()
    except Exception as e:
        logger.exception("web_search(%r) failed: %s", query, e)
        return f"[web_search] Failed to fetch results for {query!r}: {e}"

    results = data.get("results") or data.get("response", {}).get("results") or []

    if not results:
        return f"[web_search] No results for query: {query!r}"

    # Format a summary that the model can easily consume
    lines = []
    for i, item in enumerate(results[:5], start=1):
        title = item.get("title") or "<no title>"
        url = item.get("url") or ""
        snippet = (item.get("snippet") or item.get("text") or "").replace("\n", " ").strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        lines.append(f"{i}. {title}\n   {url}\n   {snippet}")

    return "\n".join(lines)

