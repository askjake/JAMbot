"""
Web search tool that uses the Coverity Gateway's DuckDuckGo endpoint.
"""
import logging
import httpx
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def web_search(query: str) -> str:
    """
    Search the public web for up-to-date information.
    
    Args:
        query: The search query string
        
    Returns:
        A formatted string with search results including titles, snippets, and URLs
    """
    try:
        async with httpx.AsyncClient(
            base_url=settings.COVERITY_GATEWAY_URL,
            timeout=30.0
        ) as client:
            response = await client.post(
                "/web-search",
                json={"query": query}
            )
            response.raise_for_status()
            data = response.json()
        
        # Extract results from the response
        results = data.get("response", {}).get("results", [])
        
        if not results:
            return f"No results found for query: {query}"
        
        # Format results as a readable string
        formatted_results = [f"Search results for: {query}\n"]
        
        for i, result in enumerate(results[:5], 1):  # Top 5 results
            title = result.get("title", "No title")
            snippet = result.get("snippet", "No description")
            url = result.get("url", "")
            
            formatted_results.append(
                f"\n{i}. {title}\n"
                f"   {snippet}\n"
                f"   URL: {url}"
            )
        
        return "\n".join(formatted_results)
        
    except httpx.HTTPError as e:
        logger.error(f"Web search failed for query '{query}': {e}")
        return f"Web search failed: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in web search: {e}")
        return f"Web search error: {str(e)}"


# LangChain tool wrapper
from langchain_core.tools import StructuredTool

# Create the tool instance that can be used by agents
web_search_langchain_tool = StructuredTool.from_function(
    name="web_search",
    description=(
        "Search the public web for up-to-date information, including "
        "weather, news, prices, current events, and other time-sensitive facts. "
        "Use this when you need real-time information that may not be in your "
        "training data. Returns formatted search results with titles, snippets, and URLs."
    ),
    func=web_search,
    coroutine=web_search,
)
