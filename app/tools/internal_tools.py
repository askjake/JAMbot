# app/tools/internal_tools.py
import logging
import json
from typing import Optional

from langchain.tools import tool
from app.agent_mode.thought_interceptor import interceptor
import httpx

logger = logging.getLogger(__name__)

from app.config import get_settings
settings = get_settings()

# Get configuration from settings
NETRA_BASE_URL = getattr(settings, "NETRA_BASE_URL", "http://netra.internal.dish.com/api")
NETRA_API_KEY = getattr(settings, "NETRA_API_KEY", None)
GRASSHOPPER_BASE_URL = getattr(settings, "GRASSHOPPER_BASE_URL", "http://grasshopper.internal.dish.com/api")
GRASSHOPPER_API_KEY = getattr(settings, "GRASSHOPPER_API_KEY", None)
DISH_CART_URL = getattr(settings, "DISH_CART_URL", "http://internal-tools.dish.com/cart")
GOOGLE_DRIVE_API_KEY = getattr(settings, "GOOGLE_DRIVE_API_KEY", None)

DEFAULT_TIMEOUT = 15.0


@tool("netra_search")
async def netra_search(
    rec_id: Optional[str] = None,
    search_date: Optional[str] = None,
    query: Optional[str] = None
) -> str:
    """Search Netra (DISH internal log/record search system).
    
    Netra is used to find logs, records, and system data across DISH infrastructure.
    
    Args:
        rec_id: Record ID to search for (e.g., "1971450629")
        search_date: Date to search in YYYYMMDD format (e.g., "20260204")
        query: Optional text query for general search
        
    Returns:
        JSON string with search results or error information
        
    Examples:
        - Search by record ID: netra_search(rec_id="1971450629", search_date="20260204")
        - General search: netra_search(query="error logs", search_date="20260204")
    """
    interceptor.tool_call("netra_search", params={"rec_id": rec_id, "search_date": search_date, "query": query})
    interceptor.thought(f"Searching Netra for: rec_id={rec_id}, date={search_date}, query={query}", "tool")
    
    if not NETRA_BASE_URL:
        return json.dumps({"error": "NETRA_BASE_URL not configured"})
    
    try:
        params = {}
        if rec_id:
            params["rec_id"] = rec_id
        if search_date:
            params["search_date"] = search_date
        if query:
            params["query"] = query
            
        if not params:
            return json.dumps({"error": "At least one search parameter required (rec_id, search_date, or query)"})
        
        headers = {}
        if NETRA_API_KEY:
            headers["Authorization"] = f"Bearer {NETRA_API_KEY}"
            
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{NETRA_BASE_URL}/search",
                params=params,
                headers=headers
            )
            resp.raise_for_status()
            return resp.text
            
    except httpx.HTTPStatusError as e:
        logger.error(f"Netra search HTTP error: {e.response.status_code}")
        return json.dumps({"error": f"HTTP {e.response.status_code}", "details": str(e)})
    except Exception as e:
        logger.error(f"Netra search failed: {e}")
        return json.dumps({"error": type(e).__name__, "message": str(e)})


@tool("dish_internal_tool")
async def dish_internal_tool(
    service: str,
    endpoint: Optional[str] = None,
    params: Optional[dict] = None,
    method: str = "GET"
) -> str:
    """Access DISH internal tools and services (CART, CCTools, Portal).
    
    Available services:
    - cart: Customer Account Research Tool (account lookup, research)
    - cctools: Customer Care Tools (support, troubleshooting)
    - portal: Internal Tools Portal (gateway to tools)
    
    Args:
        service: Which internal service to access
        endpoint: API endpoint path (e.g., "/api/search", "/api/account")
        params: Query parameters or POST data
        method: HTTP method (GET or POST)
        
    Returns:
        JSON string with results or error information
    """
    interceptor.tool_call("dish_internal_tool", params={"service": service, "endpoint": endpoint})
    interceptor.thought(f"Accessing DISH tool: {service}", "tool")
    
    service_urls = {
        "cart": DISH_CART_URL,
        "cctools": getattr(settings, "DISH_CCTOOLS_URL", "http://internal-tools.dish.com/cctools"),
        "portal": getattr(settings, "DISH_PORTAL_URL", "http://internal-tools.dish.com/portal")
    }
    
    if service not in service_urls:
        return json.dumps({"error": f"Unknown service: {service}", "available": list(service_urls.keys())})
    
    base_url = service_urls[service]
    url = f"{base_url}{endpoint}" if endpoint else base_url
    
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            if method.upper() == "POST":
                resp = await client.post(url, json=params or {})
            else:
                resp = await client.get(url, params=params or {})
                
            resp.raise_for_status()
            return resp.text
            
    except httpx.HTTPStatusError as e:
        logger.error(f"DISH internal tool HTTP error: {e.response.status_code}")
        return json.dumps({"error": f"HTTP {e.response.status_code}", "details": str(e)})
    except Exception as e:
        logger.error(f"DISH internal tool failed: {e}")
        return json.dumps({"error": type(e).__name__, "message": str(e)})


@tool("google_drive_search")
async def google_drive_search(
    query: str,
    max_results: int = 10,
    file_type: Optional[str] = None
) -> str:
    """Search Google Drive for files and documents.
    
    Args:
        query: Search query (supports Google Drive query syntax)
        max_results: Maximum number of results to return (default: 10)
        file_type: Filter by file type (document, spreadsheet, presentation, pdf)
        
    Returns:
        JSON string with file results including name, id, mimeType, webViewLink
    """
    interceptor.tool_call("google_drive_search", params={"query": query, "max_results": max_results})
    interceptor.thought(f"Searching Google Drive for: {query}", "tool")
    
    if not GOOGLE_DRIVE_API_KEY:
        return json.dumps({"error": "GOOGLE_DRIVE_API_KEY not configured"})
    
    try:
        drive_query = f"name contains '{query}' and trashed=false"
        
        if file_type:
            mime_types = {
                "document": "application/vnd.google-apps.document",
                "spreadsheet": "application/vnd.google-apps.spreadsheet",
                "presentation": "application/vnd.google-apps.presentation",
                "pdf": "application/pdf",
            }
            if file_type.lower() in mime_types:
                drive_query += f" and mimeType='{mime_types[file_type.lower()]}'"
        
        params = {
            "q": drive_query,
            "pageSize": max_results,
            "fields": "files(id,name,mimeType,webViewLink,modifiedTime)",
            "key": GOOGLE_DRIVE_API_KEY
        }
        
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                "https://www.googleapis.com/drive/v3/files",
                params=params
            )
            resp.raise_for_status()
            return resp.text
            
    except httpx.HTTPStatusError as e:
        logger.error(f"Google Drive search HTTP error: {e.response.status_code}")
        return json.dumps({"error": f"HTTP {e.response.status_code}", "details": str(e)})
    except Exception as e:
        logger.error(f"Google Drive search failed: {e}")
        return json.dumps({"error": type(e).__name__, "message": str(e)})


@tool("grasshopper_search")
async def grasshopper_search(
    query: str,
    category: Optional[str] = None,
    max_results: int = 10
) -> str:
    """Search Grasshopper (DISH internal tool).
    
    Args:
        query: Search query
        category: Optional category filter
        max_results: Maximum results to return (default: 10)
        
    Returns:
        JSON string with search results
    """
    interceptor.tool_call("grasshopper_search", params={"query": query, "category": category})
    interceptor.thought(f"Searching Grasshopper for: {query}", "tool")
    
    if not GRASSHOPPER_BASE_URL:
        return json.dumps({"error": "GRASSHOPPER_BASE_URL not configured"})
    
    try:
        params = {"q": query, "limit": max_results}
        if category:
            params["category"] = category
            
        headers = {}
        if GRASSHOPPER_API_KEY:
            headers["Authorization"] = f"Bearer {GRASSHOPPER_API_KEY}"
            
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            resp = await client.get(
                f"{GRASSHOPPER_BASE_URL}/search",
                params=params,
                headers=headers
            )
            resp.raise_for_status()
            return resp.text
            
    except httpx.HTTPStatusError as e:
        logger.error(f"Grasshopper search HTTP error: {e.response.status_code}")
        return json.dumps({"error": f"HTTP {e.response.status_code}", "details": str(e)})
    except Exception as e:
        logger.error(f"Grasshopper search failed: {e}")
        return json.dumps({"error": type(e).__name__, "message": str(e)})
