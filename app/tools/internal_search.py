# app/tools/internal_search.py

import os
import logging
from typing import List, Dict, Any

import httpx

from app.agent_mode.thought_interceptor import interceptor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

# Mode controls how we route the query:
#   "service"     -> POST to INTERNAL_SEARCH_URL (generic aggregator)
#   "confluence"  -> query Confluence only
#   "gitlab"      -> query GitLab only
#   "jira"        -> query Jira only
#   "multi"       -> query multiple backends listed in INTERNAL_SEARCH_SOURCES
INTERNAL_SEARCH_MODE = os.getenv("INTERNAL_SEARCH_MODE", "multi").lower()

# Comma-separated list of backends to use when MODE == "multi"
# e.g. "confluence,gitlab,jira"
INTERNAL_SEARCH_SOURCES = [
    s.strip().lower()
    for s in os.getenv("INTERNAL_SEARCH_SOURCES", "confluence,gitlab,jira").split(",")
    if s.strip()
]

DEFAULT_TIMEOUT = float(os.getenv("INTERNAL_SEARCH_TIMEOUT", "10.0"))

# ---------------------------------------------------------------------------
# Generic internal search service (future central aggregator)
# ---------------------------------------------------------------------------

INTERNAL_SEARCH_URL = os.getenv(
    "INTERNAL_SEARCH_URL",
    "http://internal-search-service.local/api/search",
)


async def _search_via_service(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Generic mode: POST to an internal search service that implements:
      POST INTERNAL_SEARCH_URL {query, top_k}
      -> {results: [{title, url, snippet}, ...]}
    """
    if not INTERNAL_SEARCH_URL:
        logger.warning("INTERNAL_SEARCH_URL is not set; skipping service backend")
        return []

    payload = {"query": query, "top_k": top_k}

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        resp = await client.post(INTERNAL_SEARCH_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])
    # normalize to dict list
    return list(results) if isinstance(results, list) else []


# ---------------------------------------------------------------------------
# Confluence search backend
# ---------------------------------------------------------------------------

CONFLUENCE_BASE_URL = os.getenv(
    "CONFLUENCE_BASE_URL",
    "https://dishtech-dishtv.atlassian.net/wiki",
)
CONFLUENCE_USER_EMAIL = os.getenv("CONFLUENCE_USER_EMAIL")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")


async def _search_confluence(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Direct integration with Confluence Cloud search API.

    Returns a list of dicts with: title, url, snippet.
    """
    if not (CONFLUENCE_USER_EMAIL and CONFLUENCE_API_TOKEN):
        logger.warning("Confluence search requested but credentials are missing")
        return []

    # Confluence Cloud search endpoint:
    #   {base}/rest/api/search
    # For Cloud, base is usually ".../wiki"
    url = CONFLUENCE_BASE_URL.rstrip("/") + "/rest/api/search"

    # Very simple CQL: search in text
    cql = f'text ~ "{query}"'

    params = {
        "cql": cql,
        "limit": str(top_k),
    }

    auth = httpx.BasicAuth(CONFLUENCE_USER_EMAIL, CONFLUENCE_API_TOKEN)

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, auth=auth) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    results: List[Dict[str, Any]] = []

    for item in data.get("results", []):
        content = item.get("content", {}) or {}
        title = content.get("title") or item.get("title") or "Untitled"

        links = content.get("_links", {}) or item.get("_links", {}) or {}
        webui = links.get("webui") or ""
        full_url = CONFLUENCE_BASE_URL.rstrip("/") + webui

        snippet = item.get("excerpt") or ""

        results.append(
            {
                "title": title,
                "url": full_url,
                "snippet": snippet,
                "source": "confluence",
            }
        )

    return results


# ---------------------------------------------------------------------------
# GitLab search backend
# ---------------------------------------------------------------------------

GITLAB_BASE_URL = os.getenv("GITLAB_BASE_URL", "").rstrip("/")
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN")

# e.g. "projects,blobs"
GITLAB_SEARCH_SCOPES = [
    s.strip()
    for s in os.getenv("GITLAB_SEARCH_SCOPES", "projects").split(",")
    if s.strip()
]


async def _search_gitlab(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search GitLab using the /api/v4/search endpoint.

    We support two scopes by default:
      - projects: finds matching repositories
      - blobs:    finds matching file contents and links directly to the blob view
    """
    if not (GITLAB_BASE_URL and GITLAB_TOKEN):
        logger.warning("GitLab search requested but GITLAB_BASE_URL or GITLAB_TOKEN missing")
        return []

    base_api = GITLAB_BASE_URL + "/api/v4"
    headers = {"PRIVATE-TOKEN": GITLAB_TOKEN}

    results: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers=headers) as client:
        project_cache: Dict[int, str] = {}

        async def get_project_web_url(project_id: int) -> str:
            if project_id in project_cache:
                return project_cache[project_id]
            try:
                resp = await client.get(f"{base_api}/projects/{project_id}")
                resp.raise_for_status()
                proj = resp.json()
                web_url = proj.get("web_url") or ""
                project_cache[project_id] = web_url
                return web_url
            except Exception:
                logger.exception("Failed to fetch GitLab project %s", project_id)
                return ""

        for scope in GITLAB_SEARCH_SCOPES:
            params = {
                "scope": scope,
                "search": query,
                "per_page": str(top_k),
            }
            try:
                resp = await client.get(f"{base_api}/search", params=params)

                # Common when using gitlab.com without a PAT or with limited perms.
                if resp.status_code == 403:
                    logger.info(
                        "GitLab search returned 403 for scope=%s; skipping this scope. "
                        "You probably need a PAT or a different GitLab base URL.",
                        scope,
                    )
                    continue

                resp.raise_for_status()

            except httpx.HTTPStatusError as e:
                logger.info("GitLab search HTTP error for scope=%s: %s", scope, e)
                continue

            except Exception as e:
                logger.info("GitLab search skipped for scope=%s: %s", scope, e)
                continue

            data = resp.json()
            if not isinstance(data, list):
                continue

            for item in data:
                if scope == "projects":
                    title = item.get("name_with_namespace") or item.get("name") or "GitLab project"
                    url = item.get("web_url") or ""
                    snippet = item.get("description") or ""
                elif scope == "blobs":
                    project_id = item.get("project_id")
                    file_path = item.get("path")
                    ref = item.get("ref") or item.get("blob_id") or "master"

                    url = ""
                    if project_id and file_path:
                        proj_url = await get_project_web_url(project_id)
                        if proj_url:
                            url = f"{proj_url}/-/blob/{ref}/{file_path}"

                    title = f"{file_path} [{ref}]" if file_path else "GitLab file match"
                    snippet = item.get("data") or ""
                else:
                    # generic fallback
                    title = item.get("title") or "GitLab result"
                    url = item.get("web_url") or ""
                    snippet = item.get("data") or item.get("description") or ""

                results.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                        "source": "gitlab",
                    }
                )

            if len(results) >= top_k:
                break

    # Trim overall
    return results[:top_k]


# ---------------------------------------------------------------------------
# Jira search backend
# ---------------------------------------------------------------------------

JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "").rstrip("/")
JIRA_USER_EMAIL = os.getenv("JIRA_USER_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")


async def _jira_search_raw(api_path: str, query: str, top_k: int) -> Dict[str, Any]:
    """
    Low-level Jira search caller for a given API path, e.g.:
      /rest/api/3/search  or  /rest/api/2/search
    """
    url = JIRA_BASE_URL + api_path

    # Very simple "text search anywhere" JQL
    jql = f'text ~ "{query}" ORDER BY updated DESC'

    params = {
        "jql": jql,
        "maxResults": str(top_k),
    }

    auth = httpx.BasicAuth(JIRA_USER_EMAIL, JIRA_API_TOKEN)

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, auth=auth) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


async def _search_jira(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Jira search using Cloud v3 API first, then falling back to v2 if needed.

    Returns:
      [{"title": "KEY: Summary", "url": ".../browse/KEY", "snippet": "..."}]
    """
    if not (JIRA_BASE_URL and JIRA_USER_EMAIL and JIRA_API_TOKEN):
        logger.warning("Jira search requested but config/env is missing")
        return []

    # Try v3 (Cloud) first
    data: Dict[str, Any]
    try:
        data = await _jira_search_raw("/rest/api/3/search", query, top_k)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status in (404, 410):
            logger.info(
                "Jira /rest/api/3/search unavailable (status %s); "
                "falling back to /rest/api/2/search",
                status,
            )
            try:
                data = await _jira_search_raw("/rest/api/2/search", query, top_k)
            except Exception as e2:
                logger.info("Jira v2 search also failed: %s", e2)
                return []
        else:
            logger.info("Jira search HTTP error: %s", e)
            return []
    except Exception as e:
        logger.info("Jira search failed: %s", e)
        return []

    issues = data.get("issues", []) or []
    results: List[Dict[str, Any]] = []

    for issue in issues:
        key = issue.get("key") or ""
        fields = issue.get("fields", {}) or {}
        summary = fields.get("summary") or ""
        description = fields.get("description") or ""

        issue_url = f"{JIRA_BASE_URL}/browse/{key}" if key else ""

        snippet_parts: List[str] = []
        if summary:
            snippet_parts.append(summary)
        if description:
            desc_clean = " ".join(str(description).split())
            if len(desc_clean) > 240:
                desc_clean = desc_clean[:240] + "…"
            snippet_parts.append(desc_clean)

        snippet = " — ".join(snippet_parts)

        title = f"{key}: {summary}" if key and summary else (summary or key or "Jira issue")

        results.append(
            {
                "title": title,
                "url": issue_url,
                "snippet": snippet,
                "source": "jira",
            }
        )

    return results


# ---------------------------------------------------------------------------
# Orchestration + formatting
# ---------------------------------------------------------------------------

async def _run_backend(backend: str, query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    Run a single backend safely, tagging each result with its source.
    """
    backend = backend.lower()
    try:
        if backend == "confluence":
            results = await _search_confluence(query, top_k=top_k)
        elif backend == "gitlab":
            results = await _search_gitlab(query, top_k=top_k)
        elif backend == "jira":
            results = await _search_jira(query, top_k=top_k)
        elif backend == "service":
            results = await _search_via_service(query, top_k=top_k)
        else:
            logger.warning("Unknown internal_search backend: %s", backend)
            return []

        # Ensure source is present
        for r in results:
            r.setdefault("source", backend)

        return results

    except Exception:
        logger.exception("internal_search backend %s failed", backend)
        return []


def _format_results_markdown(results: List[Dict[str, Any]]) -> str:
    """
    Group results by source and format as markdown.
    """
    if not results:
        return (
            "I tried internal search but didn't find anything useful for that query, "
            "or none of the configured backends are reachable with the current "
            "credentials / environment."
        )

    # Group by source
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        src = (r.get("source") or "internal").lower()
        grouped.setdefault(src, []).append(r)

    lines: List[str] = ["**Internal search results:**", ""]

    for src, items in grouped.items():
        pretty_src = src.capitalize()
        lines.append(f"### {pretty_src}")
        for item in items:
            title = item.get("title") or "Untitled"
            url = item.get("url") or ""
            raw_snippet = item.get("snippet") or ""

            # Normalize snippet whitespace
            cleaned = " ".join(str(raw_snippet).split())
            if len(cleaned) > 400:
                cleaned = cleaned[:400] + "…"

            if url:
                lines.append(f"- [{title}]({url})")
            else:
                lines.append(f"- {title}")

            if cleaned:
                lines.append(f"  - {cleaned}")

        lines.append("")  # blank line between groups

    return "\n".join(lines)


async def internal_search(query: str, top_k: int = 5) -> str:
    """
    Main entry point used by the agent.

    Args:
        query:      free-form search query
        top_k:      requested max results *per backend*

    Returns:
        Markdown formatted string summarizing all matches.
    """
    interceptor.tool_call("internal_search", params={
        "query": query[:100], 
        "top_k": top_k, 
        "mode": INTERNAL_SEARCH_MODE,
        "sources": INTERNAL_SEARCH_SOURCES
    })
    interceptor.thought(f"Searching internal systems ({INTERNAL_SEARCH_MODE} mode) for: {query[:50]}", "tool")
    
    mode = INTERNAL_SEARCH_MODE

    # Single-backend modes preserve old behaviour
    if mode in {"confluence", "gitlab", "jira", "service"}:
        results = await _run_backend(mode, query, top_k)
        interceptor.tool_call("internal_search", result=f"Found {len(results)} results from {mode}")
        return _format_results_markdown(results)

    # Multi-backend fan-out
    backends = INTERNAL_SEARCH_SOURCES or ["confluence"]
    all_results: List[Dict[str, Any]] = []

    for backend in backends:
        backend_results = await _run_backend(backend, query, top_k)
        all_results.extend(backend_results)

    interceptor.tool_call("internal_search", result=f"Found {len(all_results)} total results from {len(backends)} backends")
    return _format_results_markdown(all_results)

