"""
Web Browsing Tool for Agent Mode (v3 - Playwright + SSE-aware)
===============================================================
Allows the agent to fetch, render, and interact with web pages.
Supports:
  - Simple HTTP fetch + HTML parsing (fast, no JS)
  - Playwright-based full page rendering (for JS-heavy/SPA pages like Streamlit)
  - SSE-aware: detects slow-loading dashboards and attempts direct API fetch
  - Element extraction, link following, form data reading
  - Page text extraction with structure preservation
  - API requests (GET/POST/PUT/DELETE)

Created: 2026-06-29
Updated: 2026-07-02 - v3: Playwright + SSE/WebSocket dashboard support
"""

import asyncio
import os
import json
import logging
import re
import time
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Comment
from langchain.tools import tool
from app.agent_mode.thought_interceptor import interceptor
from app.tools.tool_result_compressor import compress_web_browse_result

logger = logging.getLogger(__name__)

# Default timeout for HTTP requests
DEFAULT_TIMEOUT = 30.0

# Maximum content length to return (to avoid overwhelming context)
MAX_CONTENT_LENGTH = 20000  # ~5000 tokens max (was 50000 = ~12500 tokens)

# Playwright browser singleton (lazy-loaded)
_playwright_instance = None
_browser_instance = None

# Path for saved session state (written by local_web_browse_manual_login)
BROWSER_STATE_PATH = os.path.expanduser("~/.dishchat_browser_state.json")


def _compress_tool_result(tool_name: str, result: str, token_budget: Optional[int] = None) -> str:
    """Compress large web browse outputs before LangGraph stores them."""
    from app.tools.context_budget import get_current_budget

    budget = token_budget if token_budget is not None else get_current_budget(tool_name)
    return compress_web_browse_result(
        tool_name=tool_name,
        raw_result=result,
        token_budget=budget,
    )


async def _get_browser():
    """Lazy-load a headless Chromium browser via Playwright (async)."""
    global _playwright_instance, _browser_instance
    if _browser_instance is None or not _browser_instance.is_connected():
        try:
            from playwright.async_api import async_playwright
            _playwright_instance = await async_playwright().start()
            _browser_instance = await _playwright_instance.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            logger.info('Playwright Chromium browser initialized (headless)')
        except Exception as e:
            logger.error(f'Failed to initialize Playwright browser: {e}')
            raise RuntimeError(
                f'Playwright browser unavailable: {e}. '
                f'Use mode="simple" for basic HTTP fetch (no JS rendering).'
            )
    return _browser_instance


async def _new_page_with_session(browser):
    """Create a new page, loading saved session cookies/storage if available."""
    if os.path.exists(BROWSER_STATE_PATH):
        context = await browser.new_context(storage_state=BROWSER_STATE_PATH)
        logger.debug(f'Loaded browser session from {BROWSER_STATE_PATH}')
    else:
        context = await browser.new_context()
    return await context.new_page()


def _clean_html_to_text(soup: BeautifulSoup, include_links: bool = True) -> str:
    """Convert parsed HTML to clean, readable text preserving structure."""
    for element in soup.find_all(['script', 'style', 'noscript', 'meta', 'link']):
        element.decompose()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()
    
    lines = []
    
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        level = int(tag.name[1])
        prefix = '#' * level
        text = tag.get_text(strip=True)
        if text:
            lines.append(f'\n{prefix} {text}\n')
        tag.decompose()
    
    for table in soup.find_all('table'):
        table_text = _extract_table(table)
        if table_text:
            lines.append(f'\n{table_text}\n')
        table.decompose()
    
    for ul in soup.find_all(['ul', 'ol']):
        for i, li in enumerate(ul.find_all('li', recursive=False)):
            text = li.get_text(strip=True)
            if text:
                if ul.name == 'ol':
                    lines.append(f'  {i+1}. {text}')
                else:
                    lines.append(f'  - {text}')
        lines.append('')
        ul.decompose()
    
    links_found = []
    if include_links:
        for a in soup.find_all('a', href=True):
            href = a.get('href', '')
            text = a.get_text(strip=True)
            if text and href and not href.startswith('#') and not href.startswith('javascript:'):
                links_found.append((text, href))
    
    remaining_text = soup.get_text(separator='\n', strip=True)
    
    result_parts = []
    if lines:
        result_parts.append('\n'.join(lines))
    if remaining_text:
        cleaned = re.sub(r'\n{3,}', '\n\n', remaining_text)
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        result_parts.append(cleaned)
    
    result = '\n'.join(result_parts)
    
    if links_found:
        result += '\n\n--- Links Found ---\n'
        seen = set()
        for text, href in links_found[:50]:
            key = (text, href)
            if key not in seen:
                seen.add(key)
                result += f'  [{text}] -> {href}\n'
    
    return result[:MAX_CONTENT_LENGTH]


def _extract_table(table) -> str:
    """Extract table content as formatted text."""
    rows = []
    for tr in table.find_all('tr'):
        cells = []
        for td in tr.find_all(['td', 'th']):
            cells.append(td.get_text(strip=True))
        if cells:
            rows.append(' | '.join(cells))
    
    if not rows:
        return ''
    
    result = rows[0] + '\n'
    result += '-' * min(len(rows[0]), 80) + '\n'
    result += '\n'.join(rows[1:])
    return result


async def _fetch_page_simple(url: str, headers: Optional[Dict] = None) -> Dict[str, Any]:
    """Fetch a page using httpx (fast, no JavaScript rendering)."""
    default_headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) DishChat-Agent/1.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    if headers:
        default_headers.update(headers)
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, verify=False, follow_redirects=True) as client:
        response = await client.get(url, headers=default_headers)
        return {
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'content_type': response.headers.get('content-type', ''),
            'url': str(response.url),
            'text': response.text,
        }


async def _try_sse_api_fetch(page, page_url: str, timeout: float = 30.0) -> str:
    """Try to directly fetch the API data that a slow-loading page is waiting for.
    
    Inspects page JavaScript to find fetch() URLs, then reads them directly.
    Handles SSE (Server-Sent Events) responses.
    """
    try:
        # Extract API URLs from the page's JavaScript
        api_urls = await page.evaluate("""() => {
            const scripts = document.querySelectorAll('script:not([src])');
            const urls = [];
            scripts.forEach(s => {
                const txt = s.textContent;
                // Find fetch( patterns by string search (no regex)
                let idx = 0;
                while (true) {
                    idx = txt.indexOf('fetch(', idx);
                    if (idx === -1) break;
                    const start = idx + 6;
                    // Look for the URL argument - could be template literal or string
                    const nextChar = txt[start];
                    if (nextChar === '`' || nextChar === "'" || nextChar === '"') {
                        let end = start + 1;
                        while (end < txt.length && end < start + 200 && txt[end] !== nextChar) end++;
                        const urlStr = txt.substring(start + 1, end);
                        if (urlStr.includes('/api') || urlStr.includes('/rest')) {
                            urls.push(urlStr);
                        }
                    }
                    idx = start + 1;
                }
            });
            return [...new Set(urls)];
        }""")
        
        if not api_urls:
            return ''
        
        # Resolve template variables using page context
        resolved_urls = await page.evaluate("""(urls) => {
            return urls.map(u => {
                try {
                    const roleSel = document.getElementById('role-sel');
                    const userIn = document.getElementById('user-in');
                    const role_val = roleSel ? roleSel.value : 'reviewer';
                    const user_val = userIn ? userIn.value : '';
                    let resolved = u;
                    resolved = resolved.replace('${role()}', role_val);
                    resolved = resolved.replace('${encodeURIComponent(user())}', encodeURIComponent(user_val));
                    // Remove remaining unresolved template vars
                    while (resolved.includes('${')) {
                        const s = resolved.indexOf('${');
                        const e = resolved.indexOf('}', s);
                        if (e === -1) break;
                        resolved = resolved.substring(0, s) + resolved.substring(e + 1);
                    }
                    return resolved;
                } catch(e) { return u; }
            });
        }""", api_urls)
        
        parsed_page = urlparse(page_url)
        base_url = f'{parsed_page.scheme}://{parsed_page.netloc}'
        
        for api_url in resolved_urls[:3]:
            if api_url.startswith('/'):
                api_url = base_url + api_url
            elif not api_url.startswith('http'):
                continue
            
            try:
                async with httpx.AsyncClient(timeout=timeout, verify=False) as client:
                    async with client.stream('GET', api_url) as resp:
                        chunks = []
                        total_size = 0
                        async for chunk in resp.aiter_text():
                            chunks.append(chunk)
                            total_size += len(chunk)
                            joined = ''.join(chunks)
                            if 'event: message' in joined and '"result"' in joined:
                                break
                            if total_size > 200000:
                                break
                        
                        raw = ''.join(chunks)
                        if 'event: message' not in raw:
                            continue
                        
                        # Parse SSE data lines
                        data_matches = re.findall(r'^data:\s*(.+)$', raw, re.MULTILINE)
                        if not data_matches:
                            continue
                        
                        for data_str in reversed(data_matches):
                            data_str = data_str.strip()
                            if not data_str.startswith('{'):
                                continue
                            try:
                                parsed = json.loads(data_str)
                                if 'result' in parsed:
                                    inner = parsed['result']
                                    if 'content' in inner:
                                        for item in inner['content']:
                                            if item.get('type') == 'text':
                                                return item['text']
                                    return json.dumps(inner, indent=2, default=str)[:MAX_CONTENT_LENGTH]
                                return json.dumps(parsed, indent=2, default=str)[:MAX_CONTENT_LENGTH]
                            except json.JSONDecodeError:
                                continue
                        
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                logger.debug(f'SSE fetch timeout for {api_url}: {e}')
                continue
            except Exception as e:
                logger.debug(f'SSE fetch error for {api_url}: {e}')
                continue
    
    except Exception as e:
        logger.debug(f'SSE API fetch overall failed: {e}')
    
    return ''


async def _fetch_page_rendered(url: str, wait_seconds: int = 10, wait_for_selector: Optional[str] = None) -> Dict[str, Any]:
    """Fetch a page using Playwright with full JavaScript rendering.
    
    Uses 'domcontentloaded' to avoid hanging on SSE/WebSocket pages.
    Includes smart content stabilization and SSE fallback for slow dashboards.
    """
    browser = await _get_browser()
    page = await _new_page_with_session(browser)
    try:
        response = await page.goto(url, wait_until='domcontentloaded', timeout=60000)
        
        if wait_for_selector:
            try:
                await page.wait_for_selector(wait_for_selector, timeout=wait_seconds * 1000)
            except Exception:
                pass
        else:
            # Smart wait: poll until content stabilizes
            max_polls = max(wait_seconds // 2, 5)
            prev_len = 0
            stable_count = 0
            for _ in range(max_polls):
                await page.wait_for_timeout(2000)
                try:
                    curr_text = await page.evaluate('() => document.body.innerText')
                    curr_len = len(curr_text)
                except Exception:
                    curr_len = 0
                
                if curr_len == prev_len and curr_len > 50:
                    stable_count += 1
                    if stable_count >= 2:
                        break
                else:
                    stable_count = 0
                prev_len = curr_len
        
        content = await page.content()
        title = await page.title()
        current_url = page.url
        
        try:
            visible_text = await page.evaluate('() => document.body.innerText')
        except Exception:
            visible_text = ''
        
        # SSE fallback: if page still shows loading, try direct API fetch
        api_data = ''
        loading_indicators = ['Loading queue', 'Loading...', 'loading data', 'Fetching']
        page_still_loading = any(ind.lower() in visible_text.lower() for ind in loading_indicators)
        
        if page_still_loading:
            logger.info(f'Page {url} still loading after wait, trying SSE API fallback')
            api_data = await _try_sse_api_fetch(page, current_url, timeout=10.0)
            
            if api_data:
                visible_text += '\n\n--- Dashboard API Data (fetched directly from SSE endpoint) ---\n' + api_data
            else:
                visible_text += (
                    '\n\n--- NOTE: Dashboard uses SSE streaming for data ---\n'
                    'The page is waiting for data from a Server-Sent Events stream.\n'
                    'The SSE data has not arrived within the timeout period.\n'
                    'Suggestions:\n'
                    '  1. Use web_browse_api with stream_sse=True to fetch the endpoint directly\n'
                    '  2. Try: web_browse_api(url="<base>/api/queue?role=admin&user=local-dashboard-user", stream_sse=True, stream_timeout=90)\n'
                    '  3. Use dedicated MCP tools (e.g., human_review_dashboard_status) if available\n'
                    '  4. Try again with wait_seconds=60\n'
                )
        
        status = response.status if response else 200
        
        return {
            'status_code': status,
            'url': current_url,
            'title': title,
            'text': content,
            'visible_text': visible_text,
        }
    finally:
        await page.close()


@tool('local_web_browse')
async def web_browse(
    url: str,
    mode: str = 'auto',
    selector: Optional[str] = None,
    extract: str = 'text',
    include_links: bool = True,
    wait_seconds: int = 10,
    wait_for_selector: Optional[str] = None,
) -> str:
    """Browse a web page and extract its content.
    
    Fetches and parses web pages, returning readable text content.
    Use this to view dashboards, documentation, status pages, or any HTTP endpoint.
    
    Parameters:
      - url: Full URL to browse (e.g., 'http://10.79.85.47:8765/human-review')
      - mode: Rendering mode:
          'auto' = Try simple first; if page looks like SPA/JS-heavy, use rendered (default)
          'simple' = Fast HTTP fetch, no JavaScript (good for static pages, APIs)
          'rendered' = Full browser rendering with JavaScript (for dynamic/SPA pages)
      - selector: Optional CSS selector to extract specific elements
      - extract: What to extract: 'text', 'html', 'links', 'tables', 'json'
      - include_links: Include a list of links found on the page (default: True)
      - wait_seconds: Seconds to wait for JS/data to load (default: 10, use 30+ for slow dashboards)
      - wait_for_selector: CSS selector to wait for before extracting (rendered mode only)
    
    Returns formatted page content or error information.
    
    Examples:
      - View a dashboard: web_browse(url='http://10.79.85.47:8765/human-review')
      - Slow dashboard: web_browse(url='http://...', mode='rendered', wait_seconds=30)
      - Get JSON: web_browse(url='http://api.example.com/data', extract='json')
      - Rendered SPA: web_browse(url='http://app.example.com', mode='rendered')
    """
    interceptor.tool_call('web_browse', params={
        'url': url, 'mode': mode, 'selector': selector, 'extract': extract
    })
    interceptor.thought(f'Browsing: {url} (mode={mode})', 'tool')
    
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            url = 'http://' + url
            parsed = urlparse(url)
        
        if parsed.scheme not in ('http', 'https'):
            return _compress_tool_result('local_web_browse', json.dumps({'error': f'Unsupported URL scheme: {parsed.scheme}'}))
        
        use_rendered = False
        if mode == 'rendered':
            use_rendered = True
        elif mode == 'auto':
            page_data = await _fetch_page_simple(url)
            html_text = page_data.get('text', '')
            is_spa = (
                '<div id="root"></div>' in html_text or
                '<div id="app"></div>' in html_text or
                '_stcore' in html_text or
                'streamlit' in html_text.lower() or
                '__NEXT_DATA__' in html_text or
                'EventSource' in html_text or
                'readStream' in html_text or
                (len(BeautifulSoup(html_text, 'lxml').get_text(strip=True)) < 200 and '<script' in html_text)
            )
            if is_spa:
                use_rendered = True
        
        if use_rendered:
            try:
                page_data = await _fetch_page_rendered(url, wait_seconds=wait_seconds, wait_for_selector=wait_for_selector)
            except Exception as e:
                logger.warning(f'Rendered mode failed: {e}')
                if mode == 'rendered':
                    return _compress_tool_result('local_web_browse', json.dumps({'error': f'Browser rendering failed: {e}', 'suggestion': 'Try mode="simple"'}))
                page_data = await _fetch_page_simple(url)
        elif mode == 'simple':
            page_data = await _fetch_page_simple(url)
        
        content_type = page_data.get('content_type', '')
        if extract == 'json' or 'application/json' in content_type:
            try:
                json_data = json.loads(page_data['text'])
                return _compress_tool_result('local_web_browse', json.dumps({
                    'url': page_data.get('url', url),
                    'status_code': page_data.get('status_code'),
                    'content_type': 'application/json',
                    'data': json_data,
                }, indent=2, default=str))
            except json.JSONDecodeError:
                pass
        
        soup = BeautifulSoup(page_data['text'], 'lxml')
        
        title = page_data.get('title', '')
        if not title:
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)
        
        if selector:
            elements = soup.select(selector)
            if not elements:
                return _compress_tool_result('local_web_browse', json.dumps({
                    'url': page_data.get('url', url),
                    'status_code': page_data.get('status_code'),
                    'title': title,
                    'error': f'No elements found matching selector: {selector}',
                    'suggestion': 'Try without a selector first'
                }))
            selected_html = '\n'.join(str(el) for el in elements)
            soup = BeautifulSoup(selected_html, 'lxml')
        
        if extract == 'html':
            content = str(soup)[:MAX_CONTENT_LENGTH]
        elif extract == 'links':
            links = []
            for a in soup.find_all('a', href=True):
                href = a.get('href', '')
                text = a.get_text(strip=True)
                full_url = urljoin(url, href)
                if text or href:
                    links.append({'text': text or href, 'url': full_url})
            content = json.dumps(links[:100], indent=2)
        elif extract == 'tables':
            tables = []
            for table in soup.find_all('table'):
                table_data = []
                headers = []
                for th in table.find_all('th'):
                    headers.append(th.get_text(strip=True))
                for tr in table.find_all('tr'):
                    row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if row:
                        table_data.append(row)
                if table_data:
                    tables.append({'headers': headers if headers else None, 'rows': table_data})
            content = json.dumps(tables, indent=2) if tables else 'No tables found on page.'
        else:
            visible_text = page_data.get('visible_text', '')
            if visible_text and use_rendered and not selector:
                content = visible_text[:MAX_CONTENT_LENGTH]
            else:
                content = _clean_html_to_text(soup, include_links=include_links)
        
        result = f'--- Web Page: {title or url} ---\n'
        result += f'URL: {page_data.get("url", url)}\n'
        result += f'Status: {page_data.get("status_code", "unknown")}\n'
        result += f'Mode: {"rendered (Playwright)" if use_rendered else "simple (HTTP)"}\n'
        if selector:
            result += f'Selector: {selector}\n'
        result += f'---\n\n'
        result += content
        
        return _compress_tool_result('local_web_browse', result)
        
    except httpx.TimeoutException:
        return _compress_tool_result('local_web_browse', json.dumps({'error': f'Request timed out after {DEFAULT_TIMEOUT}s', 'url': url}))
    except httpx.ConnectError as e:
        return _compress_tool_result('local_web_browse', json.dumps({'error': f'Connection failed: {e}', 'url': url}))
    except Exception as e:
        logger.error(f'web_browse error for {url}: {e}', exc_info=True)
        return _compress_tool_result('local_web_browse', json.dumps({'error': f'{type(e).__name__}: {e}', 'url': url}))


@tool('local_web_browse_interact')
async def web_browse_interact(
    url: str,
    action: str = 'click',
    selector: str = '',
    value: Optional[str] = None,
    wait_seconds: int = 5,
) -> str:
    """Interact with a web page using a real browser (click, type, scroll, screenshot).
    
    Opens the page in a headless Chromium browser and performs interactions.
    
    Parameters:
      - url: URL to navigate to
      - action: 'click', 'type', 'scroll', 'screenshot', 'get_source', 'list_elements'
      - selector: CSS selector for the target element
      - value: Text to type (for 'type') or scroll pixels (for 'scroll')
      - wait_seconds: Seconds to wait after action (default: 5)
    
    Examples:
      - Click: web_browse_interact(url='...', action='click', selector='button.submit')
      - Type: web_browse_interact(url='...', action='type', selector='input#search', value='query')
      - List elements: web_browse_interact(url='...', action='list_elements')
    """
    interceptor.tool_call('web_browse_interact', params={
        'url': url, 'action': action, 'selector': selector
    })
    interceptor.thought(f'Interacting with {url}: {action} on {selector}', 'tool')
    
    try:
        browser = await _get_browser()
        page = await _new_page_with_session(browser)
        
        try:
            await page.goto(url, wait_until='domcontentloaded', timeout=60000)
            await page.wait_for_timeout(max(wait_seconds, 5) * 1000)
            
            title = await page.title()
            result = f'Page: {title}\nURL: {page.url}\n\n'
            
            if action == 'click':
                if not selector:
                    return _compress_tool_result('local_web_browse_interact', json.dumps({'error': 'selector required for click action'}))
                # Wait for the element to be visible/attached before clicking
                click_timeout = max(wait_seconds, 5) * 1000  # at least 5s, scales with wait_seconds
                element_hidden = False
                try:
                    await page.wait_for_selector(selector, state='visible', timeout=click_timeout)
                except Exception:
                    # Element not visible - try attached (for hidden modals, overlays, etc)
                    try:
                        await page.wait_for_selector(selector, state='attached', timeout=5000)
                        element_hidden = True
                    except Exception:
                        return _compress_tool_result('local_web_browse_interact', json.dumps({
                            'error': f'Element not found: {selector}',
                            'suggestion': 'Use action="list_elements" to discover valid selectors on this page',
                            'url': url
                        }))
                if element_hidden:
                    # Element exists in DOM but isn't visible (display:none modal, etc)
                    # Use JS dispatch to click it since Playwright can't physically click invisible elements
                    await page.locator(selector).first.dispatch_event('click')
                else:
                    await page.click(selector, timeout=click_timeout)
                await page.wait_for_timeout(wait_seconds * 1000)
                result += f'Clicked: {selector}\n'
                result += f'New URL: {page.url}\n\n'
                content = await page.content()
                soup = BeautifulSoup(content, 'lxml')
                result += _clean_html_to_text(soup)
                
            elif action == 'type':
                if not selector:
                    return _compress_tool_result('local_web_browse_interact', json.dumps({'error': 'selector required for type action'}))
                if value is None:
                    return _compress_tool_result('local_web_browse_interact', json.dumps({'error': 'value required for type action'}))
                # Wait for element to be visible before typing
                type_timeout = max(wait_seconds, 5) * 1000
                try:
                    await page.wait_for_selector(selector, state='visible', timeout=type_timeout)
                except Exception:
                    return _compress_tool_result('local_web_browse_interact', json.dumps({
                        'error': f'Input element not found: {selector}',
                        'suggestion': 'Use action="list_elements" to discover valid selectors on this page',
                        'url': url
                    }))
                await page.fill(selector, value)
                await page.wait_for_timeout(wait_seconds * 1000)
                result += f'Typed "{value}" into: {selector}\n'
                
            elif action == 'scroll':
                scroll_amount = int(value) if value else 500
                await page.evaluate(f'window.scrollBy(0, {scroll_amount})')
                await page.wait_for_timeout(1000)
                result += f'Scrolled down {scroll_amount}px\n'
                visible = await page.evaluate('() => document.body.innerText')
                result += f'\n--- Visible content ---\n{visible[:MAX_CONTENT_LENGTH]}'
                
            elif action == 'screenshot':
                visible = await page.evaluate('() => document.body.innerText')
                result += visible[:MAX_CONTENT_LENGTH]
                
            elif action == 'get_source':
                content = await page.content()
                result += content[:MAX_CONTENT_LENGTH]
                
            elif action == 'list_elements':
                elements = await page.evaluate("""() => {
                    const results = [];
                    function getSelector(el) {
                        if (el.id) return '#' + el.id;
                        if (el.dataset && el.dataset.testid) return '[data-testid=' + JSON.stringify(el.dataset.testid) + ']';
                        const ariaLabel = el.getAttribute('aria-label');
                        if (ariaLabel) return '[aria-label=' + JSON.stringify(ariaLabel) + ']';
                        if (el.name) return el.tagName.toLowerCase() + '[name=' + JSON.stringify(el.name) + ']';
                        const text = (el.innerText || el.value || '').trim();
                        if (text && text.length < 60 && !text.includes(String.fromCharCode(10))) {
                            return 'text=' + JSON.stringify(text);
                        }
                        const parent = el.parentElement;
                        if (parent) {
                            const siblings = Array.from(parent.children).filter(c => c.tagName === el.tagName);
                            if (siblings.length === 1) return getSelector(parent) + ' > ' + el.tagName.toLowerCase();
                            const idx = siblings.indexOf(el) + 1;
                            return getSelector(parent) + ' > ' + el.tagName.toLowerCase() + ':nth-of-type(' + idx + ')';
                        }
                        return el.tagName.toLowerCase();
                    }
                    document.querySelectorAll('button, [role="button"], input[type="submit"]').forEach((el) => {
                        results.push({type: 'button', text: (el.innerText || el.value || '').trim().substring(0, 80), selector: getSelector(el)});
                    });
                    document.querySelectorAll('a[href]').forEach((el) => {
                        const txt = (el.innerText || '').trim();
                        if (txt) results.push({type: 'link', text: txt.substring(0, 80), href: el.href, selector: getSelector(el)});
                    });
                    document.querySelectorAll('input, textarea, select').forEach((el) => {
                        results.push({type: 'input', input_type: el.type || 'text', name: el.name || '', id: el.id || '', placeholder: el.placeholder || '', value: (el.value || '').substring(0, 50), selector: getSelector(el)});
                    });
                    return results.slice(0, 100);
                }""")
                result += f'Found {len(elements)} interactive elements:\n\n'
                result += json.dumps(elements, indent=2)
                
            else:
                return _compress_tool_result('local_web_browse_interact', json.dumps({'error': f'Unknown action: {action}. Valid: click, type, scroll, screenshot, get_source, list_elements'}))
            
            return _compress_tool_result('local_web_browse_interact', result)
            
        finally:
            await page.close()
        
    except Exception as e:
        logger.error(f'web_browse_interact error: {e}', exc_info=True)
        return _compress_tool_result('local_web_browse_interact', json.dumps({'error': f'{type(e).__name__}: {e}', 'url': url}))


@tool('local_web_browse_api')
async def web_browse_api(
    url: str,
    method: str = 'GET',
    headers: Optional[str] = None,
    body: Optional[str] = None,
    params: Optional[str] = None,
    stream_sse: bool = False,
    stream_timeout: int = 60,
) -> str:
    """Make HTTP API requests (GET, POST, PUT, DELETE) and return the response.
    
    Useful for REST APIs, health endpoints, structured data, or SSE streams.
    
    Parameters:
      - url: Full URL for the API request
      - method: HTTP method (GET, POST, PUT, DELETE, PATCH)
      - headers: JSON string of additional headers
      - body: JSON string of request body (for POST/PUT/PATCH)
      - params: JSON string of query parameters
      - stream_sse: If True, read as SSE stream and extract data events (default: False)
      - stream_timeout: Max seconds to wait for SSE data (default: 60)
    
    Examples:
      - GET: web_browse_api(url='http://api.example.com/status')
      - POST: web_browse_api(url='http://api.example.com/data', method='POST', body='{"key": "value"}')
      - SSE: web_browse_api(url='http://dashboard/api/queue?role=admin', stream_sse=True, stream_timeout=90)
    """
    interceptor.tool_call('web_browse_api', params={
        'url': url, 'method': method, 'stream_sse': stream_sse
    })
    interceptor.thought(f'API {method} {url}' + (' (SSE)' if stream_sse else ''), 'tool')
    
    try:
        req_headers = {'User-Agent': 'DishChat-Agent/1.0', 'Accept': 'application/json'}
        if headers:
            try:
                req_headers.update(json.loads(headers))
            except json.JSONDecodeError:
                return _compress_tool_result('local_web_browse_api', json.dumps({'error': 'Invalid JSON in headers parameter'}))
        
        req_body = None
        if body:
            try:
                req_body = json.loads(body)
            except json.JSONDecodeError:
                req_body = body
        
        req_params = None
        if params:
            try:
                req_params = json.loads(params)
            except json.JSONDecodeError:
                return _compress_tool_result('local_web_browse_api', json.dumps({'error': 'Invalid JSON in params parameter'}))
        
        # SSE streaming mode
        if stream_sse:
            async with httpx.AsyncClient(timeout=stream_timeout, verify=False, follow_redirects=True) as client:
                async with client.stream(
                    method.upper(), url, headers=req_headers,
                    json=req_body if isinstance(req_body, (dict, list)) else None,
                    content=req_body if isinstance(req_body, str) else None,
                    params=req_params,
                ) as resp:
                    chunks = []
                    total = 0
                    found_data = False
                    async for chunk in resp.aiter_text():
                        chunks.append(chunk)
                        total += len(chunk)
                        joined = ''.join(chunks)
                        if 'event: message' in joined and '"result"' in joined:
                            found_data = True
                            break
                        if total > 300000:
                            break
                    
                    raw = ''.join(chunks)
                    
                    data_matches = re.findall(r'^data:\s*(.+)$', raw, re.MULTILINE)
                    result = {
                        'status_code': resp.status_code,
                        'url': str(resp.url),
                        'stream_mode': 'sse',
                        'data_events_found': len(data_matches),
                        'found_result': found_data,
                    }
                    
                    if data_matches:
                        for data_str in reversed(data_matches):
                            data_str = data_str.strip()
                            if data_str.startswith('{'):
                                try:
                                    parsed = json.loads(data_str)
                                    if 'result' in parsed and 'content' in parsed.get('result', {}):
                                        for item in parsed['result']['content']:
                                            if item.get('type') == 'text':
                                                try:
                                                    result['data'] = json.loads(item['text'])
                                                except json.JSONDecodeError:
                                                    result['data'] = item['text']
                                                break
                                    else:
                                        result['data'] = parsed
                                    break
                                except json.JSONDecodeError:
                                    continue
                    
                    if 'data' not in result:
                        result['raw_tail'] = raw[-2000:] if raw else 'No data received'
                        result['note'] = 'SSE data event not received within timeout. The stream may need more time.'
                    
                    return _compress_tool_result('local_web_browse_api', json.dumps(result, indent=2, default=str)[:MAX_CONTENT_LENGTH])
        
        # Standard request mode
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, verify=False, follow_redirects=True) as client:
            response = await client.request(
                method=method.upper(),
                url=url,
                headers=req_headers,
                json=req_body if isinstance(req_body, (dict, list)) else None,
                content=req_body if isinstance(req_body, str) else None,
                params=req_params,
            )
        
        result = {
            'status_code': response.status_code,
            'url': str(response.url),
            'content_type': response.headers.get('content-type', ''),
            'response_headers': dict(response.headers),
        }
        
        try:
            result['data'] = response.json()
        except (json.JSONDecodeError, Exception):
            result['text'] = response.text[:MAX_CONTENT_LENGTH]
        
        return _compress_tool_result('local_web_browse_api', json.dumps(result, indent=2, default=str))
        
    except httpx.TimeoutException:
        return _compress_tool_result('local_web_browse_api', json.dumps({'error': f'Request timed out after {stream_timeout if stream_sse else DEFAULT_TIMEOUT}s', 'url': url}))
    except httpx.ConnectError as e:
        return _compress_tool_result('local_web_browse_api', json.dumps({'error': f'Connection failed: {e}', 'url': url}))
    except Exception as e:
        logger.error(f'web_browse_api error for {url}: {e}', exc_info=True)
        return _compress_tool_result('local_web_browse_api', json.dumps({'error': f'{type(e).__name__}: {e}', 'url': url}))

# ─────────────────────────────────────────────────────────────────────────────
# Manual Login Tool  (Option B: user types credentials into visible browser)
# ─────────────────────────────────────────────────────────────────────────────

def _manual_login_error_payload(exc) -> dict:
    """Classify headed-browser failures without weakening HTTPS verification."""
    text = str(exc or "")
    upper = text.upper()
    tls_markers = (
        "ERR_CERT_AUTHORITY_INVALID",
        "ERR_CERT_COMMON_NAME_INVALID",
        "ERR_CERT_DATE_INVALID",
        "CERTIFICATE_VERIFY_FAILED",
        "CERTIFICATE VERIFY FAILED",
        "UNKNOWN CA",
    )
    if any(marker in upper for marker in tls_markers):
        return {
            "ok": False,
            "schema": "local_web_browse_manual_login.v2",
            "result_code": "TLS_TRUST_REQUIRED",
            "error": "The headed browser could not validate the HTTPS certificate chain.",
            "required_action": (
                "Configure the trusted corporate CA certificate in the OS/browser trust store, "
                "then retry. Do not disable TLS verification or downgrade to plaintext."
            ),
            "tls_verification_disabled": False,
            "session_saved": False,
        }
    return {
        "ok": False,
        "schema": "local_web_browse_manual_login.v2",
        "result_code": "BROWSER_MANUAL_LOGIN_ERROR",
        "error": f"{type(exc).__name__}: {text}"[:1200],
        "tls_verification_disabled": False,
        "session_saved": False,
    }


@tool('local_web_browse_manual_login')
async def local_web_browse_manual_login(
    url: str,
    success_url_not_contains: str = 'login',
    timeout_seconds: int = 180,
) -> str:
    '''Open a VISIBLE browser window so the user can log in manually, including
    SSO / Okta / 2FA flows. Once the URL no longer contains success_url_not_contains
    the tool saves the authenticated session to disk and returns. All subsequent
    local_web_browse and local_web_browse_interact calls will automatically use
    this saved session.

    Args:
        url: The login URL to navigate to.
        success_url_not_contains: Token that should NOT be in URL when done (default: login).
        timeout_seconds: How long to wait for the user to finish (default 180s).
    '''
    from playwright.async_api import async_playwright
    import asyncio as _asyncio

    pw = None
    browser = None
    try:
        # ── Resolve a connectable X display ──────────────────────────────────────
        # The server process may have inherited a stale DISPLAY (e.g.
        # localhost:11.0 from a long-dead SSH X11-forwarding tunnel).
        # We must verify the display is actually reachable before using it,
        # and fall back to the local GDM session (:1) if it is not.
        def _x11_display_reachable(display: str) -> bool:
            """Return True if we can open a TCP/Unix socket to the X display."""
            import socket as _sock
            try:
                if display.startswith(':'):
                    # Unix-domain socket: /tmp/.X11-unix/X<n>
                    n = display.split(':')[1].split('.')[0]
                    path = f'/tmp/.X11-unix/X{n}'
                    s = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
                    s.settimeout(1)
                    s.connect(path)
                    s.close()
                    return True
                else:
                    # TCP display: host:D.S  →  port 6000+D
                    host, disp_screen = display.rsplit(':', 1)
                    disp_num = int(disp_screen.split('.')[0])
                    port = 6000 + disp_num
                    s = _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM)
                    s.settimeout(1)
                    s.connect((host or '127.0.0.1', port))
                    s.close()
                    return True
            except Exception:
                return False

        # Priority 1: existing DISPLAY if it is actually reachable
        # Priority 2: local GDM session :1
        # Priority 3: bare :1 fallback
        uid = os.getuid()
        gdm_xauth = f'/run/user/{uid}/gdm/Xauthority'
        home_xauth = os.path.expanduser('~/.Xauthority')

        current_display = os.environ.get('DISPLAY', '')
        if current_display and _x11_display_reachable(current_display):
            resolved_display = current_display
            # Pick the best auth file for this display
            resolved_xauth = os.environ.get('XAUTHORITY') or home_xauth or ''
        else:
            if current_display:
                logger.warning(
                    f'DISPLAY={current_display!r} is not reachable '
                    f'(stale SSH tunnel?); falling back to :1'
                )
            resolved_display = ':1'
            # GDM Xauthority has the MIT-MAGIC-COOKIE for the :1 session
            resolved_xauth = gdm_xauth if os.path.exists(gdm_xauth) else home_xauth

        # Build an explicit env dict for the browser subprocess so that
        # os.environ mutations are guaranteed to reach the child process.
        browser_env = dict(os.environ)
        browser_env['DISPLAY'] = resolved_display
        if resolved_xauth and os.path.exists(resolved_xauth):
            browser_env['XAUTHORITY'] = resolved_xauth
        elif 'XAUTHORITY' in browser_env:
            del browser_env['XAUTHORITY']  # don't pass a path that doesn't exist

        logger.info(
            f'Headed browser: DISPLAY={resolved_display!r}, '
            f'XAUTHORITY={resolved_xauth!r} (exists={os.path.exists(resolved_xauth) if resolved_xauth else False})'
        )
        pw = await async_playwright().start()
        try:
            browser = await pw.chromium.launch(
                headless=False,
                env=browser_env,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    # Chrome 149+ (Chromium 1228) requires gl=egl-angle. SwiftShader flags removed.
                    # --use-angle=default lets Chrome choose the best ANGLE backend
                    # (hardware GL on local GPU, SwiftShader on headless/software).
                    '--ozone-platform=x11',        # Force X11, don't auto-detect Wayland
                    '--use-angle=default',          # Chrome 149+: ANGLE default backend (egl-angle,angle=default)
                    # '--use-angle=swiftshader' and '--use-gl=swiftshader' are rejected
                    # by Chrome 130+ and crash the GPU process → blank window.
                    '--no-xshm',                   # Disable MIT-SHM (safe for remote X11 / Xming)
                    '--disable-backgrounding-occluded-windows',  # Don't suspend rendering
                    '--disable-renderer-backgrounding',
                    '--force-device-scale-factor=1',
                    '--window-size=1280,900',      # Explicit size avoids maximize bugs on remote X
                    '--window-position=0,0',
                ],
            )
        except Exception as e:
            return json.dumps({
                'ok': False,
                'schema': 'local_web_browse_manual_login.v2',
                'result_code': 'BROWSER_DISPLAY_UNAVAILABLE',
                'error': f'Could not launch headed browser: {e}',
                'hint': 'Ensure DISPLAY is set or X11 forwarding is active.',
                'session_saved': False,
            })

        context = await browser.new_context(no_viewport=True)
        page = await context.new_page()
        await page.goto(url, wait_until='domcontentloaded', timeout=30_000)

        start = _asyncio.get_event_loop().time()
        while True:
            current_url = page.url
            if success_url_not_contains.lower() not in current_url.lower():
                break
            if _asyncio.get_event_loop().time() - start > timeout_seconds:
                await browser.close()
                await pw.stop()
                return json.dumps({
                    'ok': False,
                    'schema': 'local_web_browse_manual_login.v2',
                    'result_code': 'LOGIN_TIMEOUT',
                    'error': 'Timeout',
                    'detail': f'Login not completed within {timeout_seconds}s.',
                    'last_url': current_url,
                    'session_saved': False,
                })
            await _asyncio.sleep(0.8)

        final_url = page.url
        await context.storage_state(path=BROWSER_STATE_PATH)
        try:
            title = await page.title()
        except Exception:
            title = '(unknown)'
        await browser.close()
        await pw.stop()

        return json.dumps({
            'ok': True,
            'schema': 'local_web_browse_manual_login.v2',
            'result_code': 'LOGIN_SESSION_SAVED',
            'success': True,
            'message': 'Login detected — session saved.',
            'final_url': final_url,
            'page_title': title,
            'session_saved_to': BROWSER_STATE_PATH,
            'note': 'All subsequent local_web_browse calls now use this session.',
        }, indent=2)

    except Exception as e:
        for obj, method in [(browser, 'close'), (pw, 'stop')]:
            if obj:
                try:
                    await getattr(obj, method)()
                except Exception:
                    pass
        logger.error(f'local_web_browse_manual_login error: {e}', exc_info=True)
        return json.dumps(_manual_login_error_payload(e))


@tool('local_web_browse_clear_session')
async def local_web_browse_clear_session() -> str:
    '''Delete the saved browser session so subsequent local_web_browse calls
    start unauthenticated. Use this to log out or switch accounts.'''
    import os as _os
    if _os.path.exists(BROWSER_STATE_PATH):
        _os.remove(BROWSER_STATE_PATH)
        return json.dumps({'success': True, 'message': f'Session deleted: {BROWSER_STATE_PATH}'})
    return json.dumps({'success': True, 'message': 'No saved session found.'})
