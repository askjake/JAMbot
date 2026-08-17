# Internal Tools Integration - Jakes-agent

## Overview

This document describes the newly integrated internal tools for Jakes-agent, including Netra, DISH Internal Tools (CART, CCTools, Portal), Google Drive, and Grasshopper.

## Installation Summary

### Files Added/Modified

1. **NEW**: `app/tools/internal_tools.py` - Core implementation of all internal tools
2. **MODIFIED**: `app/agent/agents/tools/registry.py` - Updated to register new tools
3. **NEW**: `internal_tools.env` - Environment configuration template

### Backup Created

- Original registry backed up to: `registry.py.backup-YYYYMMDD-HHMMSS`

## Available Tools

### 1. Netra Search (`netra_search`)

Search DISHs internal log/record search system.

**Parameters:**
- `rec_id` (optional): Record ID to search for (e.g., "1971450629")
- `search_date` (optional): Date in YYYYMMDD format (e.g., "20260204")
- `query` (optional): Text query for general search

**Example Usage:**
```python
# Search by record ID
result = await netra_search(rec_id="1971450629", search_date="20260204")

# General search
result = await netra_search(query="error logs", search_date="20260204")
```

**Environment Variables:**
```bash
NETRA_BASE_URL=http://netra.internal.dish.com/api
NETRA_API_KEY=your_api_key_here
```

### 2. DISH Internal Tool (`dish_internal_tool`)

Access DISH internal services: CART, CCTools, and Portal.

**Parameters:**
- `service` (required): Service name ("cart", "cctools", or "portal")
- `endpoint` (optional): API endpoint path
- `params` (optional): Query parameters or POST data
- `method` (optional): HTTP method (GET or POST, default: GET)

**Example Usage:**
```python
# Search CART
result = await dish_internal_tool(
    service="cart",
    endpoint="/api/search",
    params={"account": "12345"}
)

# Lookup in CCTools
result = await dish_internal_tool(
    service="cctools",
    endpoint="/api/lookup",
    params={"phone": "555-1234"}
)
```

**Environment Variables:**
```bash
DISH_INTERNAL_TOOLS_BASE=http://internal-tools.dish.com
CART_URL=http://internal-tools.dish.com/cart
CCTOOLS_URL=http://internal-tools.dish.com/cctools
PORTAL_URL=http://internal-tools.dish.com/portal
```

### 3. Google Drive Search (`google_drive_search`)

Search Google Drive for files and documents.

**Parameters:**
- `query` (required): Search query
- `max_results` (optional): Maximum results (default: 10)
- `file_type` (optional): Filter by type ("document", "spreadsheet", "presentation", "pdf")

**Example Usage:**
```python
# Search all files
result = await google_drive_search(query="project plan")

# Search documents only
result = await google_drive_search(query="requirements", file_type="document")

# Search spreadsheets
result = await google_drive_search(query="budget", file_type="spreadsheet")
```

**Environment Variables:**
```bash
GOOGLE_DRIVE_API_URL=https://www.googleapis.com/drive/v3
GOOGLE_DRIVE_API_KEY=your_api_key_here
# OR
GOOGLE_DRIVE_SERVICE_ACCOUNT=your_service_account_token
```

### 4. Google Drive Get File (`google_drive_get_file`)

Get file content or metadata from Google Drive.

**Parameters:**
- `file_id` (required): Google Drive file ID
- `export_format` (optional): Export format ("text/plain", "text/html", "application/pdf")

**Example Usage:**
```python
# Get file metadata
result = await google_drive_get_file(file_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms")

# Export as text
result = await google_drive_get_file(
    file_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
    export_format="text/plain"
)
```

### 5. Grasshopper Search (`grasshopper_search`)

Search Grasshopper internal tool.

**Parameters:**
- `query` (required): Search query
- `category` (optional): Category filter
- `max_results` (optional): Maximum results (default: 10)

**Example Usage:**
```python
# Basic search
result = await grasshopper_search(query="network configuration")

# Filtered search
result = await grasshopper_search(query="router", category="hardware")
```

**Environment Variables:**
```bash
GRASSHOPPER_BASE_URL=http://grasshopper.internal.dish.com/api
GRASSHOPPER_API_KEY=your_api_key_here
```

### 6. Grasshopper Get Resource (`grasshopper_get_resource`)

Get detailed information about a specific Grasshopper resource.

**Parameters:**
- `resource_id` (required): Resource identifier

**Example Usage:**
```python
result = await grasshopper_get_resource(resource_id="GH-12345")
```

## Configuration Steps

### 1. Copy Environment Variables

```bash
cd ~/Jakes-agent
cat internal_tools.env >> .env.local
```

### 2. Update Environment Variables

Edit `.env.local` and replace placeholder values:
- `your_netra_api_key_here` - Get from Netra admin
- `your_google_drive_api_key_here` - Get from Google Cloud Console
- `your_grasshopper_api_key_here` - Get from Grasshopper admin

### 3. Verify API Endpoints

Confirm all base URLs are correct for your environment. If running in a different environment, update:
- NETRA_BASE_URL
- DISH_INTERNAL_TOOLS_BASE
- GRASSHOPPER_BASE_URL

### 4. Enable Tools in Agent Configuration

The tools are registered under the "dish_internal" tool set. To use them in your agent, you need to include this tool set when initializing the agent.

Example in your agent configuration:
```python
from app.agent.agents.tools.registry import get_tools_set

# Get the DISH internal tools
dish_tools = get_tools_set("dish_internal")

# Combine with other tool sets as needed
all_tools = (
    get_tools_set("search") +
    get_tools_set("agent_mode") +
    get_tools_set("dish_internal")
)
```

## Testing

### Test Netra Search
```bash
curl -X GET "http://netra.internal.dish.com/api/search?rec_id=1971450629&search_date=20260204" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Test Google Drive Search
```bash
curl -X GET "https://www.googleapis.com/drive/v3/files?q=name+contains+test&key=YOUR_API_KEY"
```

### Test Grasshopper Search
```bash
curl -X GET "http://grasshopper.internal.dish.com/api/search?q=test" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Restart Service

After configuration, restart the Jakes-agent service:

```bash
cd ~/Jakes-agent
./stop-dishchat.sh
./start-dishchat.sh

# Or use restart script if available
./restart-dishchat.sh
```

## Troubleshooting

### Tools Not Available

If tools are not showing up:
1. Check that `internal_tools.py` is in `app/tools/` directory
2. Verify `registry.py` has been updated with the import and factory
3. Check logs for import errors: `tail -f ~/Jakes-agent/logs/backend.log`

### Authentication Errors

If you get 401/403 errors:
1. Verify API keys are correct in `.env.local`
2. Check that API keys have proper permissions
3. Ensure URLs are accessible from your network

### Import Errors

If you see import errors related to `app.agent_mode.thought_interceptor`:
1. The interceptor decorator is optional - you can remove `@interceptor` decorators if not needed
2. Or ensure the thought_interceptor module exists in your codebase

### Connection Timeouts

If requests timeout:
1. Increase `INTERNAL_TOOLS_TIMEOUT` in `.env.local`
2. Verify network connectivity to internal services
3. Check firewall rules

## Security Notes

- API keys should never be committed to git
- Use `.env.local` for sensitive configuration (not tracked in git)
- Rotate API keys periodically
- Use service accounts with minimal required permissions for Google Drive
- Monitor API usage to detect anomalies

## Additional Resources

- **Netra Documentation**: Search Confluence for "Netra API"
- **CART Documentation**: Search Confluence for "CART Integration"
- **Grasshopper Documentation**: Search Confluence for "Grasshopper Tool"
- **Google Drive API**: https://developers.google.com/drive/api/v3/reference

## Support

For issues or questions:
1. Check Confluence for tool-specific documentation
2. Contact the respective tool owners
3. File a ticket in JIRA under the appropriate project

## Changelog

### 2026-02-27 - Initial Implementation
- Added Netra search integration
- Added DISH Internal Tools (CART, CCTools, Portal)
- Added Google Drive search and file retrieval
- Added Grasshopper search and resource lookup
- Updated tool registry with new "dish_internal" tool set
- Created configuration templates and documentation
