# Dish-Chat Startup Fix Report
**Date:** 2026-03-02  
**Fixed By:** AI Assistant  
**Issue:** Backend failing to start due to MCP initialization hanging

---

## Problem Summary

The backend was hanging during startup after loading MCP (Model Context Protocol) tools. The application would get stuck at:

```
2026-03-02 11:17:27,493 INFO Loaded MCP tool set beta_report with 10 tools
2026-03-02 11:17:27,736 ERROR Unexpected content type: application/octet-stream
INFO: Waiting for application startup.
```

And would never reach "Application startup complete."

### Root Cause

1. The MCP client was trying to connect to an external Lambda service:
   - URL: `https://7quifnvo576d2m5rhbnguwgvfq0qbivs.lambda-url.us-west-2.on.aws/mcp`
2. The connection was hanging indefinitely (no timeout configured)
3. This prevented the FastAPI lifespan from completing startup
4. Additionally, `initialize_mcp_tools()` was being called twice in the lifespan

---

## Solution Implemented

### Modified File: `app/main.py`

**Backup created:** `app/main.py.backup-20260302-113811`

### Changes Made:

1. **Added asyncio import** for timeout functionality
2. **Created timeout wrapper function:**
   ```python
   async def initialize_mcp_tools_with_timeout(timeout_seconds=30):
       """Initialize MCP tools with a timeout to prevent hanging."""
       try:
           logger.info(f"Initializing MCP tools with {timeout_seconds}s timeout...")
           await asyncio.wait_for(initialize_mcp_tools(), timeout=timeout_seconds)
           logger.info("MCP tools initialized successfully")
       except asyncio.TimeoutError:
           logger.warning(f"MCP tools initialization timed out after {timeout_seconds}s - continuing without MCP tools")
       except Exception as e:
           logger.error(f"Error initializing MCP tools: {type(e).__name__} - {e}")
           logger.warning("Continuing without MCP tools")
   ```

3. **Replaced first MCP call** with timeout version:
   - Changed: `await initialize_mcp_tools()`
   - To: `await initialize_mcp_tools_with_timeout(timeout_seconds=30)`

4. **Removed duplicate MCP call** (second invocation was unnecessary)

---

## Results

### Before Fix:
- Backend stuck indefinitely during startup
- No timeout, application never became available
- Last successful startup: 08:26:56 (before MCP error started)

### After Fix:
- Backend starts successfully in ~33 seconds
- MCP initialization times out gracefully after 30s
- Application continues without MCP tools
- Log shows:
  ```
  2026-03-02 11:39:52,651 WARNING MCP tools initialization timed out after 30s - continuing without MCP tools
  2026-03-02 11:39:52,697 INFO Started idle chat checker
  INFO: Application startup complete.
  ```

---

## Service Status

✅ **Backend:** Running on http://10.79.85.35:8000
   - Process ID: 2836372
   - Health endpoint: http://10.79.85.35:8000/rest/api/v1/health
   - Status: Healthy (version 2.1.0)

✅ **Frontend:** Running on http://10.79.85.35:3001
   - Next.js application serving the UI
   - Status: Responding correctly

✅ **PostgreSQL:** Running in Docker container
   - Container: postgres-dev-dishchat
   - Ports: 5433, 5434

---

## Impact

- **MCP Tools:** Currently not available (timing out)
  - This affects the beta_report tool set
  - May need to investigate Lambda service separately
- **Core Functionality:** Fully operational
  - Chat interface working
  - Database connections successful
  - All other features available

---

## Recommended Next Steps

1. **Investigate MCP Lambda service** - Check why its returning unexpected content type
