# AWS Token Refresh Loop Bug Fix

**Date:** 2026-02-23  
**Issue:** Backend stuck in infinite loop when refreshing AWS tokens  
**Impact:** 6.8GB log file created with millions of repeated refresh messages

## Root Cause

Race condition in app/core/llm/chat_models.py in the start_token_refresh_task() function.

### The Problem

When multiple requests called get_model() simultaneously during app startup:
1. Each call checked if _task_started (all saw False)
2. Each created a new _refresh_models_periodically() task
3. Hundreds of concurrent refresh tasks started
4. All logged Proactively refreshing AWS tokens every 1-2ms
5. Created 6.8GB log file in minutes

### Evidence

Timestamps from backend.log.old showed refresh messages every 1-2 milliseconds:
- 2026-02-23 13:59:07,419
- 2026-02-23 13:59:07,420
- 2026-02-23 13:59:07,422
- ... repeated thousands of times within same second

## The Fix

Added proper async locking to prevent race condition.

### Changes Made

1. Made function async: def -> async def start_token_refresh_task()
2. Added lock protection: async with _refresh_lock
3. Updated call site: asyncio.create_task(start_token_refresh_task())

### Files Modified

- app/core/llm/chat_models.py (backup: chat_models.py.backup)

### Verification

- Log file size stable at 1.5M (not growing)
- No repeated refresh messages
- Backend healthy and responding

## Prevention

The _refresh_lock ensures only ONE task can check and set _task_started atomically.
