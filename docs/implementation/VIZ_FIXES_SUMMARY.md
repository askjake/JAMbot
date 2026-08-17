# Viz Data Alignment Fixes - Round 2

## Date: 2026-02-17 (Afternoon)
## Status: ✅ COMPLETE

## Issues Found

### 1. Metrics Not Updating ✗
- **Tokens**: Not being tracked at all
- **Time**: Not being tracked properly
- **Tools**: Working correctly

### 2. Metrics Replacing Instead of Accumulating ✗
- viz_router was doing `metrics[key] = value`
- Should be doing `metrics[key] += value`
- This caused stale/incorrect cumulative counts

### 3. Reasoning Graph ✓
- Already working correctly
- Nodes and links being created properly

## Fixes Implemented

### Fix 1: Add Token Tracking to agent.py
**Location**: `app/agent_mode/agent.py`

Added after LLM response:
```python
# Track tokens and time metrics
if hasattr(response, "response_metadata"):
    token_usage = response.response_metadata.get("usage", {})
    total_tokens = token_usage.get("total_tokens", 0)
    if total_tokens > 0:
        interceptor.metric_update("tokens", total_tokens)

interceptor.metric_update("time", llm_time)
```

**Result**: Tokens from each LLM call now tracked and sent to viz

### Fix 2: Add Time Tracking to agent.py
**Location**: `app/agent_mode/agent.py`

Ensured proper order:
1. Call LLM
2. Calculate `llm_time = time.time() - start_time`
3. Send metric update

**Result**: Each LLM call time now tracked

### Fix 3: Metrics Accumulation in viz_router.py
**Location**: `app/viz_router.py`

Changed from:
```python
elif event.get("type") == "metric":
    metrics[event["metric"]] = event["value"]  # REPLACES
```

To:
```python
elif event.get("type") == "metric":
    metric_name = event.get("metric")
    metric_value = event.get("value", 0)
    
    # Accumulate metrics instead of replacing
    if metric_name in metrics:
        metrics[metric_name] += metric_value
    else:
        metrics[metric_name] = metric_value
```

**Result**: Metrics now accumulate across multiple events

## Testing

Created enhanced E2E test: `test_viz_e2e_enhanced.py`

Tests:
1. ✓ Metrics accumulation (sends 3x 100 tokens, expects 300)
2. ✓ Time metrics tracking
3. ✓ Reasoning graph nodes with proper IDs
4. ✓ All data points aligned (graph + metrics + context)

## Files Modified

```
M  app/agent_mode/agent.py              (added token/time tracking)
M  app/viz_router.py                    (fixed metrics accumulation)
A  test_viz_e2e_enhanced.py            (new comprehensive tests)
```

## Expected Behavior After Fixes

### Before:
- Tokens: Always 0 (not tracked)
- Time: Always 0 (not tracked)
- Tools: Correct count
- Graph: Working
- Metrics: Last value only (not cumulative)

### After:
- Tokens: Accumulates with each LLM call (100 + 200 + 50 = 350)
- Time: Accumulates with each LLM call (1.2 + 0.8 + 1.5 = 3.5s)
- Tools: Accumulates with each tool call
- Graph: Working (no change)
- Metrics: Cumulative totals for the conversation

## Verification Steps

1. Start backend:
   ```bash
   cd ~/dish-chat
   ./start-dishchat.sh
   ```

2. Run enhanced tests:
   ```bash
   python test_viz_e2e_enhanced.py
   ```

3. Open viz UI and have a conversation:
   ```
   http://localhost:8000/rest/api/v1/viz/
   ```
   
4. Watch metrics accumulate:
   - Ask multiple questions
   - Watch tokens increase with each response
   - Watch time accumulate
   - Watch tools count up

## Technical Details

### Token Tracking
- Extracts from `response.response_metadata.usage.total_tokens`
- Only tracks if tokens > 0
- Sent via `interceptor.metric_update("tokens", count)`

### Time Tracking
- Measured as `time.time() - start_time`
- Sent after each LLM call
- Represents seconds as float

### Accumulation Logic
- Checks if metric already exists
- Adds new value to existing value
- Handles both int and float values

## Notes

- All tracking is non-blocking (background thread)
- Failed sends don't affect agent performance
- Metrics persist until `/clear` endpoint called
- UI polls `/state` every 2 seconds for updates
