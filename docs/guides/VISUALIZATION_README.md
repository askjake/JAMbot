# AI Thought Visualization System - Production Ready

## Overview
The AI Thought Visualization system captures and displays real-time agent reasoning, tool calls, and decision-making processes. This has been updated to use **live data** only - all demo/test modes have been removed.

## Architecture

```
Agent/Tools → ThoughtInterceptor → HTTP POST → VizRouter → State Storage → UI
                                                                          ↓
                                                                    /viz endpoint
```

### Components

1. **thought_interceptor.py** - Captures events from agent and tools
   - Singleton pattern, thread-safe
   - Non-blocking HTTP POST to viz server
   - Environment-configurable

2. **viz_router.py** - FastAPI router for visualization
   - Receives events via POST /viz/event
   - Stores state in memory
   - Serves UI at GET /viz/
   - Provides state at GET /viz/state

3. **agent.py** - Main agent with integrated thought capture
   - Captures user requests
   - Logs LLM decisions
   - Records tool execution plans

4. **Tools** - All tools have integrated thought capture
   - cluster_inspect, web_search, internal_search, etc.
   - Automatic logging of tool calls and results

## Configuration

Add to your `.env` file:

```bash
# AI Thought Visualization
AGENT_THOUGHT_CAPTURE_ENABLED=true
AGENT_VIZ_SERVER_URL=http://localhost:8000/rest/api/v1/viz/event
```

### Environment Variables

- `AGENT_THOUGHT_CAPTURE_ENABLED` - Enable/disable thought capture (default: true)
- `AGENT_VIZ_SERVER_URL` - URL for viz server (default: http://localhost:8000/rest/api/v1/viz/event)

## Usage

### Starting the Server

The visualization is integrated into the main Dish-Chat backend:

```bash
# Start Dish-Chat normally
./start-dishchat.sh

# Or manually:
cd ~/dish-chat
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Accessing the Visualization

Open your browser to:
```
http://localhost:8000/rest/api/v1/viz/
```

You'll see:
- **Reasoning Graph** - Visual network of agent thoughts and decisions
- **Event Log** - Real-time stream of thoughts, tools, and decisions
- **Context Memory** - Current agent context and state
- **Metrics** - Performance metrics (tokens, tools, time)

### API Endpoints

#### GET /viz/
Returns the visualization UI (HTML page)

#### GET /viz/state
Returns current visualization state as JSON:
```json
{
  "graph": {
    "nodes": [...],
    "links": [...]
  },
  "context": {...},
  "metrics": {
    "tokens": 0,
    "tools": 0,
    "time": 0
  },
  "events": [...]
}
```

#### POST /viz/event
Receives thought events (used by interceptor):
```json
{
  "type": "thought",
  "category": "thinking",
  "text": "Analyzing the problem...",
  "timestamp": "2026-02-17T10:00:00",
  "elapsed": 1.5
}
```

Event types:
- `thought` - Reasoning steps
- `tool` - Tool executions
- `decision` - Decision points
- `context` - Context updates
- `metric` - Performance metrics

#### POST /viz/clear
Clears all visualization state

## Testing

### E2E Tests

Run the comprehensive end-to-end tests:

```bash
cd ~/dish-chat
python test_viz_e2e.py
```

This tests:
1. Viz UI accessibility
2. State endpoint functionality
3. Thought event capture
4. Tool event capture
5. Context update capture
6. State clearing

### Manual Testing

Use the thought interceptor directly:

```python
from app.agent_mode.thought_interceptor import interceptor

# Capture a thought
interceptor.thought("Testing the visualization", "testing")

# Log a tool call
interceptor.tool_call("test_tool", {"param": "value"}, "success")

# Record a decision
interceptor.decision("Should I do A or B?", ["Option A", "Option B"])

# Update context
interceptor.context_update("test_key", "test_value")
```

## Integration in Your Code

The interceptor is already integrated into:
- `app/agent_mode/agent.py` - Main agent reasoning
- `app/tools/*.py` - All tool implementations

To add to new tools/modules:

```python
from app.agent_mode.thought_interceptor import interceptor

def my_new_tool(arg1, arg2):
    # Log the tool call
    interceptor.tool_call("my_new_tool", {"arg1": arg1, "arg2": arg2})
    
    # Log thinking
    interceptor.thought("Processing the request...", "tool")
    
    # Do work
    result = do_work(arg1, arg2)
    
    # Log result
    interceptor.tool_call("my_new_tool", result=str(result))
    
    return result
```

## Changes Made

### Removed
- ❌ `ai_thought_viz_live.py` - Deprecated Flask server (moved to .deprecated)
- ❌ `test_viz_http.py` - Test script (moved to .deprecated)

### Updated
- ✅ `thought_interceptor.py` - Fixed default URL to point to FastAPI viz_router
- ✅ `.env` - Added correct visualization configuration

### No Changes Needed
- ✅ `viz_router.py` - Already production-ready
- ✅ `agent.py` - Already has live integration
- ✅ All tools - Already have live integration

## Troubleshooting

### Visualization not showing events

1. Check if backend is running:
   ```bash
   curl http://localhost:8000/rest/api/v1/viz/state
   ```

2. Check environment variables:
   ```bash
   # In your .env file
   AGENT_THOUGHT_CAPTURE_ENABLED=true
   ```

3. Check interceptor initialization:
   ```python
   from app.agent_mode.thought_interceptor import interceptor
   print(f"Enabled: {interceptor.enabled}")
   print(f"URL: {interceptor.viz_url}")
   ```

### Events not appearing in graph

- The UI polls every 2 seconds for updates
- Check browser console for errors
- Verify the state endpoint returns data:
  ```bash
  curl http://localhost:8000/rest/api/v1/viz/state | jq
  ```

### Memory concerns

The state is stored in memory and grows unbounded. To clear:
```bash
curl -X POST http://localhost:8000/rest/api/v1/viz/clear
```

Consider adding periodic auto-clearing in production.

## Performance

- **Non-blocking** - Events are sent asynchronously via background thread
- **Fire-and-forget** - Failed sends don't block agent execution
- **Configurable** - Can be disabled via environment variable
- **Lightweight** - Minimal overhead on agent performance

## Next Steps

Potential enhancements:
- [ ] Persist state to database for history
- [ ] Add authentication/authorization
- [ ] Implement auto-clear after N events
- [ ] Add export functionality (JSON/CSV)
- [ ] Implement WebSocket for real-time updates (instead of polling)
- [ ] Add filtering and search in UI
- [ ] Create multiple visualization views (timeline, tree, etc.)

## Support

For issues or questions, check:
- The E2E test results
- Backend logs: `~/dish-chat-logs/backend.log`
- Gateway logs: `~/dish-chat-logs/gateway.log`
