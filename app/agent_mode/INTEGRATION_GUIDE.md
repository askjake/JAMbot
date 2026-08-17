# 🔌 Integrating Live Thought Visualization with Dish-Chat

## Overview

This guide shows you exactly where to add the interceptor to capture live agent reasoning in the Dish-Chat backend.

## Architecture

```
User Request
    ↓
app/agent_mode/service.py (HTTP endpoint)
    ↓
app/agent_mode/agent.py (LangGraph workflow)
    ↓
agent_mode_node() → LLM reasoning
    ↓
tools_condition() → Should call tools?
    ↓
ToolNode → Execute tools (app/agent_mode/tools.py)
    ↓
Back to agent_mode_node()
```

## Integration Points

### 1. Add Interceptor Module

**File:** `app/agent_mode/thought_interceptor.py` (NEW FILE)

```python
"""
Thought interceptor for live visualization.
Captures agent reasoning and broadcasts to visualization server.
"""

import queue
import time
from datetime import datetime
from typing import Any, Dict, Optional
import threading
import json

class ThoughtInterceptor:
    """
    Captures agent thoughts and sends them to the visualization server.
    Thread-safe singleton that can be used across the application.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.enabled = True
        self.start_time = time.time()
        self.event_queue = queue.Queue(maxsize=1000)
        self._initialized = True
        
        # Start background thread to send events to viz server
        self._sender_thread = threading.Thread(
            target=self._send_events,
            daemon=True
        )
        self._sender_thread.start()
    
    def _send_events(self):
        """Background thread that sends events to visualization server."""
        import requests
        
        viz_url = "http://localhost:5000/api/event"  # Adjust as needed
        
        while True:
            try:
                event = self.event_queue.get(timeout=1.0)
                
                # Try to send to viz server (non-blocking)
                try:
                    requests.post(
                        viz_url,
                        json=event,
                        timeout=0.5
                    )
                except Exception:
                    # Silently fail if viz server not available
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                # Log but don't crash
                print(f"Interceptor error: {e}")
    
    def thought(self, text: str, category: str = "thinking"):
        """Capture a reasoning step."""
        if not self.enabled:
            return
        
        event = {
            "type": "thought",
            "category": category,
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "elapsed": time.time() - self.start_time
        }
        
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass  # Drop event if queue full
    
    def tool_call(self, tool_name: str, params: Optional[Dict] = None, 
                  result: Optional[Any] = None):
        """Capture a tool execution."""
        if not self.enabled:
            return
        
        event = {
            "type": "tool",
            "tool": tool_name,
            "params": params or {},
            "result": str(result)[:200] if result else None,  # Truncate
            "timestamp": datetime.now().isoformat(),
            "elapsed": time.time() - self.start_time
        }
        
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass
    
    def decision(self, text: str, options: Optional[list] = None):
        """Capture a decision point."""
        if not self.enabled:
            return
        
        event = {
            "type": "decision",
            "text": text,
            "options": options or [],
            "timestamp": datetime.now().isoformat(),
            "elapsed": time.time() - self.start_time
        }
        
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass
    
    def context_update(self, key: str, value: Any):
        """Update context memory."""
        if not self.enabled:
            return
        
        event = {
            "type": "context",
            "key": key,
            "value": str(value)[:200],  # Truncate
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass
    
    def metric_update(self, metric_type: str, value: Any):
        """Update performance metrics."""
        if not self.enabled:
            return
        
        event = {
            "type": "metric",
            "metric": metric_type,
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass

# Global singleton instance
interceptor = ThoughtInterceptor()
```

### 2. Modify Agent Node

**File:** `app/agent_mode/agent.py`

**Add import at the top:**
```python
from app.agent_mode.thought_interceptor import interceptor
```

**Modify the `agent_mode_node` function:**

```python
async def agent_mode_node(state: AgentModeState, config: dict[str, Any] | None = None):
    """Main reasoning node for agent mode."""
    chat_id = state["chat_id"]
    messages = state["messages"]
    
    # 🔴 CAPTURE: Starting reasoning
    last_message = messages[-1].content if messages else "No messages"
    interceptor.thought(f"Processing user request: {last_message[:100]}", "thinking")
    interceptor.context_update("chat_id", chat_id)
    interceptor.context_update("iteration", state.get("iterations", 0))

    system = SystemMessage(
        content=(
            "You are running in *agent mode* for a Dish internal engineering assistant.\n"
            f"The current workspace / chat_id is: {chat_id}.\n\n"
            "You have tools that can:\n"
            "- clone Git repositories (agent_git_clone)\n"
            "- create a Python virtualenv (agent_create_venv)\n"
            "- write and run Python code (agent_run_python)\n"
            "- list generated artifacts (agent_list_artifacts)\n\n"
            "When calling any agent_* tool, ALWAYS include the chat_id argument using this value.\n"
            "Work iteratively: plan, call tools, observe results, and refine your plan. "
            "Prefer small, testable steps over giant changes."
        )
    )

    model = get_model(role="complex")
    set_model_config(
        model,
        {
            "temperature": 0.7,
            "reasoning": True,
        },
    )

    tools = get_tools_set("agent_mode")
    llm_with_tools = model.bind_tools(tools)

    input_messages = [system, *messages]
    
    # 🔴 CAPTURE: Calling LLM
    interceptor.thought("Invoking LLM with tools", "thinking")
    
    response = await llm_with_tools.ainvoke(input_messages, config=config)
    
    # 🔴 CAPTURE: LLM response
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_names = [tc.get('name', 'unknown') for tc in response.tool_calls]
        interceptor.decision(
            f"LLM decided to call {len(response.tool_calls)} tool(s)",
            options=tool_names
        )
    else:
        interceptor.thought("LLM provided direct response (no tools)", "result")

    iterations = state.get("iterations", 0) + 1
    interceptor.metric_update("iterations", iterations)
    
    return {"messages": [response], "iterations": iterations}
```

### 3. Instrument Tool Calls

**File:** `app/agent_mode/tools.py`

**Add import at the top:**
```python
from app.agent_mode.thought_interceptor import interceptor
```

**Modify each tool function. Example for `agent_git_clone`:**

```python
@tool("agent_git_clone")
def agent_git_clone(chat_id: str, repo_url: str, branch: str = "main") -> str:
    """Clone (or update) a Git repo into the agent workspace."""
    
    # 🔴 CAPTURE: Tool call start
    interceptor.tool_call(
        "agent_git_clone",
        params={"repo_url": repo_url, "branch": branch}
    )
    interceptor.thought(f"Cloning {repo_url} (branch: {branch})", "tool")
    
    ws = _safe_workspace(chat_id)
    repo_dir = ws / "repo"

    if repo_dir.exists():
        try:
            # ... existing code ...
            
            # 🔴 CAPTURE: Success
            interceptor.thought(f"Updated existing repo at {repo_dir}", "result")
            return f"Updated {repo_url} to {branch} at {repo_dir}"
            
        except Exception as e:
            # 🔴 CAPTURE: Error
            interceptor.thought(f"Error updating repo: {str(e)}", "error")
            # Fall through to re-clone
    
    try:
        # ... existing clone code ...
        
        # 🔴 CAPTURE: Success
        interceptor.thought(f"Successfully cloned to {repo_dir}", "result")
        return f"Cloned {repo_url} to {repo_dir} on branch {branch}."
        
    except Exception as e:
        # 🔴 CAPTURE: Error
        error_msg = f"git clone failed: {e}"
        interceptor.thought(error_msg, "error")
        raise RuntimeError(error_msg)
```

**Apply similar changes to:**
- `agent_create_venv`
- `agent_run_python`
- `agent_list_artifacts`
- `agent_run_shell`

### 4. Add Visualization Endpoint (Optional)

**File:** `app/agent_mode/router.py`

Add an endpoint to serve the visualization:

```python
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/agent-mode", tags=["agent-mode"])

@router.get("/viz", response_class=HTMLResponse)
async def visualization():
    """Serve the live thought visualization interface."""
    # Return the HTML from ai_thought_viz_live.py
    # Or redirect to external viz server
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Agent Visualization</title></head>
    <body>
        <h1>Agent Thought Visualization</h1>
        <p>Open the visualization server at http://localhost:5000</p>
        <p>Or embed it here with an iframe</p>
    </body>
    </html>
    """
```

## Configuration

### Environment Variables

Add to your `.env` or config:

```bash
# Enable/disable thought capture
AGENT_THOUGHT_CAPTURE_ENABLED=true

# Visualization server URL
AGENT_VIZ_SERVER_URL=http://localhost:5000
```

### Settings

**File:** `app/config.py`

```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    AGENT_THOUGHT_CAPTURE_ENABLED: bool = True
    AGENT_VIZ_SERVER_URL: str = "http://localhost:5000"
```

## Testing

### 1. Start the visualization server

```bash
cd /path/to/viz
python ai_thought_viz_live.py
```

### 2. Start Dish-Chat backend

```bash
cd /path/to/dish-chat
python -m uvicorn app.main:app --reload
```

### 3. Make an agent request

```bash
curl -X POST http://localhost:8000/api/v1/agent-mode/run \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "test123",
    "message": "Clone the FM RDS Mapper repo and analyze it"
  }'
```

### 4. Watch the visualization

Open http://localhost:5000 and see the agent's thoughts appear in real-time!

## Minimal Integration (Quick Start)

If you want to test quickly, just add these 3 lines:

**In `app/agent_mode/agent.py`:**

```python
# At the top
from app.agent_mode.thought_interceptor import interceptor

# In agent_mode_node(), right before calling the LLM:
interceptor.thought(f"User: {messages[-1].content[:100]}", "thinking")

# Right after LLM responds:
if response.tool_calls:
    interceptor.decision(f"Calling {len(response.tool_calls)} tools")
```

That's it! You'll see thoughts appearing in the visualization.

## Advanced: Custom Reasoning Capture

For deeper insights, capture reasoning from the LLM's internal thoughts:

```python
# If using Claude with extended thinking:
if hasattr(response, 'thinking'):
    for thought in response.thinking:
        interceptor.thought(thought, "reasoning")
```

## Troubleshooting

**Visualization not updating:**
- Check that viz server is running on port 5000
- Check browser console for WebSocket errors
- Verify interceptor is enabled: `interceptor.enabled = True`

**Performance impact:**
- The interceptor uses a non-blocking queue
- Events are sent in a background thread
- Failed sends are silently dropped
- Minimal impact on agent performance

**Too many events:**
- Adjust queue size: `queue.Queue(maxsize=100)`
- Filter events by category
- Increase event truncation limits

## Next Steps

1. Create `app/agent_mode/thought_interceptor.py`
2. Add imports to `agent.py` and `tools.py`
3. Add capture calls at key points
4. Test with visualization server
5. Refine based on what you see!

---

**Questions?** The interceptor is designed to be non-invasive and fail-safe. If the viz server isn't running, events are simply dropped with no errors.
