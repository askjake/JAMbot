# 🎯 QUICK START: Add Live Visualization to Dish-Chat

## 📥 Download These Files

1. **thought_interceptor.py** - Drop into `app/agent_mode/`
2. **INTEGRATION_GUIDE.md** - Complete integration instructions
3. **agent_py.patch** - Changes needed for agent.py
4. **tool_instrumentation_example.py** - Example tool modifications

## ⚡ 5-Minute Integration

### Step 1: Add the Interceptor (2 min)

Copy `thought_interceptor.py` to your backend:
```bash
cp thought_interceptor.py /path/to/dish-chat/app/agent_mode/
```

### Step 2: Instrument Agent Node (2 min)

Edit `app/agent_mode/agent.py`:

```python
# Add at top (line ~15)
from app.agent_mode.thought_interceptor import interceptor

# In agent_mode_node(), add after docstring:
if messages:
    last_msg = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
    interceptor.thought(f"User: {last_msg[:100]}", "thinking")

# Before LLM call:
interceptor.thought("Invoking LLM", "thinking")

# After LLM response:
if hasattr(response, 'tool_calls') and response.tool_calls:
    interceptor.decision(f"Calling {len(response.tool_calls)} tools")
```

### Step 3: Instrument Tools (1 min)

Edit `app/agent_mode/tools.py`:

```python
# Add at top
from app.agent_mode.thought_interceptor import interceptor

# In each tool function, add:
interceptor.tool_call("tool_name", params={...})
interceptor.thought("Doing something...", "tool")
```

See `tool_instrumentation_example.py` for complete examples.

### Step 4: Test It! (30 sec)

Terminal 1 - Start visualization:
```bash
python ai_thought_viz_live.py
```

Terminal 2 - Start Dish-Chat:
```bash
cd /path/to/dish-chat
python -m uvicorn app.main:app --reload
```

Browser:
```
http://localhost:5000  # Visualization
```

Make an agent request and watch your thoughts appear in real-time! 🧠✨

## 🎨 What You'll See

- **Green nodes** = Agent thinking
- **Red nodes** = Tool executions  
- **Yellow nodes** = Decisions
- **Live event log** with timestamps
- **Context memory** showing current state
- **Performance metrics** (tokens, tools, time)

## 🔧 Configuration

Add to your `.env`:
```bash
AGENT_THOUGHT_CAPTURE_ENABLED=true
AGENT_VIZ_SERVER_URL=http://localhost:5000
```

## 📊 Example Output

When a user asks: "Clone the FM RDS Mapper repo and analyze it"

You'll see:
```
1. [Thinking] User: Clone the FM RDS Mapper repo and analyze it
2. [Thinking] Invoking LLM with tools
3. [Decision] Calling 1 tools: agent_git_clone
4. [Tool] agent_git_clone(repo_url=..., branch=main)
5. [Tool] Cloning https://github.com/askjake/FM_RDS_Mapper.git
6. [Result] Successfully cloned to /tmp/dish_chat_agent/...
7. [Thinking] Invoking LLM with tools
8. [Decision] Calling 1 tools: agent_run_python
9. [Tool] agent_run_python(code=..., filename=analyze.py)
10. [Result] Found 5 versions: v27, v28, v29, v30, v31
... and so on!
```

## 🐛 Troubleshooting

**Nothing appears in visualization:**
- Check that viz server is running on port 5000
- Verify `AGENT_THOUGHT_CAPTURE_ENABLED=true`
- Check browser console for errors

**Performance concerns:**
- Events are sent asynchronously (non-blocking)
- Failed sends are silently dropped
- Queue size limited to 1000 events
- Minimal impact on agent performance

## 🚀 Next Steps

1. **Basic integration** (5 min) - Follow steps above
2. **Test with simple request** - Watch it work!
3. **Add more capture points** - Instrument more tools
4. **Customize visualization** - Adjust colors, layout
5. **Record sessions** - Save thought graphs for analysis

## 📚 Full Documentation

See `INTEGRATION_GUIDE.md` for:
- Complete code examples
- Advanced features
- Architecture details
- Customization options
- Troubleshooting guide

## 💡 Tips

- Start with just the agent node instrumentation
- Add tool instrumentation gradually
- Use categories to organize thoughts: "thinking", "tool", "result", "error"
- Truncate long strings to avoid performance issues
- The interceptor is fail-safe - won't crash your app

---

**Questions?** Check the integration guide or run the demo agent to see it in action!
