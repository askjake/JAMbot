#!/bin/bash
set -e

echo "Starting Dish-Chat with Agent Visualization..."
python3 app/agent_mode/ai_thought_viz_live.py &
VIZ_PID=$!
echo "✓ Visualization server started (PID: $VIZ_PID) on http://localhost:5001"
sleep 2
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
trap "kill $VIZ_PID 2>/dev/null" EXIT
