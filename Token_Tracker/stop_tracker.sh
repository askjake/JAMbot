#!/bin/bash
# Stop the Token Tracker
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/tracker.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping Token Tracker (PID: $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        echo "✅ Stopped"
    else
        echo "Process $PID not running. Cleaning up PID file."
        rm -f "$PID_FILE"
    fi
else
    echo "No PID file found. Checking for streamlit processes..."
    pkill -f "streamlit run app.py" 2>/dev/null && echo "✅ Killed streamlit" || echo "No streamlit process found"
fi
