#!/bin/bash
# ServiceNow MCP Server Startup Script

cd "$(dirname "$0")"

# Load SNOW credentials from the agent's .env
if [ -f "/home/jakebot/Jakes-agent/.env" ]; then
    set -a
    source /home/jakebot/Jakes-agent/.env
    set +a
fi

export SNOW_MCP_PORT="${SNOW_MCP_PORT:-8095}"
export SNOW_MCP_HOST="${SNOW_MCP_HOST:-127.0.0.1}"

echo "Starting ServiceNow MCP Server on ${SNOW_MCP_HOST}:${SNOW_MCP_PORT}..."

/home/jakebot/Jakes-agent/.venv/bin/uvicorn servicenow_mcp.server:mcp_app \
    --host "${SNOW_MCP_HOST}" \
    --port "${SNOW_MCP_PORT}" \
    --log-level info \
    >> mcp-server.log 2>&1 &

PID=$!
echo $PID > mcp-server.pid
echo "Server started with PID: $PID. Logs: tail -f $(pwd)/mcp-server.log"
