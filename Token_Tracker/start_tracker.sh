#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# DishChat Agent Intelligence Tracker v2.0 - Startup Script
# Monitors token usage, Opus routing, cache efficiency, and AWS token health
# ═══════════════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║       DishChat Agent Intelligence Tracker v2.0                 ║"
echo "║       Token Usage • Opus Routing • Cache • AWS Health          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "🖥️  Server: $(hostname)"
echo "📁 Directory: $(pwd)"
echo ""

# Load environment if present
if [ -f "$SCRIPT_DIR/db_config.env" ]; then
    set -a
    source "$SCRIPT_DIR/db_config.env"
    set +a
    echo "✅ Loaded db_config.env"
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi
echo "✅ $(python3 --version)"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip3 install --quiet -r requirements.txt 2>/dev/null || true
echo "✅ Dependencies ready"

# Test database connection
echo ""
echo "🔍 Testing database connection..."
python3 extract_real_data.py --test
if [ $? -ne 0 ]; then
    echo "⚠️  Database not reachable. Dashboard will show limited data."
fi

# Extract fresh data
echo ""
echo "📊 Extracting token usage data..."
python3 extract_real_data.py --output "$SCRIPT_DIR/agent_usage.log" --limit 1000
echo ""

# Find available port (start from 8503 to avoid conflicts)
STREAMLIT_PORT=${STREAMLIT_PORT:-8503}
while lsof -Pi :$STREAMLIT_PORT -sTCP:LISTEN -t >/dev/null 2>&1; do
    echo "   Port $STREAMLIT_PORT in use, trying next..."
    STREAMLIT_PORT=$((STREAMLIT_PORT + 1))
    if [ $STREAMLIT_PORT -gt 8600 ]; then
        echo "❌ No available ports in range 8503-8600"
        exit 1
    fi
done

HOST_IP=$(hostname -I | awk '{print $1}')

echo "════════════════════════════════════════════════════════════════"
echo "🚀 Launching Agent Intelligence Tracker"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Dashboard URLs:"
echo "   Local:   http://localhost:$STREAMLIT_PORT"
echo "   Network: http://$HOST_IP:$STREAMLIT_PORT"
echo ""
echo "📡 Data sources:"
echo "   DB: ${DB_HOST:-127.0.0.1}:${DB_PORT:-5433}/${DB_NAME:-dishchat}"
echo "   Agent API: ${AGENT_API_BASE:-http://127.0.0.1:8000/rest/api/v1}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Store PID for service management
echo $$ > "$SCRIPT_DIR/tracker.pid"

exec streamlit run app.py \
    --server.port $STREAMLIT_PORT \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --theme.primaryColor "#667eea" \
    --theme.backgroundColor "#ffffff" \
    --theme.secondaryBackgroundColor "#f8f9fa"
