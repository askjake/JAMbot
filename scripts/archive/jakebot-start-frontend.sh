#!/usr/bin/env bash
set -euo pipefail

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║          JAKEBOT FRONTEND STARTUP - Runs jakebot's frontend with         ║"
echo "║          montjac's Node environment (requires sudo/cooperation)           ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "⚠️  NOTE: This will create /tmp/start-jakebot-fe-as-montjac.sh"
echo "          that montjac needs to run to start jakebot's frontend"
echo ""

# Configuration
FRONTEND_DIR="$HOME/Jakes-agent-fe"
BACKEND_DIR="$HOME/Jakes-agent"
LOG_DIR="$HOME/Jakes-agent-fe-logs"

mkdir -p "$LOG_DIR"

if [ ! -d "$FRONTEND_DIR" ]; then
  echo "❌ Frontend not found at $FRONTEND_DIR"
  exit 1
fi

echo "🔍 Finding Available Port"
FE_PORT=3001
while ss -ltn "sport = :$FE_PORT" 2>/dev/null | tail -n +2 | grep -q .; do
  ((FE_PORT++))
  [ $FE_PORT -gt 3100 ] && { echo "❌ No ports"; exit 1; }
done
echo "✓ Port: $FE_PORT"
echo ""

echo "📡 Backend Configuration"
BACKEND_PORT=$(grep "^FASTAPI_PORT=" "$BACKEND_DIR/.env" 2>/dev/null | cut -d= -f2 || echo "8000")
HOST_IP=$(hostname -I | awk '{print $1}')
echo "✓ Backend: $HOST_IP:$BACKEND_PORT"
echo ""

echo "📝 Updating .env.local"
cat > "$FRONTEND_DIR/apps/chats/.env.local" << ENVEOF
# Jakebot's Frontend (Auto-generated - points to jakebot's backend)
NEXT_PUBLIC_BACKEND_URL=http://$HOST_IP:$BACKEND_PORT
BACKEND_URL=http://127.0.0.1:$BACKEND_PORT
ENVEOF
echo "✓ Updated"
echo ""

echo "🗑️  Clearing Cache"
rm -rf "$FRONTEND_DIR/apps/chats/.next" 2>/dev/null || true
echo "✓ Done"
echo ""

echo "📜 Creating montjac startup script"
cat > /tmp/start-jakebot-fe-as-montjac.sh << 'MONTJACSCRIPT'
#!/usr/bin/env bash
set -euo pipefail

# This script runs as montjac to start jakebot's frontend
# It uses montjac's node/pnpm environment

echo "Starting jakebot's frontend as montjac..."

FRONTEND_DIR="/home/jakebot/Jakes-agent-fe"
LOG_DIR="/home/jakebot/Jakes-agent-fe-logs"
FE_PORT="REPLACE_PORT"
BACKEND_PORT="REPLACE_BACKEND_PORT"
HOST_IP="REPLACE_HOST_IP"

# Stop existing
pkill -f "next dev.*Jakes-agent-fe" || true
sleep 2

# Setup fnm environment
export FNM_DIR="$HOME/.local/share/fnm"
if [ -f "$FNM_DIR/fnm" ]; then
  export PATH="$FNM_DIR:$PATH"
  eval "$($FNM_DIR/fnm env --shell bash)"
fi

# Remove Volta
export PATH=$(echo $PATH | tr ':' '\n' | grep -v volta | tr '\n' ':' | sed 's/:$//')

cd "$FRONTEND_DIR"

export NEXT_PUBLIC_BACKEND_URL="http://$HOST_IP:$BACKEND_PORT"
export BACKEND_URL="http://127.0.0.1:$BACKEND_PORT"
export PORT=$FE_PORT
export NEXT_PUBLIC_DISABLE_MERMAID=true

nohup pnpm -C apps/chats exec next dev --hostname 0.0.0.0 --port $FE_PORT --turbo > "$LOG_DIR/frontend.log" 2>&1 &

FE_PID=$!
echo $FE_PID > "$LOG_DIR/frontend.pid"
echo "✓ Started (pid $FE_PID) on port $FE_PORT"
echo "✓ Log: $LOG_DIR/frontend.log"
MONTJACSCRIPT

# Replace placeholders
sed -i "s/REPLACE_PORT/$FE_PORT/g" /tmp/start-jakebot-fe-as-montjac.sh
sed -i "s/REPLACE_BACKEND_PORT/$BACKEND_PORT/g" /tmp/start-jakebot-fe-as-montjac.sh
sed -i "s/REPLACE_HOST_IP/$HOST_IP/g" /tmp/start-jakebot-fe-as-montjac.sh
chmod +x /tmp/start-jakebot-fe-as-montjac.sh

echo "✓ Created: /tmp/start-jakebot-fe-as-montjac.sh"
echo ""

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                    READY TO START                                         ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 Next Step: Have montjac run the startup script:"
echo ""
echo "   ssh montjac@10.79.85.35 'bash /tmp/start-jakebot-fe-as-montjac.sh'"
echo ""
echo "📊 This will start jakebot's frontend on:"
echo "   Port:     $FE_PORT"
echo "   Backend:  http://$HOST_IP:$BACKEND_PORT"
echo "   Frontend: http://$HOST_IP:$FE_PORT"
echo ""
