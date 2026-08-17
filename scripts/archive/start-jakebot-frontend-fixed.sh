#!/bin/bash
# Load fnm environment
export FNM_DIR="$HOME/.local/share/fnm"
if [ -d "$FNM_DIR" ]; then
  export PATH="$FNM_DIR:$PATH"
  eval "$(fnm env --shell bash)"
fi

# Change to frontend directory
cd $HOME/Jakes-agent-fe

# Clear any corrupted cache
rm -rf apps/chats/.next 2>/dev/null

# Set environment variables
export NEXT_PUBLIC_BACKEND_URL=http://10.79.85.35:8000
export BACKEND_URL=http://127.0.0.1:8000
export PORT=3002
export NEXT_PUBLIC_DISABLE_MERMAID=true

# Start frontend WITHOUT --turbo (this was causing crashes)
nohup pnpm -C apps/chats exec next dev --hostname 0.0.0.0 --port 3002 > $HOME/Jakes-agent-fe-logs/frontend.log 2>&1 &

# Save PID
echo $! > $HOME/Jakes-agent-fe-logs/frontend.pid

echo "Started frontend PID: $!"
echo "Frontend will be available at http://10.79.85.35:3002"
echo "Check logs: tail -f ~/Jakes-agent-fe-logs/frontend.log"

