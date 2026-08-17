#!/usr/bin/env bash
set -euo pipefail

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║           JAKEBOT FRONTEND SETUP - Dynamic Port Configuration            ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
FRONTEND_DIR="$HOME/Jakes-agent-fe"
BACKEND_DIR="$HOME/Jakes-agent"
GIT_REPO="${GIT_REPO:-git@gitlab.com:dish-cloud/dt/sse/datasolutions/dish-chat-fe.git}"

echo "📦 Step 1: Clone Frontend Repository"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d "$FRONTEND_DIR" ]; then
  echo "⚠️  Frontend directory already exists: $FRONTEND_DIR"
  read -p "Do you want to remove it and re-clone? (y/N): " confirm
  if [[ "$confirm" =~ ^[Yy]$ ]]; then
    rm -rf "$FRONTEND_DIR"
    echo "✓ Removed existing directory"
  else
    echo "ℹ️  Using existing directory"
  fi
fi

if [ ! -d "$FRONTEND_DIR" ]; then
  echo "🔄 Cloning frontend repository..."
  git clone "$GIT_REPO" "$FRONTEND_DIR"
  if [ $? -eq 0 ]; then
    echo "✓ Frontend cloned successfully"
  else
    echo "❌ Failed to clone frontend"
    exit 1
  fi
else
  echo "✓ Frontend directory exists"
fi
echo ""

echo "📝 Step 2: Get Backend Port Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Get backend port from jakebot's .env
if [ -f "$BACKEND_DIR/.env" ]; then
  BACKEND_PORT=$(grep "^FASTAPI_PORT=" "$BACKEND_DIR/.env" | cut -d= -f2)
  if [ -z "$BACKEND_PORT" ]; then
    BACKEND_PORT=8000
  fi
  echo "✓ Backend port from .env: $BACKEND_PORT"
else
  BACKEND_PORT=8000
  echo "⚠️  No .env found, using default: $BACKEND_PORT"
fi

# Get host IP
HOST_IP=$(hostname -I | awk '{print $1}')
echo "✓ Host IP: $HOST_IP"
echo ""

echo "🔧 Step 3: Install Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$FRONTEND_DIR"

# Check for pnpm
PNPM_CMD=""
if command -v pnpm >/dev/null 2>&1; then
  PNPM_CMD="pnpm"
  echo "✓ pnpm found in PATH"
else
  echo "⚠️  pnpm not in PATH, will use fnm"
  
  # Setup fnm
  export FNM_DIR="$HOME/.local/share/fnm"
  if [ -f "$FNM_DIR/fnm" ]; then
    export PATH="$FNM_DIR:$PATH"
    eval "$($FNM_DIR/fnm env --shell bash)"
    
    # Remove Volta from PATH if present
    export PATH=$(echo $PATH | tr ':' '\n' | grep -v volta | tr '\n' ':' | sed 's/:$//')
    
    if command -v pnpm >/dev/null 2>&1; then
      PNPM_CMD="pnpm"
      echo "✓ pnpm found via fnm"
    fi
  fi
fi

if [ -z "$PNPM_CMD" ]; then
  echo "❌ Error: pnpm not found"
  echo "   Please install pnpm: npm install -g pnpm"
  echo "   Or ensure fnm is properly set up"
  exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
  echo "📦 Installing frontend dependencies..."
  $PNPM_CMD install
  echo "✓ Dependencies installed"
else
  echo "✓ Dependencies already installed"
fi
echo ""

echo "📝 Step 4: Configure Frontend Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create .env.local for the frontend
cat > "$FRONTEND_DIR/apps/chats/.env.local" << ENVEOF
# Jakebot's Frontend Configuration
# Browser-side API base (MUST be reachable from your laptop/browser)
NEXT_PUBLIC_BACKEND_URL=http://$HOST_IP:$BACKEND_PORT

# Server-side API base (used by Next.js server)
BACKEND_URL=http://127.0.0.1:$BACKEND_PORT
ENVEOF

echo "✓ Created apps/chats/.env.local:"
cat "$FRONTEND_DIR/apps/chats/.env.local"
echo ""

echo "✅ Setup Complete!"
echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                         SETUP SUMMARY                                     ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📂 Frontend Directory: $FRONTEND_DIR"
echo "🔗 Backend URL:        http://$HOST_IP:$BACKEND_PORT"
echo "🌐 Host IP:            $HOST_IP"
echo ""
echo "🚀 Next Steps:"
echo "   1. Start the frontend with dynamic ports:"
echo "      ~/jakebot-start-frontend.sh"
echo ""
echo "   2. Or manually:"
echo "      cd $FRONTEND_DIR"
echo "      pnpm -C apps/chats exec next dev --hostname 0.0.0.0 --port 3001 --turbo"
echo ""
