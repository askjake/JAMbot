#!/bin/bash
# Internal Tools Deployment Script for Jakes-agent
# Date: 2026-02-27

set -e

JAKES_AGENT_DIR=~/Jakes-agent
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "=========================================="
echo "Internal Tools Deployment"
echo "=========================================="
echo ""

# Check if directory exists
if [ ! -d "$JAKES_AGENT_DIR" ]; then
    echo "ERROR: Jakes-agent directory not found at $JAKES_AGENT_DIR"
    exit 1
fi

cd $JAKES_AGENT_DIR

echo "Step 1: Verifying file installations..."
if [ ! -f "app/tools/internal_tools.py" ]; then
    echo "ERROR: internal_tools.py not found!"
    exit 1
fi

if [ ! -f "INTERNAL_TOOLS_README.md" ]; then
    echo "ERROR: INTERNAL_TOOLS_README.md not found!"
    exit 1
fi

if [ ! -f "internal_tools.env" ]; then
    echo "ERROR: internal_tools.env not found!"
    exit 1
fi

echo "✓ All required files present"
echo ""

echo "Step 2: Backing up .env.local (if exists)..."
if [ -f ".env.local" ]; then
    cp .env.local ".env.local.backup-$TIMESTAMP"
    echo "✓ Backed up to .env.local.backup-$TIMESTAMP"
else
    echo "  No .env.local file found - will create new one"
fi
echo ""

echo "Step 3: Adding configuration to .env.local..."
if [ -f ".env.local" ]; then
    echo "" >> .env.local
    echo "# Internal Tools Configuration - Added $TIMESTAMP" >> .env.local
    cat internal_tools.env >> .env.local
    echo "✓ Configuration appended to .env.local"
else
    cp internal_tools.env .env.local
    echo "✓ Created new .env.local with configuration"
fi
echo ""

echo "Step 4: Verifying Python syntax..."
cd app/tools
if python3 -m py_compile internal_tools.py 2>/dev/null; then
    echo "✓ internal_tools.py syntax OK"
else
    echo "ERROR: Syntax error in internal_tools.py"
    exit 1
fi

cd ../agent/agents/tools
if python3 -m py_compile registry.py 2>/dev/null; then
    echo "✓ registry.py syntax OK"
else
    echo "ERROR: Syntax error in registry.py"
    exit 1
fi

cd $JAKES_AGENT_DIR
echo ""

echo "Step 5: Checking service status..."
if [ -f "backend.pid" ]; then
    PID=$(cat backend.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "  Service is currently running (PID: $PID)"
        echo "  Service needs to be restarted for changes to take effect"
        NEEDS_RESTART=true
    else
        echo "  Service is not running (stale PID file)"
        NEEDS_RESTART=false
    fi
else
    echo "  Service is not running"
    NEEDS_RESTART=false
fi
echo ""

echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Edit .env.local and update API keys/endpoints:"
echo "   vim .env.local"
echo ""
echo "   Replace these placeholders:"
echo "   - your_netra_api_key_here"
echo "   - your_google_drive_api_key_here"
echo "   - your_grasshopper_api_key_here"
echo ""
echo "2. Verify API endpoint URLs are correct for your environment"
echo ""
echo "3. Restart the service:"
if [ -f "./restart-dishchat.sh" ]; then
    echo "   ./restart-dishchat.sh"
elif [ -f "./stop-dishchat.sh" ] && [ -f "./start-dishchat.sh" ]; then
    echo "   ./stop-dishchat.sh && ./start-dishchat.sh"
else
    echo "   (Use your standard restart procedure)"
fi
echo ""
echo "4. Test the tools after restart"
echo ""
echo "5. Review documentation:"
echo "   cat INTERNAL_TOOLS_README.md"
echo ""
echo "Files installed:"
echo "  - app/tools/internal_tools.py"
echo "  - app/agent/agents/tools/registry.py (updated)"
echo "  - app/agent/agentic_rag.py (updated)"
echo "  - INTERNAL_TOOLS_README.md"
echo "  - internal_tools.env"
echo "  - .env.local (updated)"
echo ""
echo "Backups created:"
echo "  - app/agent/agents/tools/registry.py.backup-*"
echo "  - app/agent/agentic_rag.py.backup-*"
if [ "$NEEDS_RESTART" = true ]; then
    echo "  - .env.local.backup-$TIMESTAMP"
fi
echo ""
