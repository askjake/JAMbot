#!/bin/bash
# Sentry Integration Setup Script for Dish-Chat
# Run from your dish-chat repository root: bash setup_sentry.sh

set -e  # Exit on error

echo "=========================================="
echo "Sentry Integration Setup for Dish-Chat"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "app/config.py" ]; then
    echo "❌ Error: app/config.py not found!"
    echo "Please run this script from the dish-chat repository root."
    exit 1
fi

# Step 1: Backup
echo "📦 Step 1: Creating backup..."
BACKUP_FILE="app/config.py.backup-$(date +%Y%m%d-%H%M%S)"
cp app/config.py "$BACKUP_FILE"
echo "✓ Backup created: $BACKUP_FILE"
echo ""

# Step 2: Add Sentry configuration
echo "🔧 Step 2: Adding Sentry configuration to config.py..."

# Check if Sentry config already exists
if grep -q "SENTRY_AUTH_TOKEN" app/config.py; then
    echo "⚠️  Sentry configuration already exists in config.py"
    echo "Skipping this step..."
else
    # Find the line with COVERITY_GATEWAY_URL and add after it
    sed -i '/COVERITY_GATEWAY_URL.*=.*"http/a\
\
    # Sentry Integration (for cluster inspection and monitoring)\
    SENTRY_AUTH_TOKEN: Optional[str] = None\
    SENTRY_ORG: str = "dishtv.technology"\
    SENTRY_URL: str = "https://ds-testing-sentry"\
    SENTRY_PROJECT: Optional[str] = None  # Set if needed for specific project' app/config.py
    
    echo "✓ Sentry configuration added"
fi
echo ""

# Step 3: Enable internal tools
echo "🔧 Step 3: Enabling INTERNAL_TOOLS_MCP..."
sed -i 's/ENABLE_INTERNAL_TOOLS_MCP: bool = False  # until you have that server/ENABLE_INTERNAL_TOOLS_MCP: bool = True   # Sentry cluster access enabled/' app/config.py
echo "✓ Internal tools MCP enabled"
echo ""

# Step 4: Update .env file
echo "🔧 Step 4: Updating .env file..."
if [ -f ".env" ]; then
    if grep -q "SENTRY_AUTH_TOKEN" .env; then
        echo "⚠️  Sentry variables already exist in .env"
        echo "Skipping this step..."
    else
        cat >> .env << 'EOF'

# Sentry Configuration
SENTRY_AUTH_TOKEN="${SENTRY_AUTH_TOKEN:?Set SENTRY_AUTH_TOKEN from the approved secret store}"
SENTRY_ORG=dishtv.technology
SENTRY_URL=https://ds-testing-sentry
EOF
        echo "✓ Sentry variables added to .env"
    fi
else
    echo "⚠️  .env file not found, creating it..."
    cat > .env << 'EOF'
# Sentry Configuration
SENTRY_AUTH_TOKEN="${SENTRY_AUTH_TOKEN:?Set SENTRY_AUTH_TOKEN from the approved secret store}"
SENTRY_ORG=dishtv.technology
SENTRY_URL=https://ds-testing-sentry
EOF
    echo "✓ .env file created with Sentry variables"
fi
echo ""

# Step 5: Verify changes
echo "✅ Step 5: Verifying changes..."
echo ""
echo "Sentry configuration in config.py:"
grep -A 4 "Sentry Integration" app/config.py || echo "⚠️  Could not find Sentry config"
echo ""
echo "Internal tools MCP status:"
grep "ENABLE_INTERNAL_TOOLS_MCP" app/config.py || echo "⚠️  Could not find ENABLE_INTERNAL_TOOLS_MCP"
echo ""
echo "Environment variables in .env:"
grep "SENTRY_" .env || echo "⚠️  Could not find Sentry variables"
echo ""

# Step 6: Show diff
echo "📊 Changes made to config.py:"
diff "$BACKUP_FILE" app/config.py || true
echo ""

echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the changes above"
echo "2. Test the application: python app/main.py"
echo "3. Or use your startup script: ~/start-dishchat-full.sh"
echo ""
echo "To rollback: cp $BACKUP_FILE app/config.py"
echo ""

