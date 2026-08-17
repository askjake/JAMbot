#!/usr/bin/env bash
set -e

DISH_CHAT_DIR="$HOME/dish-chat"
BACKUP_DIR="$DISH_CHAT_DIR/backups/token_compression_fix_$(date +%Y%m%d_%H%M%S)"

RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
NC="\033[0m"

print_msg() {
    echo -e "${1}${2}${NC}"
}

print_header() {
    echo ""
    echo "========================================="
    echo "$1"
    echo "========================================="
    echo ""
}

create_backups() {
    print_msg "$BLUE" "Creating backup at $BACKUP_DIR..."
    mkdir -p "$BACKUP_DIR"
    
    print_msg "$BLUE" "Backing up existing files..."
    cp "$DISH_CHAT_DIR/app/core/llm/chat_models.py" "$BACKUP_DIR/chat_models.py.backup" 2>/dev/null || true
    cp "$DISH_CHAT_DIR/app/agent/service.py" "$BACKUP_DIR/agent_service.py.backup" 2>/dev/null || true
    cp "$DISH_CHAT_DIR/app/message/service.py" "$BACKUP_DIR/message_service.py.backup" 2>/dev/null || true
    
    print_msg "$GREEN" "✓ Backups created"
}

deploy_new_files() {
    print_msg "$BLUE" "Deploying new files..."
    
    if [ -f "$DISH_CHAT_DIR/app/core/llm/chat_models.py.new" ]; then
        print_msg "$BLUE" "  - Deploying enhanced chat_models.py..."
        mv "$DISH_CHAT_DIR/app/core/llm/chat_models.py.new" "$DISH_CHAT_DIR/app/core/llm/chat_models.py"
        print_msg "$GREEN" "    ✓ chat_models.py deployed"
    else
        print_msg "$YELLOW" "  ⚠ chat_models.py.new not found, skipping"
    fi
    
    if [ -f "$DISH_CHAT_DIR/app/message/compression.py" ]; then
        print_msg "$GREEN" "  - compression.py already deployed"
    else
        print_msg "$YELLOW" "  ⚠ compression.py not found"
    fi
    
    if [ -f "$DISH_CHAT_DIR/app/agent/service.py.new" ]; then
        print_msg "$BLUE" "  - Deploying enhanced agent/service.py..."
        mv "$DISH_CHAT_DIR/app/agent/service.py.new" "$DISH_CHAT_DIR/app/agent/service.py"
        print_msg "$GREEN" "    ✓ agent/service.py deployed"
    else
        print_msg "$YELLOW" "  ⚠ agent/service.py.new not found, skipping"
    fi
    
    if [ -f "$DISH_CHAT_DIR/app/message/error_handling.py" ]; then
        print_msg "$GREEN" "  - error_handling.py already deployed"
    else
        print_msg "$YELLOW" "  ⚠ error_handling.py not found"
    fi
    
    print_msg "$GREEN" "✓ All files deployed"
}

install_dependencies() {
    print_msg "$BLUE" "Checking dependencies..."
    
    cd "$DISH_CHAT_DIR"
    
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        print_msg "$YELLOW" "⚠ Virtual environment not found"
    fi
    
    if ! python -c "import tiktoken" 2>/dev/null; then
        print_msg "$BLUE" "Installing tiktoken..."
        pip install tiktoken >/dev/null 2>&1
        print_msg "$GREEN" "✓ tiktoken installed"
    else
        print_msg "$GREEN" "✓ tiktoken already installed"
    fi
}

validate_syntax() {
    print_msg "$BLUE" "Running syntax validation..."
    
    cd "$DISH_CHAT_DIR"
    source .venv/bin/activate 2>/dev/null || true
    
    local files=(
        "app/core/llm/chat_models.py"
        "app/message/compression.py"
        "app/agent/service.py"
        "app/message/error_handling.py"
    )
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            if python -m py_compile "$file" 2>/dev/null; then
                print_msg "$GREEN" "  ✓ $(basename $file) syntax OK"
            else
                print_msg "$RED" "  ✗ $(basename $file) has syntax errors"
            fi
        fi
    done
}

main() {
    print_header "Dish-Chat Token & Compression Fix Deploy"
    
    create_backups
    echo ""
    deploy_new_files
    echo ""
    install_dependencies
    echo ""
    validate_syntax
    
    echo ""
    print_header "Deployment complete!"
    
    print_msg "$BLUE" "Next steps:"
    print_msg "$BLUE" "1. Review backups: $BACKUP_DIR"
    print_msg "$BLUE" "2. Restart application"
    print_msg "$BLUE" "3. Monitor logs: tail -f ~/dish-chat/backend.log"
    print_msg "$BLUE" "4. Run validation: ./validate_fixes.sh"
    echo ""
    print_msg "$YELLOW" "To rollback:"
    print_msg "$YELLOW" "  cp $BACKUP_DIR/*.backup app/..."
    echo ""
}

main
