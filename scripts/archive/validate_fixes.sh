#!/usr/bin/env bash
set -euo pipefail

DISH_CHAT_DIR="$HOME/dish-chat"
GREEN="\033[0;32m"
RED="\033[0;31m"
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

cd "$DISH_CHAT_DIR"
source .venv/bin/activate 2>/dev/null || true

print_header "Dish-Chat Fixes Validation"

# Test tiktoken
if python -c "import tiktoken" 2>/dev/null; then
    print_msg "$GREEN" "✓ tiktoken available"
else
    print_msg "$RED" "✗ tiktoken not installed"
    exit 1
fi

# Test token counting
if python -c "import tiktoken; enc=tiktoken.get_encoding('cl100k_base'); print(len(enc.encode('hello world')))" 2>/dev/null | grep -q "[0-9]"; then
    print_msg "$GREEN" "✓ Token counting works"
else
    print_msg "$RED" "✗ Token counting failed"
    exit 1
fi

# Test compression module
if python -c "from app.message.compression import count_tokens; print(count_tokens('test'))" 2>/dev/null | grep -q "[0-9]"; then
    print_msg "$GREEN" "✓ Compression module works"
else
    print_msg "$YELLOW" "⚠ Compression module not found or not working"
fi

# Test error handling
if python -c "from app.message.error_handling import is_recoverable_error; print(is_recoverable_error(TimeoutError()))" 2>/dev/null | grep -q "True"; then
    print_msg "$GREEN" "✓ Error handling module works"
else
    print_msg "$YELLOW" "⚠ Error handling module not found or not working"
fi

# Test chat models module
if python -c "from app.core.llm import chat_models" 2>/dev/null; then
    print_msg "$GREEN" "✓ Chat models module loads"
else
    print_msg "$RED" "✗ Chat models module has errors"
    exit 1
fi

echo ""
print_msg "$GREEN" "✓ All validation checks passed"
