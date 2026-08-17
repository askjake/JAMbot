#!/bin/bash

#==============================================================================
# Jakebot Dish-Chat Manager
# Manages both frontend and backend services
#==============================================================================

set -e

# Configuration
BACKEND_DIR="$HOME/Jakes-agent"
FRONTEND_DIR="$HOME/Jakes-agent-fe"
FRONTEND_PORT=3002
BACKEND_PORT=8000
FRONTEND_PID_FILE="/tmp/jakebot-frontend.pid"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#==============================================================================
# Helper Functions
#==============================================================================

print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

#==============================================================================
# Status Functions
#==============================================================================

check_backend_status() {
    if curl -s http://localhost:$BACKEND_PORT/rest/api/v1/health > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

check_frontend_status() {
    if [ -f "$FRONTEND_PID_FILE" ]; then
        local pid=$(cat "$FRONTEND_PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

get_backend_pid() {
    ps aux | grep "[u]vicorn app.main" | grep -v grep | awk '{print $2}'
}

get_frontend_pid() {
    if [ -f "$FRONTEND_PID_FILE" ]; then
        cat "$FRONTEND_PID_FILE"
    fi
}

#==============================================================================
# Start Functions
#==============================================================================

start_backend() {
    print_info "Starting backend..."
    
    if check_backend_status; then
        print_warn "Backend is already running"
        return 0
    fi
    
    cd "$BACKEND_DIR"
    bash start-dishchat.sh
    
    # Wait for backend to be ready
    local max_wait=30
    local count=0
    while [ $count -lt $max_wait ]; do
        if check_backend_status; then
            print_success "Backend started successfully (PID: $(get_backend_pid))"
            return 0
        fi
        sleep 1
        count=$((count + 1))
    done
    
    print_error "Backend failed to start within ${max_wait}s"
    return 1
}

start_frontend() {
    print_info "Starting frontend..."
    
    if check_frontend_status; then
        print_warn "Frontend is already running"
        return 0
    fi
    
    cd "$FRONTEND_DIR"
    
    # Start frontend in background
    nohup pnpm -C apps/chats exec next dev --hostname 0.0.0.0 --port $FRONTEND_PORT > /tmp/jakebot-frontend.log 2>&1 &
    local pid=$!
    echo "$pid" > "$FRONTEND_PID_FILE"
    
    # Wait for frontend to be ready
    sleep 3
    
    if ps -p "$pid" > /dev/null 2>&1; then
        print_success "Frontend started successfully (PID: $pid)"
        print_info "Frontend URL: http://10.79.85.35:$FRONTEND_PORT"
        print_info "Frontend logs: tail -f /tmp/jakebot-frontend.log"
        return 0
    else
        print_error "Frontend failed to start"
        rm -f "$FRONTEND_PID_FILE"
        return 1
    fi
}

#==============================================================================
# Stop Functions
#==============================================================================

stop_backend() {
    print_info "Stopping backend..."
    
    if ! check_backend_status; then
        print_warn "Backend is not running"
        return 0
    fi
    
    cd "$BACKEND_DIR"
    bash stop-dishchat.sh
    
    print_success "Backend stopped"
}

stop_frontend() {
    print_info "Stopping frontend..."
    
    if ! check_frontend_status; then
        print_warn "Frontend is not running"
        return 0
    fi
    
    local pid=$(get_frontend_pid)
    
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        sleep 2
        
        # Force kill if still running
        if ps -p "$pid" > /dev/null 2>&1; then
            kill -9 "$pid" 2>/dev/null || true
        fi
        
        rm -f "$FRONTEND_PID_FILE"
        print_success "Frontend stopped (PID: $pid)"
    fi
}

#==============================================================================
# Status Display
#==============================================================================

show_status() {
    print_header "Jakebot Dish-Chat Status"
    
    echo ""
    echo "BACKEND:"
    if check_backend_status; then
        print_success "Running (PID: $(get_backend_pid))"
        echo "  URL: http://10.79.85.35:$BACKEND_PORT"
        echo "  Logs: $BACKEND_DIR/logs/backend.log"
    else
        print_error "Not running"
    fi
    
    echo ""
    echo "FRONTEND:"
    if check_frontend_status; then
        print_success "Running (PID: $(get_frontend_pid))"
        echo "  URL: http://10.79.85.35:$FRONTEND_PORT"
        echo "  Logs: /tmp/jakebot-frontend.log"
    else
        print_error "Not running"
    fi
    
    echo ""
}

#==============================================================================
# Main Commands
#==============================================================================

cmd_start() {
    print_header "Starting Jakebot Dish-Chat Services"
    
    start_backend
    echo ""
    start_frontend
    
    echo ""
    show_status
}

cmd_stop() {
    print_header "Stopping Jakebot Dish-Chat Services"
    
    stop_frontend
    echo ""
    stop_backend
    
    echo ""
    print_success "All services stopped"
}

cmd_restart() {
    print_header "Restarting Jakebot Dish-Chat Services"
    
    stop_frontend
    stop_backend
    
    echo ""
    sleep 2
    
    start_backend
    echo ""
    start_frontend
    
    echo ""
    show_status
}

cmd_logs() {
    local service="${1:-all}"
    
    case "$service" in
        backend)
            print_info "Showing backend logs..."
            tail -f "$BACKEND_DIR/logs/backend.log"
            ;;
        frontend)
            print_info "Showing frontend logs..."
            tail -f /tmp/jakebot-frontend.log
            ;;
        all|*)
            print_info "Showing all logs (Ctrl+C to exit)..."
            echo ""
            echo "=== BACKEND LOGS ==="
            tail -20 "$BACKEND_DIR/logs/backend.log"
            echo ""
            echo "=== FRONTEND LOGS ==="
            tail -20 /tmp/jakebot-frontend.log
            echo ""
            print_info "To follow logs in real-time:"
            echo "  Backend:  tail -f $BACKEND_DIR/logs/backend.log"
            echo "  Frontend: tail -f /tmp/jakebot-frontend.log"
            ;;
    esac
}

#==============================================================================
# Usage
#==============================================================================

show_usage() {
    cat << EOF

Usage: $(basename "$0") [COMMAND]

Commands:
  start      Start both frontend and backend services
  stop       Stop both frontend and backend services
  restart    Restart both services
  status     Show current status of services
  logs       Show recent logs (backend|frontend|all)
  
Examples:
  $(basename "$0") start
  $(basename "$0") stop
  $(basename "$0") restart
  $(basename "$0") status
  $(basename "$0") logs backend
  $(basename "$0") logs frontend

EOF
}

#==============================================================================
# Main
#==============================================================================

main() {
    local command="${1:-}"
    
    case "$command" in
        start)
            cmd_start
            ;;
        stop)
            cmd_stop
            ;;
        restart)
            cmd_restart
            ;;
        status)
            show_status
            ;;
        logs)
            cmd_logs "${2:-all}"
            ;;
        -h|--help|help)
            show_usage
            ;;
        "")
            show_status
            echo ""
            echo "Run '$(basename "$0") --help' for usage information"
            ;;
        *)
            print_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
