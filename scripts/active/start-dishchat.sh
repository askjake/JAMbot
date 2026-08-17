#!/bin/bash
# Dish-Chat Startup Script for jakebot@10.79.85.35
# Updated: 2026-05-06 - Backend on port 8002

INSTALL_DIR="/home/jakebot/Jakes-agent"
LOG_DIR="${INSTALL_DIR}/logs"
VENV_DIR="${INSTALL_DIR}/.venv"
PID_FILE="${INSTALL_DIR}/backend.pid"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

cd ${INSTALL_DIR}

echo "========================================"
echo "Starting Dish-Chat Services"
echo "========================================"

# 1. Refresh AWS tokens
echo "[1/4] Refreshing AWS tokens..."
if [ -f ~/secgateway/bin/secgateway.py ]; then
    python3 ~/secgateway/bin/secgateway.py
    if [ $? -eq 0 ]; then
        log_info "✓ AWS tokens refreshed successfully"
    else
        log_error "✗ AWS token refresh failed!"
        log_warn "Continuing anyway (may affect AWS-related features)..."
    fi
else
    log_warn "SecGateway not found at ~/secgateway/bin/secgateway.py"
    log_warn "Continuing without AWS token refresh..."
fi

# 2. Check PostgreSQL
echo "[2/4] Checking PostgreSQL status..."
if docker ps | grep -q "postgres-dev-dishchat"; then
    log_info "✓ PostgreSQL container is running"
else
    log_error "✗ PostgreSQL container is not running"
    log_info "Starting PostgreSQL..."
    cd dev_postgres && docker compose up -d && cd ..
    sleep 3
fi

# 3. Check if backend is already running
echo "[3/4] Checking backend status..."
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat ${PID_FILE})
    if kill -0 $OLD_PID 2>/dev/null; then
        log_warn "Backend is already running (PID: $OLD_PID)"
        read -p "Kill and restart? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Stopping existing backend..."
            kill $OLD_PID
            sleep 2
        else
            log_info "Keeping existing backend running"
            exit 0
        fi
    else
        log_warn "Stale PID file found, removing..."
        rm -f ${PID_FILE}
    fi
fi

# 4. Start backend on port 8002
echo "[4/4] Starting backend on port 8002..."
source ${VENV_DIR}/bin/activate

mkdir -p ${LOG_DIR}

nohup ${VENV_DIR}/bin/python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8002 \
    --reload \
    >> ${LOG_DIR}/backend.log 2>&1 &

BACKEND_PID=$!
echo $BACKEND_PID > ${PID_FILE}

log_info "Backend started (PID: $BACKEND_PID)"

# -------------------------------------------------
# Start Coverity Assist Gateway (web search) on port 5001
# -------------------------------------------------
GATEWAY_PID_FILE="${INSTALL_DIR}/gateway.pid"
GATEWAY_LOG="${LOG_DIR}/gateway.log"
ORPHAN_GW=$(lsof -ti:5001 2>/dev/null)
if [ -n "$ORPHAN_GW" ]; then
    kill $ORPHAN_GW 2>/dev/null
fi
if [ -f "${INSTALL_DIR}/coverity_assist_gateway.py" ]; then
    nohup ${VENV_DIR}/bin/python ${INSTALL_DIR}/coverity_assist_gateway.py --port 5001         >> "${GATEWAY_LOG}" 2>&1 &
    GATEWAY_PID=$!
    echo $GATEWAY_PID > "${GATEWAY_PID_FILE}"
    log_info "Coverity gateway started (PID: $GATEWAY_PID) on port 5001"
else
    log_warn "coverity_assist_gateway.py not found -- web search unavailable"
fi



# Wait and verify
sleep 3

if kill -0 $BACKEND_PID 2>/dev/null; then
    log_info "✓ Backend process is running"
    
    # Test HTTP endpoint
    if curl -s http://localhost:8002/ > /dev/null 2>&1; then
        log_info "✓ Backend is responding on http://localhost:8002"
    else
        log_warn "Backend process running but not responding yet..."
        log_info "Check logs: tail -f ${LOG_DIR}/backend.log"
    fi
else
    log_error "✗ Backend failed to start"
    log_info "Check logs: tail -f ${LOG_DIR}/backend.log"
    exit 1
fi

echo ""
echo "========================================"
echo "Dish-Chat Services Started!"
echo "========================================"
echo ""
echo "Backend URL: http://10.79.85.35:8002"
echo "Backend PID: $BACKEND_PID"
echo "Gateway:     http://127.0.0.1:5001  (PID: $(cat ${INSTALL_DIR}/gateway.pid 2>/dev/null || echo N/A))"
echo "Log file:    ${LOG_DIR}/backend.log"
echo ""
echo "To view logs:"
echo "  tail -f ${LOG_DIR}/backend.log"
echo ""
echo "To stop:"
echo "  bash stop-dishchat.sh"
echo ""
echo "========================================"
