#!/bin/bash
# Dish-Chat Stop Script for jakebot@10.79.85.35
# Created: 2026-02-27

INSTALL_DIR="/home/jakebot/Jakes-agent"
PID_FILE="${INSTALL_DIR}/backend.pid"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

echo "========================================"
echo "Stopping Dish-Chat Services"
echo "========================================"

# Stop backend
if [ -f "${PID_FILE}" ]; then
    PID=$(cat ${PID_FILE})
    if kill -0 $PID 2>/dev/null; then
        log_info "Stopping backend (PID: $PID)..."
        kill $PID
        sleep 2
        
        # Force kill if still running
        if kill -0 $PID 2>/dev/null; then
            log_warn "Backend didn't stop gracefully, force killing..."
            kill -9 $PID
        fi
        
        rm -f ${PID_FILE}
        log_info "✓ Backend stopped"
    else
        log_warn "Backend is not running (stale PID file)"
        rm -f ${PID_FILE}
    fi
else
    log_warn "Backend PID file not found"
fi

# Stop Qodo port-forward
QODO_PF_PID_FILE="${INSTALL_DIR}/qodo-portforward.pid"
if [ -f "${QODO_PF_PID_FILE}" ]; then
    QPID=$(cat "${QODO_PF_PID_FILE}")
    if kill -0 $QPID 2>/dev/null; then
        kill $QPID 2>/dev/null
        log_info "✓ Qodo port-forward stopped (PID: $QPID)"
    fi
    rm -f "${QODO_PF_PID_FILE}"
fi

# Optionally stop PostgreSQL
read -p "Stop PostgreSQL container? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd ${INSTALL_DIR}/dev_postgres
    docker compose down
    log_info "✓ PostgreSQL stopped"
fi

echo ""
echo "========================================"
echo "Dish-Chat Services Stopped"
echo "========================================"


echo
echo "[Final cleanup] Freeing backend port 8002 if still occupied..."
if ss -ltnp 2>/dev/null | grep -q ':8002 '; then
    ss -ltnp 2>/dev/null | grep ':8002 ' || true
    if command -v fuser >/dev/null 2>&1; then
        fuser -k 8002/tcp 2>/dev/null || true
    fi
fi

echo "[Final cleanup] Freeing gateway port 5001 if still occupied..."
if ss -ltnp 2>/dev/null | grep -q ':5001 '; then
    ss -ltnp 2>/dev/null | grep ':5001 ' || true
    if command -v fuser >/dev/null 2>&1; then
        fuser -k 5001/tcp 2>/dev/null || true
    fi
fi
