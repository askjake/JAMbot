#!/bin/bash
# Dish-Chat Restart Script for jakebot@10.79.85.35
# Updated: 2026-05-28 - Auto-recovery for PostgreSQL corruption

INSTALL_DIR="/home/jakebot/Jakes-agent"
LOG_DIR="${INSTALL_DIR}/logs"
VENV_DIR="${INSTALL_DIR}/.venv"
PID_FILE="${INSTALL_DIR}/backend.pid"
PG_CONTAINER="dishchat-postgres"
PG_PORT=5434
PG_USER="dev_user"
PG_DB="dishchat"
PG_VOLUME="dishchat-postgres_pgdata_dishchat"
PG_IMAGE="pgvector/pgvector:pg17"
MAX_PG_WAIT=30
MAX_BACKEND_WAIT=20

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

cd ${INSTALL_DIR}

echo "========================================"
echo "Restarting Dish-Chat Services"
echo "========================================"

# ─────────────────────────────────────────────
# PHASE 1: Stop Backend
# ─────────────────────────────────────────────
echo ""
echo "[Phase 1] Stopping backend..."
if [ -f "${PID_FILE}" ]; then
    PID=$(cat ${PID_FILE})
    if kill -0 $PID 2>/dev/null; then
        kill $PID 2>/dev/null
        sleep 2
        if kill -0 $PID 2>/dev/null; then
            kill -9 $PID 2>/dev/null
        fi
        log_info "Backend stopped (PID: $PID)"
    else
        log_warn "Backend not running (stale PID)"
    fi
    rm -f ${PID_FILE}
else
    # Kill any orphaned uvicorn on port 8002
    ORPHAN_PID=$(lsof -ti:8002 2>/dev/null)
    if [ -n "$ORPHAN_PID" ]; then
        kill $ORPHAN_PID 2>/dev/null
        sleep 1
        log_warn "Killed orphaned process on port 8002 (PID: $ORPHAN_PID)"
    fi
fi

# ─────────────────────────────────────────────
# PHASE 2: Ensure PostgreSQL is Healthy
# ─────────────────────────────────────────────
echo ""
echo "[Phase 2] Ensuring PostgreSQL is healthy..."

pg_is_healthy() {
    docker exec ${PG_CONTAINER} pg_isready -U ${PG_USER} -d ${PG_DB} >/dev/null 2>&1
}

pg_is_running() {
    docker ps --filter "name=^${PG_CONTAINER}$" --filter "status=running" --format "{{.Names}}" | grep -q "${PG_CONTAINER}"
}

pg_is_crash_looping() {
    local status=$(docker inspect ${PG_CONTAINER} --format="{{.State.Status}}" 2>/dev/null)
    local restarting=$(docker inspect ${PG_CONTAINER} --format="{{.State.Restarting}}" 2>/dev/null)
    if [ "$restarting" = "true" ] || [ "$status" = "restarting" ]; then
        return 0
    fi
    # Check if recent logs show WAL/checkpoint corruption
    if docker logs --tail 5 ${PG_CONTAINER} 2>&1 | grep -q "PANIC.*checkpoint"; then
        return 0
    fi
    return 1
}

attempt_wal_recovery() {
    log_warn "=== WAL RECOVERY ==="
    log_warn "Detected corrupted WAL checkpoint. Running pg_resetwal..."

    # Stop the container first
    docker stop ${PG_CONTAINER} 2>/dev/null
    sleep 2

    # Run pg_resetwal in a temporary container
    local output
    output=$(docker run --rm \
        -v ${PG_VOLUME}:/var/lib/postgresql/data \
        ${PG_IMAGE} \
        bash -c "chown -R postgres:postgres /var/lib/postgresql/data && su postgres -c 'pg_resetwal -f /var/lib/postgresql/data'" 2>&1)

    if [ $? -eq 0 ]; then
        log_info "WAL reset successful: $output"
        return 0
    else
        log_error "WAL reset failed: $output"
        return 1
    fi
}

recreate_postgres_fresh() {
    log_warn "Creating fresh PostgreSQL database (data will be lost)..."

    docker stop ${PG_CONTAINER} 2>/dev/null
    docker rm ${PG_CONTAINER} 2>/dev/null

    # Remove corrupted volume
    docker volume rm ${PG_VOLUME} 2>/dev/null

    # Recreate via compose
    cd ${INSTALL_DIR}/dev_postgres
    docker compose up -d
    cd ${INSTALL_DIR}

    # Wait for it
    local waited=0
    while [ $waited -lt ${MAX_PG_WAIT} ]; do
        if pg_is_healthy; then
            log_info "Fresh PostgreSQL is ready"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done

    log_error "Fresh PostgreSQL failed to start"
    return 1
}

# Main PostgreSQL startup logic
if pg_is_healthy; then
    log_info "PostgreSQL is already healthy on port ${PG_PORT}"
elif pg_is_running; then
    log_warn "PostgreSQL container running but not healthy, waiting..."
    waited=0
    while [ $waited -lt ${MAX_PG_WAIT} ]; do
        if pg_is_healthy; then
            log_info "PostgreSQL became healthy"
            break
        fi
        sleep 2
        waited=$((waited + 2))
    done
    if ! pg_is_healthy; then
        log_error "PostgreSQL not responding after ${MAX_PG_WAIT}s"
        if pg_is_crash_looping; then
            log_error "Detected crash loop (likely WAL corruption)"
            if attempt_wal_recovery; then
                docker start ${PG_CONTAINER}
                sleep 3
            else
                recreate_postgres_fresh
            fi
        fi
    fi
elif docker ps -a --format "{{.Names}}" | grep -q "^${PG_CONTAINER}$"; then
    # Container exists but is stopped
    log_info "Starting existing PostgreSQL container..."

    # Check for corruption before starting
    if docker logs --tail 10 ${PG_CONTAINER} 2>&1 | grep -q "PANIC.*checkpoint"; then
        log_warn "Previous WAL corruption detected in logs"
        if attempt_wal_recovery; then
            docker start ${PG_CONTAINER}
            sleep 3
        else
            recreate_postgres_fresh
        fi
    else
        docker start ${PG_CONTAINER}
        sleep 3
    fi
else
    # Container doesn't exist at all, create via compose
    log_info "Creating PostgreSQL container via docker-compose..."
    cd ${INSTALL_DIR}/dev_postgres
    docker compose up -d
    cd ${INSTALL_DIR}
    sleep 3
fi

# Final health verification with crash loop detection
waited=0
while [ $waited -lt ${MAX_PG_WAIT} ]; do
    if pg_is_healthy; then
        break
    fi
    # Check for crash loop during wait
    if pg_is_crash_looping; then
        log_error "PostgreSQL entered crash loop during startup"
        if attempt_wal_recovery; then
            docker start ${PG_CONTAINER} 2>/dev/null
            sleep 3
        else
            log_error "Recovery failed. Recreating database from scratch..."
            recreate_postgres_fresh
            break
        fi
    fi
    sleep 2
    waited=$((waited + 2))
done

if pg_is_healthy; then
    log_info "✓ PostgreSQL verified healthy on port ${PG_PORT}"
else
    log_error "✗ PostgreSQL could not be started. Manual intervention required."
    echo "  Troubleshooting:"
    echo "    docker logs ${PG_CONTAINER}"
    echo "    docker inspect ${PG_CONTAINER} --format='{{.State.Status}}'"
    exit 1
fi

# ─────────────────────────────────────────────
# PHASE 3: Refresh AWS Tokens
# ─────────────────────────────────────────────
echo ""
echo "[Phase 3] Refreshing AWS tokens..."
if [ -f ~/secgateway/bin/secgateway.py ]; then
    python3 ~/secgateway/bin/secgateway.py 2>/dev/null
    if [ $? -eq 0 ]; then
        log_info "✓ AWS tokens refreshed"
    else
        log_warn "AWS token refresh failed (non-critical, continuing)"
    fi
else
    log_warn "SecGateway not found, skipping"
fi

# ─────────────────────────────────────────────
# PHASE 4: Start Backend
# ─────────────────────────────────────────────
echo ""

echo
echo "[Phase 3.5] Forcing backend port 8002 free..."
if ss -ltnp 2>/dev/null | grep -q ':8002 '; then
    echo "[WARN] Port 8002 still occupied:"
    ss -ltnp 2>/dev/null | grep ':8002 ' || true

    # Kill only the process bound to the backend port.
    if command -v fuser >/dev/null 2>&1; then
        fuser -k 8002/tcp 2>/dev/null || true
    else
        PIDS="$(ss -ltnp 2>/dev/null | awk '/:8002 / {print $NF}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)"
        for pid in $PIDS; do
            echo "[WARN] Killing backend port pid=$pid"
            kill "$pid" 2>/dev/null || true
            sleep 1
            kill -9 "$pid" 2>/dev/null || true
        done
    fi

    sleep 2
fi

if ss -ltnp 2>/dev/null | grep -q ':8002 '; then
    echo "[ERROR] Port 8002 is still occupied after cleanup"
    ss -ltnp 2>/dev/null | grep ':8002 ' || true
    exit 1
fi
echo "[INFO] ✓ Port 8002 is free"

echo "[Phase 4] Starting backend on port 8002..."
mkdir -p ${LOG_DIR}

source ${VENV_DIR}/bin/activate

nohup ${VENV_DIR}/bin/python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8002 \
    --reload \
    >> ${LOG_DIR}/backend.log 2>&1 &

BACKEND_PID=$!
echo $BACKEND_PID > ${PID_FILE}
log_info "Backend started (PID: $BACKEND_PID)"

# -------------------------------------------------
# PHASE 4b: Start Coverity Assist Gateway (web search)
# -------------------------------------------------
GATEWAY_PID_FILE="${INSTALL_DIR}/gateway.pid"
GATEWAY_LOG="${LOG_DIR}/gateway.log"

# Kill any existing gateway on port 5001
ORPHAN_GW=$(lsof -ti:5001 2>/dev/null)
if [ -n "$ORPHAN_GW" ]; then
    kill $ORPHAN_GW 2>/dev/null
    log_info "Killed existing process on port 5001"
fi
if [ -f "${GATEWAY_PID_FILE}" ]; then
    OLD_GW_PID=$(cat "${GATEWAY_PID_FILE}")
    kill $OLD_GW_PID 2>/dev/null
    rm -f "${GATEWAY_PID_FILE}"
fi

if [ -f "${INSTALL_DIR}/coverity_assist_gateway.py" ]; then
    nohup ${VENV_DIR}/bin/python ${INSTALL_DIR}/coverity_assist_gateway.py --port 5001         >> "${GATEWAY_LOG}" 2>&1 &
    GATEWAY_PID=$!
    echo $GATEWAY_PID > "${GATEWAY_PID_FILE}"
    log_info "Coverity gateway started (PID: $GATEWAY_PID) on port 5001"
else
    log_warn "coverity_assist_gateway.py not found -- web search unavailable"
fi




# ─────────────────────────────────────────────
# PHASE 4c: Start Qodo SSH Proxy Port-Forward
# ─────────────────────────────────────────────
echo ""
echo "[Phase 4c] Starting Qodo SSH proxy port-forward (localhost:18443 → cluster)..."
QODO_PF_PID_FILE="${INSTALL_DIR}/qodo-portforward.pid"

# Kill any previous port-forward on 18443
OLD_QODO_PF=$(lsof -ti:18443 2>/dev/null)
if [ -n "$OLD_QODO_PF" ]; then
    kill $OLD_QODO_PF 2>/dev/null
    sleep 1
fi
[ -f "${QODO_PF_PID_FILE}" ] && rm -f "${QODO_PF_PID_FILE}"

QODO_KUBE_CTX="arn:aws:eks:us-west-2:233532778289:cluster/apps-xx-eks-gltqp-2fq9x"
if kubectl --context \"\\${QODO_KUBE_CTX}\" get svc qodo-ssh-proxy -n open-webui-dev >/dev/null 2>&1; then
    nohup kubectl --context \"\\${QODO_KUBE_CTX}\" port-forward -n open-webui-dev svc/qodo-ssh-proxy 18443:8443 \
        >> "${LOG_DIR}/qodo-portforward.log" 2>&1 &
    QODO_PF_PID=$!
    echo $QODO_PF_PID > "${QODO_PF_PID_FILE}"
    sleep 2
    if kill -0 $QODO_PF_PID 2>/dev/null; then
        log_info "Qodo port-forward started (PID: $QODO_PF_PID) on localhost:18443"
    else
        log_warn "Qodo port-forward failed to start -- qodo_context_mcp will be disabled"
    fi
else
    log_warn "qodo-ssh-proxy service not found in cluster -- qodo_context_mcp will be disabled"
fi


# ─────────────────────────────────────────────
# PHASE 5: Verify Backend Health
# ─────────────────────────────────────────────
echo ""

echo "[Phase 4d] Starting dish-code-tools MCP server (localhost:8087)..."
DCT_DIR="/home/montjac/dish-code-tools"
DCT_PID_FILE="${DCT_DIR}/dish-code-tools.pid"
DCT_LOG="${LOG_DIR}/dish-code-tools.log"

# Stop any existing instance
OLD_DCT=$(lsof -ti:8087 2>/dev/null)
[ -n "$OLD_DCT" ] && kill $OLD_DCT 2>/dev/null && sleep 1
[ -f "${DCT_PID_FILE}" ] && rm -f "${DCT_PID_FILE}"

if [ -f "${DCT_DIR}/start-dish-code-tools.sh" ] && [ -d "${DCT_DIR}/.venv" ]; then
    cd "${DCT_DIR}"
    nohup "${DCT_DIR}/.venv/bin/python" -m app         >> "${DCT_LOG}" 2>&1 &
    DCT_PID=$!
    echo $DCT_PID > "${DCT_PID_FILE}"
    sleep 3
    if kill -0 $DCT_PID 2>/dev/null; then
        log_info "dish-code-tools started (PID: $DCT_PID) on localhost:8087"
    else
        log_warn "dish-code-tools failed to start -- dish_code_tools MCP will be disabled"
    fi
    cd - > /dev/null
else
    log_warn "dish-code-tools not installed at ${DCT_DIR} -- dish_code_tools MCP will be disabled"
fi



echo "[Phase 4e] Starting DVA gateway + DVA MCP server on dsgpu3080 (10.79.85.47)..."
# DVA REST gateway:  10.79.85.47:5006  (local STB access, scans, jam)
# DVA MCP server:    10.79.85.47:5007/mcp  (FastMCP agent tool registration)
# Both must be running on the 3080 for dva_mcp tools to load in the agent.

# Step 1: start the local dva-gateway on 3090 (for direct STB access)
if [ -f "/home/montjac/dva-gateway/start-dva-gateway.sh" ]; then
    bash /home/montjac/dva-gateway/start-dva-gateway.sh >> "/home/montjac/dva-gateway/dva-gateway.log" 2>&1
    sleep 2
    if curl -s http://127.0.0.1:5006/health | python3.10 -c 'import sys,json; d=json.load(sys.stdin); print("[INFO] DVA gateway (3090) v"+d.get("version","?")+" on :5006")' 2>/dev/null; then
        :
    else
        echo "[WARN] DVA gateway health check failed"
    fi
else
    echo "[WARN] ~/dva-gateway/start-dva-gateway.sh not found -- DVA gateway skipped"
fi

# Step 2: start the DVA MCP server on the 3080 (agent connects to this)
if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 montjac@10.79.85.47 \
    "bash ~/dva-gateway/start-dva-mcp.sh" 2>/dev/null; then
    sleep 3
    if curl -s -o /dev/null -w '%{http_code}' http://10.79.85.47:5007/mcp | grep -qE '^(200|406)$'; then
        echo "[INFO] DVA MCP server (3080) v1.0.0 on 10.79.85.47:5007/mcp"
    else
        echo "[WARN] DVA MCP server health check failed (may still be starting)"
    fi
else
    echo "[WARN] Could not start DVA MCP server on 3080 -- dva_mcp tools will be disabled"
fi

