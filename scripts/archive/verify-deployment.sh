#!/bin/bash
# Dish-Chat Deployment Verification Script
# Run this after deployment to verify everything is working
# Created: 2026-02-27

INSTALL_DIR="/home/jakebot/Jakes-agent"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }
log_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

PASS_COUNT=0
FAIL_COUNT=0

check_pass() {
    log_pass "$1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

check_fail() {
    log_fail "$1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

echo "========================================"
echo "Dish-Chat Deployment Verification"
echo "========================================"
echo ""

cd ${INSTALL_DIR}

# 1. Check directory structure
echo "[1/10] Checking directory structure..."
if [ -d "${INSTALL_DIR}/app" ]; then
    check_pass "App directory exists"
else
    check_fail "App directory missing"
fi

if [ -d "${INSTALL_DIR}/dev_postgres" ]; then
    check_pass "PostgreSQL config directory exists"
else
    check_fail "PostgreSQL config directory missing"
fi

# 2. Check required files
echo "[2/10] Checking required files..."
REQUIRED_FILES=(
    "requirements.txt"
    "app/main.py"
    "app/config.py"
    "dev_postgres/docker-compose.yaml"
    "setup-dishchat.sh"
    "start-dishchat.sh"
    "stop-dishchat.sh"
    "restart-dishchat.sh"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "${file}" ]; then
        check_pass "File exists: ${file}"
    else
        check_fail "File missing: ${file}"
    fi
done

# 3. Check Python version
echo "[3/10] Checking Python version..."
if command -v python3.12 &> /dev/null; then
    PYTHON_VER=$(python3.12 --version)
    check_pass "Python 3.12 installed: ${PYTHON_VER}"
else
    check_fail "Python 3.12 not found"
fi

# 4. Check virtual environment
echo "[4/10] Checking virtual environment..."
if [ -d "${INSTALL_DIR}/.venv" ]; then
    check_pass "Virtual environment exists"
    
    if [ -f "${INSTALL_DIR}/.venv/bin/python" ]; then
        VENV_PYTHON=$( ${INSTALL_DIR}/.venv/bin/python --version)
        check_pass "Virtual environment Python: ${VENV_PYTHON}"
    else
        check_fail "Virtual environment Python not found"
    fi
else
    check_fail "Virtual environment not found"
fi

# 5. Check Python packages
echo "[5/10] Checking Python packages..."
source ${INSTALL_DIR}/.venv/bin/activate 2>/dev/null

REQUIRED_PACKAGES=("fastapi" "langchain" "psycopg" "alembic" "uvicorn")
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import ${pkg}" 2>/dev/null; then
        check_pass "Package installed: ${pkg}"
    else
        check_fail "Package missing: ${pkg}"
    fi
done

# 6. Check Docker
echo "[6/10] Checking Docker..."
if command -v docker &> /dev/null; then
    DOCKER_VER=$(docker --version)
    check_pass "Docker installed: ${DOCKER_VER}"
else
    check_fail "Docker not installed"
fi

if docker compose version &> /dev/null 2>&1; then
    COMPOSE_VER=$(docker compose version)
    check_pass "Docker Compose installed: ${COMPOSE_VER}"
else
    check_fail "Docker Compose not installed"
fi

# 7. Check PostgreSQL container
echo "[7/10] Checking PostgreSQL container..."
if docker ps | grep -q "postgres-dev-dishchat"; then
    check_pass "PostgreSQL container is running"
    
    # Check if DB is accessible
    if docker exec postgres-dev-dishchat pg_isready -U dev_user -d dishchat &> /dev/null; then
        check_pass "PostgreSQL is accepting connections"
    else
        check_fail "PostgreSQL is not accepting connections"
    fi
else
    check_fail "PostgreSQL container is not running"
fi

# 8. Check backend process
echo "[8/10] Checking backend process..."
if [ -f "${INSTALL_DIR}/backend.pid" ]; then
    PID=$(cat ${INSTALL_DIR}/backend.pid)
    if kill -0 $PID 2>/dev/null; then
        check_pass "Backend process is running (PID: ${PID})"
    else
        check_fail "Backend process not running (stale PID file)"
    fi
else
    log_info "Backend not started yet (no PID file)"
fi

# 9. Check HTTP endpoint
echo "[9/10] Checking HTTP endpoint..."
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    check_pass "Backend responding on http://localhost:8000"
    
    # Try health endpoint
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/health 2>&1)
    if [ $? -eq 0 ]; then
        check_pass "Health endpoint accessible"
        log_info "Response: ${HEALTH_RESPONSE}"
    else
        check_fail "Health endpoint not accessible"
    fi
else
    log_info "Backend not responding (may not be started yet)"
fi

# 10. Check database migrations
echo "[10/10] Checking database migrations..."
cd ${INSTALL_DIR}/app
source ${INSTALL_DIR}/.venv/bin/activate
ALEMBIC_CURRENT=$(alembic current 2>&1)
if [ $? -eq 0 ]; then
    check_pass "Alembic migrations status:"
    log_info "${ALEMBIC_CURRENT}"
else
    check_fail "Could not check Alembic migrations"
    log_info "${ALEMBIC_CURRENT}"
fi

cd ${INSTALL_DIR}

echo ""
echo "========================================"
echo "Verification Results"
echo "========================================"
echo -e "${GREEN}Passed: ${PASS_COUNT}${NC}"
echo -e "${RED}Failed: ${FAIL_COUNT}${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Deployment appears successful. You can now:"
    echo "1. Start the backend: bash start-dishchat.sh"
    echo "2. View logs: tail -f logs/backend.log"
    echo "3. Access API docs: http://10.79.85.35:8000/docs"
    exit 0
else
    echo -e "${RED}✗ Some checks failed${NC}"
    echo ""
    echo "Please review the failures above and:"
    echo "1. Re-run setup if needed: bash setup-dishchat.sh"
    echo "2. Check logs for errors"
    echo "3. Refer to DEPLOYMENT_README.md for troubleshooting"
    exit 1
fi

