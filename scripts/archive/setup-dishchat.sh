#!/bin/bash
# Dish-Chat Setup Script for 3090 (jakebot@10.79.85.35)
# This script sets up the dish-chat application from scratch
# Created: 2026-02-27

set -e

INSTALL_DIR="/home/jakebot/Jakes-agent"
LOG_DIR="${INSTALL_DIR}/logs"
VENV_DIR="${INSTALL_DIR}/.venv"

echo "========================================"
echo "Dish-Chat Setup Script"
echo "========================================"
echo "Installation Directory: ${INSTALL_DIR}"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if command -v $1 &> /dev/null; then
        log_info "✓ $1 is installed"
        return 0
    else
        log_error "✗ $1 is not installed"
        return 1
    fi
}

# Change to install directory
cd ${INSTALL_DIR}

echo ""
echo "=== Phase 1: System Prerequisites Check ==="
log_info "Checking system prerequisites..."

# Check Python 3.12
if check_command python3.12; then
    PYTHON_VERSION=$(python3.12 --version)
    log_info "Python version: $PYTHON_VERSION"
else
    log_error "Python 3.12 is required but not found!"
    log_info "Install with: sudo apt-get install python3.12 python3.12-venv python3.12-dev"
    exit 1
fi

# Check Docker
if ! check_command docker; then
    log_error "Docker is required but not found!"
    log_info "Install from: https://docs.docker.com/engine/install/"
    exit 1
fi

# Check Docker Compose
if docker compose version &> /dev/null; then
    log_info "✓ docker compose is installed"
else
    log_error "✗ docker compose is not installed"
    exit 1
fi

# Check git
check_command git || log_warn "git not found (optional)"

echo ""
echo "=== Phase 2: Create Virtual Environment ==="
if [ -d "${VENV_DIR}" ]; then
    log_warn "Virtual environment already exists at ${VENV_DIR}"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing existing venv..."
        rm -rf ${VENV_DIR}
    else
        log_info "Keeping existing venv..."
    fi
fi

if [ ! -d "${VENV_DIR}" ]; then
    log_info "Creating virtual environment with Python 3.12..."
    python3.12 -m venv ${VENV_DIR}
    log_info "✓ Virtual environment created"
fi

# Activate virtual environment
source ${VENV_DIR}/bin/activate
log_info "✓ Virtual environment activated"

echo ""
echo "=== Phase 3: Install Python Dependencies ==="
log_info "Upgrading pip..."
pip install --upgrade pip

log_info "Installing requirements from requirements.txt..."
pip install -r requirements.txt
log_info "✓ Python dependencies installed"

echo ""
echo "=== Phase 4: PostgreSQL Database Setup ==="
log_info "Setting up PostgreSQL with Docker..."

cd dev_postgres

# Check if container is already running
if docker ps | grep -q "postgres-dev-dishchat"; then
    log_warn "PostgreSQL container is already running"
    read -p "Restart container? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Stopping existing container..."
        docker compose down
        log_info "Starting fresh container..."
        docker compose up -d
    fi
else
    log_info "Starting PostgreSQL container..."
    docker compose up -d
fi

# Wait for PostgreSQL to be ready
log_info "Waiting for PostgreSQL to be ready..."
sleep 5

# Check if PostgreSQL is responding
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker exec postgres-dev-dishchat pg_isready -U dev_user -d dishchat &> /dev/null; then
        log_info "✓ PostgreSQL is ready"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -n "."
    sleep 1
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    log_error "PostgreSQL failed to start within timeout"
    exit 1
fi

cd ${INSTALL_DIR}

echo ""
echo "=== Phase 5: Database Migration ==="
log_info "Running Alembic migrations..."
cd app
alembic upgrade head
log_info "✓ Database migrations complete"
cd ${INSTALL_DIR}

echo ""
echo "=== Phase 6: Create Log Directory ==="
mkdir -p ${LOG_DIR}
log_info "✓ Log directory created: ${LOG_DIR}"

echo ""
echo "=== Phase 7: Environment Configuration ==="
if [ ! -f ".env" ]; then
    log_warn ".env file not found, creating from template..."
    cat > .env << 'ENV_EOF'
# Sentry Configuration
SENTRY_AUTH_TOKEN="${SENTRY_AUTH_TOKEN:?Set SENTRY_AUTH_TOKEN from the approved secret store}"
SENTRY_ORG=dishtv.technology
SENTRY_URL=https://ds-testing-sentry

# AI Thought Visualization Configuration
AGENT_THOUGHT_CAPTURE_ENABLED=true
AGENT_VIZ_SERVER_URL=http://localhost:8000/rest/api/v1/viz/event
FASTAPI_PORT=8000
FASTAPI_HOST=0.0.0.0
ENV_EOF
    log_info "✓ .env file created"
else
    log_info "✓ .env file already exists"
fi

echo ""
echo "=== Phase 8: Verification ==="
log_info "Verifying setup..."

# Check venv
if [ -f "${VENV_DIR}/bin/python" ]; then
    log_info "✓ Virtual environment: OK"
else
    log_error "✗ Virtual environment: FAILED"
fi

# Check PostgreSQL container
if docker ps | grep -q "postgres-dev-dishchat"; then
    log_info "✓ PostgreSQL container: RUNNING"
else
    log_error "✗ PostgreSQL container: NOT RUNNING"
fi

# Check required Python packages
source ${VENV_DIR}/bin/activate
if python -c "import fastapi, langchain, psycopg, alembic" 2>/dev/null; then
    log_info "✓ Python packages: OK"
else
    log_error "✗ Python packages: MISSING"
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To start Dish-Chat:"
echo "  cd ${INSTALL_DIR}"
echo "  bash start-dishchat.sh"
echo ""
echo "To stop Dish-Chat:"
echo "  bash stop-dishchat.sh"
echo ""
echo "To restart Dish-Chat:"
echo "  bash restart-dishchat.sh"
echo ""
echo "Logs are in: ${LOG_DIR}"
echo "========================================"

