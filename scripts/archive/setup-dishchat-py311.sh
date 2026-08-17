#!/bin/bash
# Modified setup script for Python 3.11
set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

INSTALL_DIR="/home/jakebot/Jakes-agent"
LOG_DIR="${INSTALL_DIR}/logs"
VENV_DIR="${INSTALL_DIR}/.venv"

echo "========================================"
echo "Dish-Chat Setup (Python 3.11)"
echo "========================================"

cd ${INSTALL_DIR}

# Check Python 3.11
if command -v python3.11 &> /dev/null; then
    PYTHON_VER=$(python3.11 --version)
    log_info "✓ Python 3.11 found: $PYTHON_VER"
else
    log_error "Python 3.11 not found"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    log_info "✓ Docker installed"
else
    log_error "Docker not installed"
    exit 1
fi

# Create virtual environment
log_info "Creating virtual environment with Python 3.11..."
python3.11 -m venv ${VENV_DIR}
log_info "✓ Virtual environment created"

# Activate and install dependencies
source ${VENV_DIR}/bin/activate
log_info "✓ Virtual environment activated"

log_info "Upgrading pip..."
pip install --upgrade pip

log_info "Installing requirements..."
pip install -r requirements.txt
log_info "✓ Python dependencies installed"

# PostgreSQL setup
log_info "Setting up PostgreSQL..."
cd dev_postgres

if docker ps | grep -q "postgres-dev-dishchat"; then
    log_warn "PostgreSQL container already running"
else
    log_info "Starting PostgreSQL container..."
    docker compose up -d
fi

# Wait for PostgreSQL
log_info "Waiting for PostgreSQL..."
sleep 5

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
    log_error "PostgreSQL failed to start"
    exit 1
fi

cd ${INSTALL_DIR}

# Database migrations
log_info "Running database migrations..."
cd app
alembic upgrade head
log_info "✓ Database migrations complete"
cd ${INSTALL_DIR}

# Create log directory
mkdir -p ${LOG_DIR}
log_info "✓ Log directory created"

# Create .env if needed
if [ ! -f ".env" ]; then
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
    log_info "✓ .env file exists"
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo "To start: bash start-dishchat.sh"

