#!/usr/bin/env bash
# Start the Google Drive MCP server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV="${REPO_ROOT}/.venv"
PORT="${GDRIVE_MCP_PORT:-8090}"

# Source .env if present
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

mkdir -p "${REPO_ROOT}/state"

echo "[gdrive-mcp] Starting on port ${PORT} ..."
exec "${VENV}/bin/uvicorn" apps.gdrive_mcp.server:mcp_app \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --log-level info \
    --app-dir "${REPO_ROOT}"
