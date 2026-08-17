#!/bin/bash
# Dish-Chat Transfer Script from 3080 to 3090
# Source: montjac@10.79.85.47:~/dish-chat
# Target: jakebot@10.79.85.35:~/Jakes-agent
# Created: 2026-02-27

set -e

SOURCE_USER="montjac"
SOURCE_HOST="10.79.85.47"
SOURCE_DIR="/home/montjac/dish-chat"

TARGET_USER="jakebot"
TARGET_HOST="10.79.85.35"
TARGET_DIR="/home/jakebot/Jakes-agent"

echo "========================================"
echo "Dish-Chat Transfer Script"
echo "========================================"
echo "Source: $SOURCE_USER@$SOURCE_HOST:$SOURCE_DIR"
echo "Target: $TARGET_USER@$TARGET_HOST:$TARGET_DIR"
echo ""

# Files/directories to EXCLUDE (not needed for clean install)
EXCLUDE_PATTERNS=(
    "__pycache__"
    "*.pyc"
    "*.pyo"
    "*.log"
    "*.pid"
    ".venv"
    ".git"
    "*.backup*"
    "*.bak*"
    "backups/"
    "tmp/"
    "nohup.out"
    "build/"
    "firmware/"
    "*TEST*.md"
    "*SUMMARY*.txt"
    "*GUIDE*.txt"
    "*STATUS*.txt"
    "CHANGES_*"
    "DEPLOYMENT_*"
    "DELIVERABLES_*"
    "VERIFICATION_*"
    "IMPLEMENTATION_*"
    "JOURNAL_*"
    "FIX_*"
    "LOOP_*"
    "TOKEN_*"
    "TOOL_*"
    "VIEWERSHIP_*"
    "VISUALIZATION_*"
    "VIZ_*"
    "FINAL_*"
    "server.log"
    "backend.log"
    "server_startup.log"
    "data-gym-cache/"
    "dev_minio_data/"
    "apps/"
    "agent/" # Will be in app/agent already
    "embedded/"
    "instructions/"
    "journals/"
    "state/"
)

# Build rsync exclude options
EXCLUDE_OPTS=""
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    EXCLUDE_OPTS="$EXCLUDE_OPTS --exclude=$pattern"
done

echo "[1/3] Preparing target directory..."
ssh ${TARGET_USER}@${TARGET_HOST} "mkdir -p ${TARGET_DIR}"

echo "[2/3] Transferring files via rsync..."
echo "Excluding: __pycache__, .venv, .git, logs, backups, temp files..."
rsync -avz --progress \
    $EXCLUDE_OPTS \
    ${SOURCE_USER}@${SOURCE_HOST}:${SOURCE_DIR}/ \
    ${TARGET_USER}@${TARGET_HOST}:${TARGET_DIR}/

echo "[3/3] Transfer complete!"
echo ""
echo "Next steps:"
echo "1. SSH to target: ssh ${TARGET_USER}@${TARGET_HOST}"
echo "2. Run setup script: cd ~/Jakes-agent && bash setup-dishchat.sh"
echo ""
echo "========================================"

