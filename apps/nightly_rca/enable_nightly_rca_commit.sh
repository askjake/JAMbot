#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# enable_nightly_rca_commit.sh — Safely enable commit mode for nightly RCA
# ═══════════════════════════════════════════════════════════════════════════════
#
# USAGE:
#   bash enable_nightly_rca_commit.sh            # Dry-run: show what would change
#   bash enable_nightly_rca_commit.sh --confirm  # Actually enable commit mode
#   bash enable_nightly_rca_commit.sh --revert   # Restore from backup
#
# This script modifies /home/jakebot/Jakes-agent/config/nightly_rca.env to set:
#   NIGHTLY_RCA_COMMIT=true
#   NIGHTLY_RCA_WRITE_AUTHORIZED=true
#
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

ENV_FILE="/home/jakebot/Jakes-agent/config/nightly_rca.env"
BACKUP_DIR="/home/jakebot/Jakes-agent/config/backups"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

MODE="dry-run"
if [[ "${1:-}" == "--confirm" ]]; then
    MODE="confirm"
elif [[ "${1:-}" == "--revert" ]]; then
    MODE="revert"
fi

if [[ "$MODE" == "revert" ]]; then
    echo "Looking for most recent backup..."
    LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/nightly_rca.env.* 2>/dev/null | head -1)
    if [[ -z "$LATEST_BACKUP" ]]; then
        echo -e "${RED}No backups found in $BACKUP_DIR${NC}"
        exit 1
    fi
    echo "Restoring from: $LATEST_BACKUP"
    cp "$LATEST_BACKUP" "$ENV_FILE"
    echo -e "${GREEN}Reverted to backup. Current values:${NC}"
    grep -E "^NIGHTLY_RCA_(COMMIT|WRITE_AUTHORIZED)=" "$ENV_FILE"
    exit 0
fi

echo "═══════════════════════════════════════════════════════════════"
echo " Nightly RCA Commit Mode Enablement"
echo " Mode: $MODE"
echo " Timestamp: $STAMP"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Show current state
echo "Current values:"
grep -E "^NIGHTLY_RCA_(COMMIT|WRITE_AUTHORIZED)=" "$ENV_FILE" || true
echo ""

if [[ "$MODE" == "dry-run" ]]; then
    echo -e "${YELLOW}DRY RUN — no changes will be made.${NC}"
    echo ""
    echo "Would change:"
    echo "  NIGHTLY_RCA_COMMIT=false → NIGHTLY_RCA_COMMIT=true"
    echo "  NIGHTLY_RCA_WRITE_AUTHORIZED=false → NIGHTLY_RCA_WRITE_AUTHORIZED=true"
    echo ""
    echo "To apply, run:"
    echo "  bash $0 --confirm"
    echo ""
    echo "To revert after applying:"
    echo "  bash $0 --revert"
    exit 0
fi

# ─── Confirm mode ───────────────────────────────────────────────────────────
echo "Creating backup..."
mkdir -p "$BACKUP_DIR"
cp "$ENV_FILE" "$BACKUP_DIR/nightly_rca.env.${STAMP}"
echo -e "  ${GREEN}Backup saved:${NC} $BACKUP_DIR/nightly_rca.env.${STAMP}"
echo ""

echo "Applying changes..."
sed -i "s/^NIGHTLY_RCA_COMMIT=false/NIGHTLY_RCA_COMMIT=true/" "$ENV_FILE"
sed -i "s/^NIGHTLY_RCA_WRITE_AUTHORIZED=false/NIGHTLY_RCA_WRITE_AUTHORIZED=true/" "$ENV_FILE"

echo -e "  ${GREEN}Done.${NC} New values:"
grep -E "^NIGHTLY_RCA_(COMMIT|WRITE_AUTHORIZED)=" "$ENV_FILE"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo -e "${GREEN}Commit mode ENABLED.${NC}"
echo ""
echo "The next cron run (0 2 * * * in the cron daemon timezone) will execute with --commit."
echo "Timezone remains unverified unless CRON_TZ or TZ is configured and recorded."
echo ""
echo "To verify the pipeline runs correctly in commit mode:"
echo "  # Manual test run (respects env file):"
echo "  /home/jakebot/Jakes-agent/apps/nightly_rca/run_nightly.sh"
echo ""
echo "To revert:"
echo "  bash $0 --revert"
echo "═══════════════════════════════════════════════════════════════"
