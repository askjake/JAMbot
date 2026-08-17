#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# verify_nightly_rca_cron.sh — Pre-flight verification for nightly RCA pipeline
# ═══════════════════════════════════════════════════════════════════════════════
#
# PURPOSE: Validates all dependencies, connectivity, and configuration before
#          enabling NIGHTLY_RCA_COMMIT=true. Run this BEFORE enabling commit mode.
#
# SAFETY MODEL:
#   The nightly RCA pipeline has a two-key safety system:
#   - NIGHTLY_RCA_COMMIT=true        → Pipeline attempts mutations
#   - NIGHTLY_RCA_WRITE_AUTHORIZED=true → Executor allows those mutations
#   Both must be true for any live write. This prevents accidental enablement.
#
# USAGE: bash verify_nightly_rca_cron.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

REPO_ROOT="/home/jakebot/Jakes-agent"
ENV_FILE="${REPO_ROOT}/config/nightly_rca.env"
VENV="${REPO_ROOT}/.venv"
LOG_DIR="${REPO_ROOT}/logs/nightly_rca"
RUN_SCRIPT="${REPO_ROOT}/apps/nightly_rca/run_nightly.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

check_pass() { echo -e "  ${GREEN}✓ PASS${NC}: $1"; ((PASS++)); }
check_fail() { echo -e "  ${RED}✗ FAIL${NC}: $1"; ((FAIL++)); }
check_warn() { echo -e "  ${YELLOW}⚠ WARN${NC}: $1"; ((WARN++)); }

echo "═══════════════════════════════════════════════════════════════"
echo " Nightly RCA Pipeline Pre-Flight Verification"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ─── 1. Cron Entry ──────────────────────────────────────────────────────────
echo "1. Cron Entry Check"
CRON_ENTRY=$(crontab -l 2>/dev/null | grep "NIGHTLY_RCA_PIPELINE")
if [[ -n "$CRON_ENTRY" ]]; then
    check_pass "Cron entry found: ${CRON_ENTRY:0:60}..."
    if echo "$CRON_ENTRY" | grep -q "0 2 \* \* \*"; then
        check_pass "Schedule is 0 2 * * * (2am UTC daily)"
    else
        check_warn "Schedule differs from expected 0 2 * * *"
    fi
    if echo "$CRON_ENTRY" | grep -q "$RUN_SCRIPT"; then
        check_pass "Points to correct run_nightly.sh"
    else
        check_fail "Cron entry points to unexpected script"
    fi
else
    check_fail "No NIGHTLY_RCA_PIPELINE cron entry found"
fi
echo ""

# ─── 2. Env File ────────────────────────────────────────────────────────────
echo "2. Environment File"
if [[ -f "$ENV_FILE" ]]; then
    check_pass "Env file exists: $ENV_FILE"
    if grep -q "NIGHTLY_RCA_S3_MCP_URL=" "$ENV_FILE"; then
        check_pass "S3 MCP URL configured"
    else
        check_fail "S3 MCP URL missing from env"
    fi
    if grep -q "NIGHTLY_RCA_RTR_MCP_URL=" "$ENV_FILE"; then
        check_pass "RTR MCP URL configured"
    else
        check_fail "RTR MCP URL missing from env"
    fi
    if grep -q "NIGHTLY_RCA_GRASSHOPPER_MCP_URL=" "$ENV_FILE"; then
        check_pass "Grasshopper MCP URL configured"
    else
        check_fail "Grasshopper MCP URL missing from env"
    fi
    COMMIT_VAL=$(grep "^NIGHTLY_RCA_COMMIT=" "$ENV_FILE" | cut -d= -f2)
    WRITE_VAL=$(grep "^NIGHTLY_RCA_WRITE_AUTHORIZED=" "$ENV_FILE" | cut -d= -f2)
    echo "    Current mode: COMMIT=$COMMIT_VAL, WRITE_AUTHORIZED=$WRITE_VAL"
else
    check_fail "Env file not found: $ENV_FILE"
fi
echo ""

# ─── 3. Virtual Environment ─────────────────────────────────────────────────
echo "3. Virtual Environment"
if [[ -x "$VENV/bin/python" ]]; then
    check_pass "Python executable: $VENV/bin/python"
    PY_VER=$("$VENV/bin/python" --version 2>&1)
    check_pass "Version: $PY_VER"
    # Check critical packages
    if "$VENV/bin/python" -c "import mcp" 2>/dev/null; then
        check_pass "mcp package available"
    else
        check_fail "mcp package NOT importable"
    fi
    if "$VENV/bin/python" -c "import httpx" 2>/dev/null; then
        check_pass "httpx package available"
    else
        check_fail "httpx package NOT importable"
    fi
    if "$VENV/bin/python" -c "import botocore" 2>/dev/null; then
        check_pass "botocore package available"
    else
        check_fail "botocore package NOT importable"
    fi
else
    check_fail "Venv python not found: $VENV/bin/python"
fi
echo ""

# ─── 4. AWS Credentials ─────────────────────────────────────────────────────
echo "4. AWS Credentials"
CRED_CHECK=$("$VENV/bin/python" -c "
from botocore.session import Session
creds = Session().get_credentials()
if creds and creds.get_frozen_credentials().access_key:
    print('VALID')
else:
    print('INVALID')
" 2>/dev/null)
if [[ "$CRED_CHECK" == "VALID" ]]; then
    check_pass "AWS credentials are fresh and valid"
else
    check_fail "AWS credentials missing or expired — run secgateway"
fi
echo ""

# ─── 5. MCP Endpoint Health ─────────────────────────────────────────────────
echo "5. MCP Endpoint Connectivity"
# Source the env to get URLs
set -a; source "$ENV_FILE" 2>/dev/null; set +a

for NAME_URL in "S3:${NIGHTLY_RCA_S3_MCP_URL:-}" "RTR:${NIGHTLY_RCA_RTR_MCP_URL:-}" "Grasshopper:${NIGHTLY_RCA_GRASSHOPPER_MCP_URL:-}"; do
    NAME="${NAME_URL%%:*}"
    URL="${NAME_URL#*:}"
    if [[ -z "$URL" ]]; then
        check_fail "$NAME MCP URL is empty"
        continue
    fi
    # Quick HTTP check (just verify the endpoint responds)
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 -X POST "$URL"         -H "Content-Type: application/json"         -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"verify","version":"1.0"}},"id":1}' 2>/dev/null)
    if [[ "$HTTP_CODE" == "200" ]] || [[ "$HTTP_CODE" == "202" ]]; then
        check_pass "$NAME MCP endpoint responding (HTTP $HTTP_CODE)"
    elif [[ "$HTTP_CODE" == "000" ]]; then
        check_fail "$NAME MCP endpoint unreachable (timeout/DNS)"
    else
        check_warn "$NAME MCP endpoint returned HTTP $HTTP_CODE"
    fi
done
echo ""

# ─── 6. Log Directory ───────────────────────────────────────────────────────
echo "6. Log Directory"
if [[ -d "$LOG_DIR" ]]; then
    check_pass "Log directory exists: $LOG_DIR"
    RECENT=$(find "$LOG_DIR" -name "run_*.log" -mtime -2 2>/dev/null | wc -l)
    if [[ "$RECENT" -gt 0 ]]; then
        check_pass "Found $RECENT log files from last 48h"
    else
        check_warn "No log files from last 48h — pipeline may not be running"
    fi
else
    check_fail "Log directory missing: $LOG_DIR"
fi
echo ""

# ─── 7. Run Script ──────────────────────────────────────────────────────────
echo "7. Run Script"
if [[ -f "$RUN_SCRIPT" ]]; then
    check_pass "run_nightly.sh exists"
    if [[ -x "$RUN_SCRIPT" ]]; then
        check_pass "run_nightly.sh is executable"
    else
        check_fail "run_nightly.sh is NOT executable"
    fi
else
    check_fail "run_nightly.sh not found: $RUN_SCRIPT"
fi
echo ""

# ─── 8. Pipeline Module ─────────────────────────────────────────────────────
echo "8. Pipeline Module"
MODULE_CHECK=$("$VENV/bin/python" -c "
import sys; sys.path.insert(0, '$REPO_ROOT')
from apps.nightly_rca.run import main
print('OK')
" 2>/dev/null)
if [[ "$MODULE_CHECK" == "OK" ]]; then
    check_pass "apps.nightly_rca.run module importable"
else
    check_fail "Pipeline module import failed"
fi
echo ""

# ─── Summary ────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo " Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}, ${YELLOW}$WARN warnings${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [[ $FAIL -eq 0 ]]; then
    echo -e "${GREEN}ALL CHECKS PASSED.${NC}"
    echo ""
    echo "To enable commit mode, run:"
    echo "  bash enable_nightly_rca_commit.sh --confirm"
    echo ""
    echo "Or verify with dry-run first:"
    echo "  bash enable_nightly_rca_commit.sh"
    exit 0
else
    echo -e "${RED}$FAIL check(s) FAILED. Fix issues before enabling commit mode.${NC}"
    exit 1
fi
