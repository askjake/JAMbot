#!/usr/bin/env bash
set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${NIGHTLY_RCA_ENV_FILE:-$REPO_ROOT/config/nightly_rca.env}"

# Load deployment-specific values before deriving paths and limits.
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  # Path only; no environment values are logged or persisted by this marker.
  export NIGHTLY_RCA_LAUNCHER_ENV_SOURCE="$ENV_FILE"
fi

VENV="${NIGHTLY_RCA_VENV:-$REPO_ROOT/.venv}"
LOG_DIR="${NIGHTLY_RCA_LOG_DIR:-$REPO_ROOT/logs/nightly_rca}"
OUTPUT_DIR="${NIGHTLY_RCA_OUTPUT_DIR:-$REPO_ROOT/var/nightly_rca_v6}"
LOCK_FILE="${NIGHTLY_RCA_LOCK_FILE:-$OUTPUT_DIR/nightly_rca.lock}"
MAX_SECONDS="${NIGHTLY_RCA_MAX_SECONDS:-5400}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "FATAL: virtualenv python missing: $VENV/bin/python" >&2
  exit 10
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "SKIP: another nightly RCA run holds $LOCK_FILE" >&2
  exit 0
fi

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export NIGHTLY_RCA_OUTPUT_DIR="$OUTPUT_DIR"

# Refresh credentials in the same Python environment used by the pipeline.
if [[ -f "$HOME/secgateway/bin/secgateway.py" ]]; then
  "$VENV/bin/python" "$HOME/secgateway/bin/secgateway.py" >/dev/null 2>&1 || true
fi

"$VENV/bin/python" - <<'PY'
from botocore.session import Session
creds = Session().get_credentials()
if creds is None or not creds.get_frozen_credentials().access_key:
    raise SystemExit("FATAL: AWS credentials unavailable")
PY

MODE=(--dry-run)
case "${NIGHTLY_RCA_COMMIT:-false}" in
  1|true|TRUE|yes|YES|on|ON)
    case "${NIGHTLY_RCA_WRITE_AUTHORIZED:-false}" in
      1|true|TRUE|yes|YES|on|ON) MODE=(--commit) ;;
      *) echo "FATAL: commit requested without NIGHTLY_RCA_WRITE_AUTHORIZED=true" >&2; exit 11 ;;
    esac
    ;;
esac

LOG_FILE="$LOG_DIR/run_${STAMP}.log"
echo "Nightly RCA v6 start: $STAMP mode=${MODE[*]}" | tee "$LOG_FILE"
set +e
timeout --signal=TERM --kill-after=60 "$MAX_SECONDS" \
  "$VENV/bin/python" -m apps.nightly_rca.run "${MODE[@]}" --log-level INFO \
  2>&1 | tee -a "$LOG_FILE"
RC=${PIPESTATUS[0]}
set -e

echo "Nightly RCA v6 exit: $RC" | tee -a "$LOG_FILE"
find "$LOG_DIR" -type f -name 'run_*.log' -mtime +30 -delete || true
exit "$RC"
