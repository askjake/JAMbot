#!/usr/bin/env bash
# Safe Dish-Chat restart for dsgpu3090-Lambda-Vector.
#
# Critical guarantees:
#   * Uses only the existing dishchat-postgres container.
#   * Never runs pg_resetwal.
#   * Never deletes/recreates a PostgreSQL volume or database.
#   * Never launches the legacy dev_postgres compose stack.
#   * Validates the LangGraph migration ledger before backend startup.
#   * Starts backend 8002 without --reload.
#   * Starts the verified chats frontend on 3002.
#   * Refuses to kill an unrelated process merely because it owns a port.

set -Eeuo pipefail
umask 077

INSTALL_DIR="${INSTALL_DIR:-/home/jakebot/Jakes-agent}"
VENV_DIR="${INSTALL_DIR}/.venv"
BACKEND_PY="${VENV_DIR}/bin/python"
BACKEND_PID_FILE="${INSTALL_DIR}/backend.pid"

FRONTEND_DIR="/home/jakebot/Jakes-agent-fe/apps/chats"
FRONTEND_PID_FILE="${FRONTEND_DIR}/frontend.pid"

# --- Deterministic frontend Node/pnpm runtime -----------------------------
# The frontend must never inherit "whichever node appears first in PATH".
# Under a clean non-interactive environment /usr/bin/node is v12 and the
# Volta shim also resolves to v12, which cannot execute pnpm 10 (optional
# chaining) or Next.js 16. These values are inputs to resolve_frontend_runtime,
# which selects absolute executables and validates them by execution.
FRONTEND_PKG_ROOT="${FRONTEND_PKG_ROOT:-/home/jakebot/Jakes-agent-fe/package.json}"
NVM_NODE_ROOT="${NVM_NODE_ROOT:-/home/montjac/.nvm/versions/node}"
DISHCHAT_NODE_BIN="${DISHCHAT_NODE_BIN:-}"
DISHCHAT_PNPM_BIN="${DISHCHAT_PNPM_BIN:-}"
FALLBACK_NODE_MAJOR="${FALLBACK_NODE_MAJOR:-20}"

# Resolved by resolve_frontend_runtime(); absolute paths only.
NODE_BIN=""
NODE_VERSION=""
PNPM_ENTRY=""
PNPM_VERSION=""
REQUIRED_NODE_MAJOR=""

LOG_DIR="${INSTALL_DIR}/logs"
FRONTEND_LOG_DIR="/home/montjac/Jakes-agent-fe-logs"
RUN_STATE_FILE="${INSTALL_DIR}/dishchat-runtime.env"
LOCK_FILE="${LOCK_FILE:-${INSTALL_DIR}/.restart-dishchat.lock}"
DVA_GATEWAY_START="${DVA_GATEWAY_START:-/home/montjac/dva-gateway/start-dva-gateway.sh}"
DVA_GATEWAY_LOG="${DVA_GATEWAY_LOG:-/home/montjac/dva-gateway/dva-gateway.log}"

PG_CONTAINER="dishchat-postgres"
PG_USER="dev_user"
PG_DB="dishchat"
PG_HOST_PORT=5434

BACKEND_HOST="0.0.0.0"
BACKEND_PORT=8002
BACKEND_HEALTH_URL="http://127.0.0.1:${BACKEND_PORT}/rest/api/v1/health"

FRONTEND_HOST="0.0.0.0"
FRONTEND_PORT=3002
FRONTEND_LOCAL_URL="http://127.0.0.1:${FRONTEND_PORT}/"
FRONTEND_NETWORK_URL="http://10.79.85.35:${FRONTEND_PORT}/"

MAX_PG_WAIT="${MAX_PG_WAIT:-60}"
MAX_BACKEND_WAIT="${MAX_BACKEND_WAIT:-480}"
MAX_FRONTEND_WAIT="${MAX_FRONTEND_WAIT:-180}"
START_OPTIONAL_SERVICES="${START_OPTIONAL_SERVICES:-1}"

CHECK_ONLY=0
CORE_ONLY=0
STARTED_BACKEND=0
STARTED_FRONTEND=0
BACKEND_PID=""
FRONTEND_PID=""
BACKEND_LOG=""
FRONTEND_LOG=""
DOCKER=()

usage() {
    cat <<'EOF'
Usage:
  restart-dishchat.sh [--check] [--core-only]

Options:
  --check       Validate PostgreSQL, schema, migration ledger, paths, and ports.
                Do not stop or start application processes.
  --core-only   Restart PostgreSQL container if needed, backend, and frontend.
                Skip optional gateways, port-forwards, and MCP helpers.
  -h, --help    Show this help.

Environment:
  START_OPTIONAL_SERVICES=0  Same effect as --core-only.
  MAX_PG_WAIT=60
  MAX_BACKEND_WAIT=480
  MAX_FRONTEND_WAIT=180
EOF
}

while (($#)); do
    case "$1" in
        --check)
            CHECK_ONLY=1
            ;;
        --core-only)
            CORE_ONLY=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 64
            ;;
    esac
    shift
done

if [[ "${CORE_ONLY}" -eq 1 ]]; then
    START_OPTIONAL_SERVICES=0
fi

if [[ -t 1 ]]; then
    GREEN=$'\033[0;32m'
    RED=$'\033[0;31m'
    YELLOW=$'\033[1;33m'
    BLUE=$'\033[0;34m'
    NC=$'\033[0m'
else
    GREEN=""
    RED=""
    YELLOW=""
    BLUE=""
    NC=""
fi

log_info()  { printf '%s[INFO]%s %s\n'  "${GREEN}" "${NC}" "$*"; }
log_warn()  { printf '%s[WARN]%s %s\n'  "${YELLOW}" "${NC}" "$*"; }
log_error() { printf '%s[ERROR]%s %s\n' "${RED}" "${NC}" "$*" >&2; }
section()   { printf '\n%s===== %s =====%s\n' "${BLUE}" "$*" "${NC}"; }

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        log_error "Required command is missing: $1"
        exit 2
    }
}

listener_pids() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | sort -nu || true
        return
    fi

    ss -ltnp 2>/dev/null |
        awk -v needle=":${port}" '
            index($0, needle) {
                text=$0
                while (match(text, /pid=[0-9]+/)) {
                    print substr(text, RSTART+4, RLENGTH-4)
                    text=substr(text, RSTART+RLENGTH)
                }
            }
        ' |
        sort -nu
}

pid_command() {
    ps -p "$1" -o args= 2>/dev/null || true
}

pid_cwd() {
    readlink -f "/proc/$1/cwd" 2>/dev/null || true
}

pid_group() {
    ps -p "$1" -o pgid= 2>/dev/null | tr -d ' ' || true
}

stop_process_group() {
    local label="$1"
    local pid="$2"
    local pgid current_pgid

    kill -0 "${pid}" 2>/dev/null || return 0

    pgid="$(pid_group "${pid}")"
    current_pgid="$(pid_group "$$")"

    log_info "Stopping ${label} PID ${pid}${pgid:+, process group ${pgid}}"

    if [[ -n "${pgid}" && "${pgid}" != "${current_pgid}" ]]; then
        kill -TERM -- "-${pgid}" 2>/dev/null || true
    else
        pkill -TERM -P "${pid}" 2>/dev/null || true
        kill -TERM "${pid}" 2>/dev/null || true
    fi

    for _ in $(seq 1 30); do
        kill -0 "${pid}" 2>/dev/null || return 0
        sleep 0.2
    done

    log_warn "${label} did not stop after SIGTERM; sending SIGKILL"
    if [[ -n "${pgid}" && "${pgid}" != "${current_pgid}" ]]; then
        kill -KILL -- "-${pgid}" 2>/dev/null || true
    else
        pkill -KILL -P "${pid}" 2>/dev/null || true
        kill -KILL "${pid}" 2>/dev/null || true
    fi
}

process_matches_backend() {
    local pid="$1"
    local cmd cwd
    cmd="$(pid_command "${pid}")"
    cwd="$(pid_cwd "${pid}")"

    [[ "${cwd}" == "${INSTALL_DIR}" &&
       "${cmd}" == *"uvicorn app.main:app"* &&
       "${cmd}" == *"--port ${BACKEND_PORT}"* ]]
}

process_matches_frontend() {
    local pid="$1"
    local cmd cwd
    cmd="$(pid_command "${pid}")"
    cwd="$(pid_cwd "${pid}")"

    [[ "${cwd}" == "${FRONTEND_DIR}" &&
       ( "${cmd}" == *"next dev"* ||
         "${cmd}" == *"next start"* ||
         "${cmd}" == *"next-server"* ) ]]
}

stop_owned_port() {
    local label="$1"
    local port="$2"
    local matcher="$3"
    local pids=()
    local pid cmd cwd

    mapfile -t pids < <(listener_pids "${port}")
    ((${#pids[@]})) || return 0

    for pid in "${pids[@]}"; do
        if "${matcher}" "${pid}"; then
            stop_process_group "${label}" "${pid}"
        else
            cmd="$(pid_command "${pid}")"
            cwd="$(pid_cwd "${pid}")"
            log_error "Refusing to kill unrelated PID ${pid} on port ${port}"
            log_error "  cwd: ${cwd:-unknown}"
            log_error "  cmd: ${cmd:-unknown}"
            return 1
        fi
    done

    sleep 1
    mapfile -t pids < <(listener_pids "${port}")
    if ((${#pids[@]})); then
        log_error "Port ${port} remains occupied after stopping ${label}: ${pids[*]}"
        return 1
    fi
}

wait_for_http_200() {
    local url="$1"
    local limit="$2"
    local pid="$3"
    local label="$4"
    local i code

    for ((i=0; i<limit; i++)); do
        code="$(
            curl -sS --connect-timeout 2 --max-time 5 \
                -o /dev/null -w '%{http_code}' "${url}" 2>/dev/null || true
        )"
        if [[ "${code}" == "200" ]]; then
            return 0
        fi

        if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
            log_error "${label} exited while waiting for ${url}"
            return 1
        fi

        sleep 1
    done

    log_error "${label} did not return HTTP 200 within ${limit} seconds"
    return 1
}

wait_for_port() {
    local port="$1"
    local limit="$2"
    local i
    for ((i=0; i<limit; i++)); do
        if (($(listener_pids "${port}" | wc -l) > 0)); then
            return 0
        fi
        sleep 1
    done
    return 1
}

detect_docker() {
    if docker info >/dev/null 2>&1; then
        DOCKER=(docker)
    elif sudo -n docker info >/dev/null 2>&1; then
        DOCKER=(sudo docker)
    else
        log_error "Docker requires privileges. Run 'sudo -v' and retry."
        exit 3
    fi
}

pg_ready() {
    "${DOCKER[@]}" exec "${PG_CONTAINER}" \
        pg_isready -U "${PG_USER}" -d "${PG_DB}" >/dev/null 2>&1
}

validate_postgres() {
    local running mapping db_row quarantine_enabled quarantine_sessions
    local installed_count migration_diff task_path_count migration_rows
    local migration_stderr="${LOG_DIR}/migration-count.stderr"

    section "PostgreSQL safety and health"

    if ! "${DOCKER[@]}" inspect "${PG_CONTAINER}" >/dev/null 2>&1; then
        log_error "Required PostgreSQL container does not exist: ${PG_CONTAINER}"
        log_error "Refusing to launch the legacy dev_postgres compose stack."
        exit 4
    fi

    running="$("${DOCKER[@]}" inspect -f '{{.State.Running}}' "${PG_CONTAINER}")"
    if [[ "${running}" != "true" ]]; then
        if [[ "${CHECK_ONLY}" -eq 1 ]]; then
            log_error "${PG_CONTAINER} exists but is stopped"
            exit 4
        fi
        log_info "Starting existing PostgreSQL container ${PG_CONTAINER}"
        "${DOCKER[@]}" start "${PG_CONTAINER}" >/dev/null
    fi

    for _ in $(seq 1 "${MAX_PG_WAIT}"); do
        pg_ready && break
        sleep 1
    done

    if ! pg_ready; then
        log_error "PostgreSQL did not become ready. No destructive recovery was attempted."
        "${DOCKER[@]}" inspect "${PG_CONTAINER}" \
            --format='status={{.State.Status}} restarting={{.State.Restarting}} exit={{.State.ExitCode}}' \
            2>/dev/null || true
        "${DOCKER[@]}" logs --tail 100 "${PG_CONTAINER}" 2>&1 || true
        exit 4
    fi

    mapping="$("${DOCKER[@]}" port "${PG_CONTAINER}" 5432/tcp 2>/dev/null || true)"
    if ! grep -Eq '(^|:)5434$' <<<"${mapping}"; then
        log_error "Unexpected PostgreSQL port mapping for ${PG_CONTAINER}: ${mapping:-none}"
        exit 4
    fi

    db_row="$(
        "${DOCKER[@]}" exec "${PG_CONTAINER}" \
            psql -X --no-psqlrc -U "${PG_USER}" -d postgres -qAtc \
            "SELECT datallowconn::int || '|' || pg_database_size(oid)
             FROM pg_database
             WHERE datname='${PG_DB}';"
    )"
    [[ -n "${db_row}" ]] || {
        log_error "Database ${PG_DB} does not exist"
        exit 4
    }
    [[ "${db_row%%|*}" == "1" ]] || {
        log_error "Database ${PG_DB} does not allow connections"
        exit 4
    }

    quarantine_enabled="$(
        "${DOCKER[@]}" exec "${PG_CONTAINER}" \
            psql -X --no-psqlrc -U "${PG_USER}" -d postgres -qAtc \
            "SELECT count(*)
             FROM pg_database
             WHERE datname LIKE 'dishchat_old_history_%'
               AND datallowconn;"
    )"
    [[ "${quarantine_enabled}" == "0" ]] || {
        log_error "A retained old-history database unexpectedly allows connections"
        exit 4
    }

    quarantine_sessions="$(
        "${DOCKER[@]}" exec "${PG_CONTAINER}" \
            psql -X --no-psqlrc -U "${PG_USER}" -d postgres -qAtc \
            "SELECT count(*)
             FROM pg_stat_activity
             WHERE datname LIKE 'dishchat_old_history_%';"
    )"
    [[ "${quarantine_sessions}" == "0" ]] || {
        log_error "A retained old-history database still has active sessions"
        exit 4
    }

    : > "${migration_stderr}"
    if ! installed_count="$(
        cd "${INSTALL_DIR}"
        PYTHONWARNINGS=ignore "${BACKEND_PY}" - <<'PY' 2>"${migration_stderr}"
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
print(len(AsyncPostgresSaver.MIGRATIONS))
PY
    )"; then
        log_error "Unable to read installed LangGraph migration count"
        cat "${migration_stderr}" >&2 || true
        exit 5
    fi
    [[ "${installed_count}" =~ ^[0-9]+$ && "${installed_count}" -gt 0 ]] || {
        log_error "Invalid installed LangGraph migration count: ${installed_count}"
        exit 5
    }

    migration_diff="$(
        "${DOCKER[@]}" exec "${PG_CONTAINER}" \
            psql -X --no-psqlrc -U "${PG_USER}" -d "${PG_DB}" -qAtc \
            "WITH expected(v) AS (
                 SELECT generate_series(0, $((installed_count - 1)))
             ),
             differences AS (
                 (SELECT v FROM public.checkpoint_migrations
                  EXCEPT SELECT v FROM expected)
                 UNION ALL
                 (SELECT v FROM expected
                  EXCEPT SELECT v FROM public.checkpoint_migrations)
             )
             SELECT count(*) FROM differences;"
    )"
    [[ "${migration_diff}" == "0" ]] || {
        log_error "LangGraph checkpoint migration ledger does not match the installed package"
        "${DOCKER[@]}" exec "${PG_CONTAINER}" \
            psql -X --no-psqlrc -U "${PG_USER}" -d "${PG_DB}" \
            -P pager=off -c "TABLE public.checkpoint_migrations;" || true
        exit 5
    }

    migration_rows="$(
        "${DOCKER[@]}" exec "${PG_CONTAINER}" \
            psql -X --no-psqlrc -U "${PG_USER}" -d "${PG_DB}" -qAtc \
            "SELECT count(*) FROM public.checkpoint_migrations;"
    )"
    task_path_count="$(
        "${DOCKER[@]}" exec "${PG_CONTAINER}" \
            psql -X --no-psqlrc -U "${PG_USER}" -d "${PG_DB}" -qAtc \
            "SELECT count(*)
             FROM information_schema.columns
             WHERE table_schema='public'
               AND table_name='checkpoint_writes'
               AND column_name='task_path';"
    )"
    [[ "${task_path_count}" == "1" ]] || {
        log_error "checkpoint_writes.task_path is missing"
        exit 5
    }

    log_info "PostgreSQL healthy on host port ${PG_HOST_PORT}"
    log_info "${PG_DB} size: ${db_row#*|} bytes"
    log_info "LangGraph migration ledger: ${migration_rows}/${installed_count}, exact"
    if [[ "$(
        "${DOCKER[@]}" exec "${PG_CONTAINER}" \
            psql -X --no-psqlrc -U "${PG_USER}" -d postgres -qAtc \
            "SELECT count(*) FROM pg_database
             WHERE datname LIKE 'dishchat_old_history_%';"
    )" -gt 0 ]]; then
        log_warn "A disabled old-history rollback database is still retained"
    fi
}

stop_legacy_service() {
    if systemctl is-active --quiet dishchat-app.service 2>/dev/null; then
        log_warn "Stopping legacy dishchat-app.service"
        if [[ "${EUID}" -eq 0 ]]; then
            systemctl stop dishchat-app.service
        elif sudo -n systemctl stop dishchat-app.service; then
            :
        else
            log_error "Unable to stop legacy dishchat-app.service non-interactively"
            log_error "Run 'sudo -v' and retry"
            return 1
        fi
    fi

    if systemctl is-enabled --quiet dishchat-app.service 2>/dev/null; then
        log_warn "Legacy dishchat-app.service is enabled and may start on reboot"
        log_warn "After acceptance, disable it with:"
        log_warn "  sudo systemctl disable --now dishchat-app.service"
    fi
}

stop_core_services() {
    section "Stop exact backend and frontend processes"

    if [[ -f "${BACKEND_PID_FILE}" ]]; then
        local_pid="$(tr -dc '0-9' < "${BACKEND_PID_FILE}")"
        if [[ -n "${local_pid}" && -d "/proc/${local_pid}" ]]; then
            if process_matches_backend "${local_pid}"; then
                stop_process_group "backend" "${local_pid}"
            else
                log_warn "Ignoring stale/unowned backend PID file: ${local_pid}"
            fi
        fi
        rm -f "${BACKEND_PID_FILE}"
    fi

    if [[ -f "${FRONTEND_PID_FILE}" ]]; then
        local_pid="$(tr -dc '0-9' < "${FRONTEND_PID_FILE}")"
        if [[ -n "${local_pid}" && -d "/proc/${local_pid}" ]]; then
            if process_matches_frontend "${local_pid}"; then
                stop_process_group "frontend" "${local_pid}"
            else
                log_warn "Ignoring stale/unowned frontend PID file: ${local_pid}"
            fi
        fi
        rm -f "${FRONTEND_PID_FILE}"
    fi

    stop_owned_port "backend" "${BACKEND_PORT}" process_matches_backend
    stop_owned_port "frontend" "${FRONTEND_PORT}" process_matches_frontend
}

refresh_aws_tokens() {
    section "Refresh AWS credentials"
    if [[ -f "/home/montjac/secgateway/bin/secgateway.py" ]]; then
        if timeout 90 python3 /home/montjac/secgateway/bin/secgateway.py 9>&-; then
            log_info "AWS credentials refreshed"
        else
            log_warn "AWS credential refresh failed; continuing"
        fi
    else
        log_warn "SecGateway not found; continuing"
    fi
}

start_backend() {
    section "Start backend on ${BACKEND_PORT} without reload"

    BACKEND_LOG="${LOG_DIR}/backend-$(date -u +%Y%m%dT%H%M%SZ).log"
    (
        cd "${INSTALL_DIR}"
        nohup setsid "${BACKEND_PY}" -m uvicorn app.main:app \
            --host "${BACKEND_HOST}" \
            --port "${BACKEND_PORT}" \
            > "${BACKEND_LOG}" 2>&1 < /dev/null 9>&- &
        echo "$!" > "${BACKEND_PID_FILE}"
    )
    BACKEND_PID="$(cat "${BACKEND_PID_FILE}")"
    STARTED_BACKEND=1

    if ! wait_for_http_200 \
        "${BACKEND_HEALTH_URL}" "${MAX_BACKEND_WAIT}" "${BACKEND_PID}" "backend"; then
        tail -n 300 "${BACKEND_LOG}" >&2 || true
        return 1
    fi

    if grep -Eqi \
        'DuplicateColumn|DataCorrupted|missing chunk number|pg_toast_2619|pg_toast_16399' \
        "${BACKEND_LOG}"; then
        log_error "Database migration/corruption error detected in backend log"
        grep -Ein \
            'DuplicateColumn|DataCorrupted|missing chunk number|pg_toast_2619|pg_toast_16399' \
            "${BACKEND_LOG}" >&2 || true
        return 1
    fi

    if grep -q -- '--reload' < <(pid_command "${BACKEND_PID}"); then
        log_error "Backend unexpectedly started with --reload"
        return 1
    fi

    ln -sfn "${BACKEND_LOG}" "${LOG_DIR}/backend.latest.log"
    log_info "Backend healthy: ${BACKEND_HEALTH_URL}"
    log_info "Backend PID ${BACKEND_PID}; log ${BACKEND_LOG}"
}

# ---------------------------------------------------------------------------
# Deterministic frontend runtime resolution
# ---------------------------------------------------------------------------

# Derive the minimum required Node major from the frontend package engines
# field. Falls back to FALLBACK_NODE_MAJOR when the field is absent.
frontend_required_node_major() {
    local pkg="$1"
    local spec=""
    local major=""

    if [[ -f "${pkg}" ]]; then
        spec="$(awk '
            /"engines"[[:space:]]*:/ { in_eng = 1 }
            in_eng && /"node"[[:space:]]*:/ {
                line = $0
                sub(/.*"node"[[:space:]]*:[[:space:]]*"/, "", line)
                sub(/".*/, "", line)
                print line
                exit
            }
            in_eng && /}/ { in_eng = 0 }
        ' "${pkg}" 2>/dev/null || true)"
    fi

    major="$(printf '%s' "${spec}" | grep -oE '[0-9]+' | head -1 || true)"
    printf '%s\n' "${major:-${FALLBACK_NODE_MAJOR}}"
}

# True when a candidate resolves into a Volta-managed path. Volta pins an
# ancient default toolchain here, so such candidates are never trusted.
is_volta_managed() {
    local resolved
    resolved="$(readlink -f "$1" 2>/dev/null || true)"
    [[ "$1" == *"/.volta/"* || "${resolved}" == *"/.volta/"* ]]
}

# Full version string of a node candidate, e.g. "v20.20.1". Empty on failure.
node_version_of() {
    local out
    out="$("$1" --version 2>/dev/null || true)"
    [[ "${out}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+ ]] || return 1
    printf '%s\n' "${out}"
}

node_major_of() {
    local ver
    ver="$(node_version_of "$1")" || return 1
    ver="${ver#v}"
    printf '%s\n' "${ver%%.*}"
}

# The exact PATH used to launch the frontend. The verified node directory is
# placed first so any child process (pnpm, next, turbopack) resolves the same
# interpreter that was validated here.
frontend_runtime_path() {
    local node_bin="$1"
    printf '%s\n' "${node_bin%/*}:/usr/local/bin:/usr/bin:/bin"
}

# Execute the exact node+pnpm pair and return the reported pnpm version.
# Runs from FRONTEND_DIR so pnpm honours the packageManager pin exactly as it
# will during the real launch. The entry may be a symlink; node resolves it.
validate_frontend_runtime_pair() {
    local node_bin="$1"
    local pnpm_entry="$2"
    local ver=""

    [[ -x "${node_bin}" ]] || return 1
    [[ -e "${pnpm_entry}" ]] || return 1

    ver="$(cd "${FRONTEND_DIR}" 2>/dev/null &&
           PATH="$(frontend_runtime_path "${node_bin}")" \
           "${node_bin}" "${pnpm_entry}" --version 2>/dev/null || true)"

    [[ "${ver}" =~ ^[0-9]+\.[0-9]+\.[0-9]+ ]] || return 1
    printf '%s\n' "${ver}"
}

# Strict check: the same pair must also work in a minimal non-interactive
# environment with no shell profile, which is what a cron/systemd/ssh restart
# actually gets.
validate_frontend_runtime_pair_minimal_env() {
    local node_bin="$1"
    local pnpm_entry="$2"
    local ver=""

    ver="$(env -i \
             HOME="${HOME}" \
             USER="${USER:-$(id -un)}" \
             PATH="$(frontend_runtime_path "${node_bin}")" \
             bash --noprofile --norc -c \
             "cd '${FRONTEND_DIR}' && '${node_bin}' '${pnpm_entry}' --version" \
             2>/dev/null || true)"

    [[ "${ver}" =~ ^[0-9]+\.[0-9]+\.[0-9]+ ]] || return 1
    printf '%s\n' "${ver}"
}

node_candidates() {
    local entry
    if [[ -n "${DISHCHAT_NODE_BIN}" ]]; then
        printf '%s\n' "${DISHCHAT_NODE_BIN}"
        return 0
    fi
    if [[ -d "${NVM_NODE_ROOT}" ]]; then
        while read -r entry; do
            [[ -n "${entry}" ]] || continue
            printf '%s\n' "${NVM_NODE_ROOT}/${entry}/bin/node"
        done < <(ls -1 "${NVM_NODE_ROOT}" 2>/dev/null | sort -Vr || true)
    fi
    printf '%s\n' "/usr/local/bin/node"
    printf '%s\n' "/usr/bin/node"
}

pnpm_candidates() {
    if [[ -n "${DISHCHAT_PNPM_BIN}" ]]; then
        printf '%s\n' "${DISHCHAT_PNPM_BIN}"
        return 0
    fi
    printf '%s\n' "/usr/local/bin/pnpm"
    printf '%s\n' "/usr/local/lib/node_modules/pnpm/bin/pnpm.cjs"
}

# Select and validate the frontend runtime. MUST run during preflight, before
# any service is stopped. Prints only absolute paths and versions.
resolve_frontend_runtime() {
    section "Resolve deterministic frontend Node/pnpm runtime"

    local cand major ver pnpm_ver minimal_ver
    local explicit_node=0
    local explicit_pnpm=0

    [[ -n "${DISHCHAT_NODE_BIN}" ]] && explicit_node=1
    [[ -n "${DISHCHAT_PNPM_BIN}" ]] && explicit_pnpm=1

    REQUIRED_NODE_MAJOR="$(frontend_required_node_major "${FRONTEND_PKG_ROOT}")"
    log_info "Frontend requires Node major >= ${REQUIRED_NODE_MAJOR} (derived from ${FRONTEND_PKG_ROOT})"

    NODE_BIN=""
    NODE_VERSION=""
    while read -r cand; do
        [[ -n "${cand}" ]] || continue
        if [[ ! -x "${cand}" ]]; then
            continue
        fi
        if is_volta_managed "${cand}"; then
            log_warn "Rejecting Volta-managed Node candidate: ${cand}"
            continue
        fi
        if ! ver="$(node_version_of "${cand}")"; then
            log_warn "Rejecting unusable Node candidate: ${cand}"
            continue
        fi
        major="${ver#v}"
        major="${major%%.*}"
        if ((major < REQUIRED_NODE_MAJOR)); then
            log_warn "Rejecting ${cand}: Node ${ver} < required major ${REQUIRED_NODE_MAJOR}"
            continue
        fi
        NODE_BIN="${cand}"
        NODE_VERSION="${ver}"
        break
    done < <(node_candidates)

    if [[ -z "${NODE_BIN}" ]]; then
        if ((explicit_node == 1)); then
            log_error "DISHCHAT_NODE_BIN is set but unusable or below required major ${REQUIRED_NODE_MAJOR}: ${DISHCHAT_NODE_BIN}"
        else
            log_error "No Node >= ${REQUIRED_NODE_MAJOR} found. Set DISHCHAT_NODE_BIN to an absolute node path."
        fi
        return 2
    fi

    PNPM_ENTRY=""
    PNPM_VERSION=""
    while read -r cand; do
        [[ -n "${cand}" ]] || continue
        if [[ ! -e "${cand}" ]]; then
            continue
        fi
        if is_volta_managed "${cand}"; then
            log_warn "Rejecting Volta-managed pnpm candidate: ${cand}"
            continue
        fi
        if ! pnpm_ver="$(validate_frontend_runtime_pair "${NODE_BIN}" "${cand}")"; then
            log_warn "Rejecting pnpm candidate (failed to execute with ${NODE_BIN}): ${cand}"
            continue
        fi
        if ! minimal_ver="$(validate_frontend_runtime_pair_minimal_env "${NODE_BIN}" "${cand}")"; then
            log_warn "Rejecting pnpm candidate (failed in minimal non-interactive env): ${cand}"
            continue
        fi
        if [[ "${pnpm_ver}" != "${minimal_ver}" ]]; then
            log_warn "Rejecting pnpm candidate (version differs between environments): ${cand}"
            continue
        fi
        PNPM_ENTRY="${cand}"
        PNPM_VERSION="${pnpm_ver}"
        break
    done < <(pnpm_candidates)

    if [[ -z "${PNPM_ENTRY}" ]]; then
        if ((explicit_pnpm == 1)); then
            log_error "DISHCHAT_PNPM_BIN is set but unusable with ${NODE_BIN}: ${DISHCHAT_PNPM_BIN}"
        else
            log_error "No working pnpm found for ${NODE_BIN}. Set DISHCHAT_PNPM_BIN to an absolute pnpm path."
        fi
        return 2
    fi

    log_info "NODE_BIN=${NODE_BIN}"
    log_info "NODE_VERSION=${NODE_VERSION}"
    log_info "PNPM_ENTRY=${PNPM_ENTRY}"
    log_info "PNPM_VERSION=${PNPM_VERSION}"
    log_info "FRONTEND_RUNTIME_PATH=$(frontend_runtime_path "${NODE_BIN}")"
    return 0
}

start_frontend() {
    section "Start verified chats frontend on ${FRONTEND_PORT}"

    FRONTEND_LOG="${FRONTEND_LOG_DIR}/frontend-$(date -u +%Y%m%dT%H%M%SZ).log"
    if [[ -z "${NODE_BIN}" || -z "${PNPM_ENTRY}" ]]; then
        log_error "Frontend runtime not resolved; resolve_frontend_runtime must run first"
        return 2
    fi

    log_info "Launching frontend with ${NODE_BIN} (${NODE_VERSION}) and ${PNPM_ENTRY} (${PNPM_VERSION})"

    (
        cd "${FRONTEND_DIR}"
        nohup setsid env \
            PATH="$(frontend_runtime_path "${NODE_BIN}")" \
            "${NODE_BIN}" "${PNPM_ENTRY}" exec next dev \
            --turbopack \
            --hostname "${FRONTEND_HOST}" \
            --port "${FRONTEND_PORT}" \
            > "${FRONTEND_LOG}" 2>&1 < /dev/null 9>&- &
        echo "$!" > "${FRONTEND_PID_FILE}"
    )
    FRONTEND_PID="$(cat "${FRONTEND_PID_FILE}")"
    STARTED_FRONTEND=1

    if ! wait_for_http_200 \
        "${FRONTEND_LOCAL_URL}" "${MAX_FRONTEND_WAIT}" "${FRONTEND_PID}" "frontend"; then
        tail -n 300 "${FRONTEND_LOG}" >&2 || true
        return 1
    fi

    curl -sS --connect-timeout 5 --max-time 30 \
        "${FRONTEND_LOCAL_URL}" > "${LOG_DIR}/frontend-index.latest.html"

    if grep -q '_febec8d9' "${LOG_DIR}/frontend-index.latest.html"; then
        log_error "Retired frontend chunk _febec8d9 reappeared"
        return 1
    fi

    ln -sfn "${FRONTEND_LOG}" "${FRONTEND_LOG_DIR}/frontend.latest.log"
    log_info "Frontend healthy: ${FRONTEND_NETWORK_URL}"
    log_info "Frontend PID ${FRONTEND_PID}; log ${FRONTEND_LOG}"
}

port_is_listening() {
    (($(listener_pids "$1" | wc -l) > 0))
}

start_coverity_gateway() {
    local port=5001
    local pid_file="${INSTALL_DIR}/gateway.pid"
    local log_file="${LOG_DIR}/gateway.log"

    if port_is_listening "${port}"; then
        log_info "Coverity gateway port ${port} already listening; leaving it untouched"
        return 0
    fi

    [[ -f "${INSTALL_DIR}/coverity_assist_gateway.py" ]] || {
        log_warn "coverity_assist_gateway.py not found"
        return 0
    }

    (
        cd "${INSTALL_DIR}"
        nohup setsid "${BACKEND_PY}" "${INSTALL_DIR}/coverity_assist_gateway.py" \
            --port "${port}" >> "${log_file}" 2>&1 < /dev/null 9>&- &
        echo "$!" > "${pid_file}"
    )

    if wait_for_port "${port}" 20; then
        log_info "Coverity gateway started on ${port}"
    else
        log_warn "Coverity gateway did not open port ${port}"
    fi
}

start_qodo_port_forward() {
    local port=18443
    local pid_file="${INSTALL_DIR}/qodo-portforward.pid"
    local context="arn:aws:eks:us-west-2:233532778289:cluster/apps-xx-eks-gltqp-2fq9x"

    if port_is_listening "${port}"; then
        log_info "Qodo port ${port} already listening; leaving it untouched"
        return 0
    fi

    command -v kubectl >/dev/null 2>&1 || {
        log_warn "kubectl not found; skipping Qodo port-forward"
        return 0
    }

    if ! timeout 25 kubectl --context "${context}" \
        get svc qodo-ssh-proxy -n open-webui-dev >/dev/null 2>&1; then
        log_warn "Qodo service lookup failed; skipping"
        return 0
    fi

    nohup setsid kubectl --context "${context}" \
        port-forward -n open-webui-dev \
        svc/qodo-ssh-proxy 18443:8443 \
        >> "${LOG_DIR}/qodo-portforward.log" 2>&1 < /dev/null 9>&- &
    echo "$!" > "${pid_file}"

    if wait_for_port "${port}" 20; then
        log_info "Qodo port-forward started on ${port}"
    else
        log_warn "Qodo port-forward did not open ${port}"
    fi
}

start_dish_code_tools() {
    local port=8087
    local dir="/home/montjac/dish-code-tools"
    local python="${dir}/.venv/bin/python"
    local pid_file="${dir}/dish-code-tools.pid"

    if port_is_listening "${port}"; then
        log_info "dish-code-tools port ${port} already listening; leaving it untouched"
        return 0
    fi

    [[ -x "${python}" ]] || {
        log_warn "dish-code-tools virtual environment not found"
        return 0
    }

    (
        cd "${dir}"
        REPOS_BASE=/mnt/tnas/public/repos/code_tools \
            nohup setsid "${python}" -m app \
            >> "${LOG_DIR}/dish-code-tools.log" 2>&1 < /dev/null 9>&- &
        echo "$!" > "${pid_file}"
    )

    if wait_for_port "${port}" 30; then
        log_info "dish-code-tools started on ${port}"
    else
        log_warn "dish-code-tools did not open ${port}"
    fi
}

start_dva_services() {
    if [[ -f "${DVA_GATEWAY_START}" ]]; then
        if timeout 60 bash "${DVA_GATEWAY_START}" \
            >> "${DVA_GATEWAY_LOG}" 2>&1 9>&-; then
            log_info "DVA gateway start command completed"
        else
            log_warn "DVA gateway start command failed or timed out"
        fi
    else
        log_warn "DVA gateway start script not found"
    fi

    if timeout 20 ssh \
        -o BatchMode=yes \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=5 \
        montjac@10.79.85.47 \
        "bash ~/dva-gateway/start-dva-mcp.sh" >/dev/null 2>&1 9>&-; then
        log_info "DVA MCP remote start command completed"
    else
        log_warn "DVA MCP remote start skipped or failed"
    fi
}

start_optional_services() {
    section "Optional gateways and MCP helpers"
    start_coverity_gateway || true
    start_qodo_port_forward || true
    start_dish_code_tools || true
    start_dva_services || true
}

write_runtime_state() {
    cat > "${RUN_STATE_FILE}" <<EOF
DISHCHAT_RUNTIME_VERSION=20260803-safe-v1
POSTGRES_CONTAINER=${PG_CONTAINER}
POSTGRES_DATABASE=${PG_DB}
BACKEND_PID=${BACKEND_PID}
BACKEND_PORT=${BACKEND_PORT}
BACKEND_LOG=${BACKEND_LOG}
FRONTEND_PID=${FRONTEND_PID}
FRONTEND_PORT=${FRONTEND_PORT}
FRONTEND_LOG=${FRONTEND_LOG}
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
    chmod 600 "${RUN_STATE_FILE}"
}

cleanup_on_error() {
    local rc=$?
    trap - ERR INT TERM

    log_error "Restart failed with exit code ${rc}"
    if [[ "${STARTED_FRONTEND}" -eq 1 && -n "${FRONTEND_PID}" ]]; then
        stop_process_group "frontend started by this run" "${FRONTEND_PID}" || true
    fi
    if [[ "${STARTED_BACKEND}" -eq 1 && -n "${BACKEND_PID}" ]]; then
        stop_process_group "backend started by this run" "${BACKEND_PID}" || true
    fi

    log_error "PostgreSQL was not reset, recreated, removed, or stopped."
    exit "${rc}"
}
trap cleanup_on_error ERR INT TERM

main() {
    [[ -d "${INSTALL_DIR}" ]] || {
        log_error "Backend directory missing: ${INSTALL_DIR}"
        exit 2
    }
    mkdir -p "${LOG_DIR}" "${FRONTEND_LOG_DIR}"

    exec 9>"${LOCK_FILE}"
    if ! flock -n 9; then
        log_error "Another restart-dishchat.sh run is already active"
        exit 1
    fi

    section "Safe Dish-Chat startup"
    log_info "Mode: $([[ "${CHECK_ONLY}" -eq 1 ]] && echo check-only || echo restart)"
    log_info "Optional services: ${START_OPTIONAL_SERVICES}"

    for cmd in awk bash curl docker env flock grep id nohup pkill ps readlink sed seq setsid sort ss systemctl timeout tr wc; do
        require_command "${cmd}"
    done

    [[ -x "${BACKEND_PY}" ]] || {
        log_error "Backend Python missing: ${BACKEND_PY}"
        exit 2
    }
    [[ -d "${FRONTEND_DIR}" ]] || {
        log_error "Frontend directory missing: ${FRONTEND_DIR}"
        exit 2
    }
    resolve_frontend_runtime || {
        log_error "Frontend Node/pnpm runtime resolution failed; no service was stopped"
        exit 2
    }

    detect_docker
    validate_postgres

    if [[ "${CHECK_ONLY}" -eq 1 ]]; then
        section "Process and port status"
        ss -ltnp 2>/dev/null |
            grep -E ':(5434|8002|3002|3012)\b' || true
        systemctl is-active dishchat-app.service 2>/dev/null || true
        log_info "Safe startup preflight: PASS"
        exit 0
    fi

    stop_legacy_service
    stop_core_services
    refresh_aws_tokens
    start_backend
    start_frontend

    if [[ "${START_OPTIONAL_SERVICES}" == "1" ]]; then
        start_optional_services
    else
        log_info "Optional services skipped"
    fi

    write_runtime_state

    section "Final status"
    curl -sS "${BACKEND_HEALTH_URL}"
    echo
    log_info "PostgreSQL: ${PG_CONTAINER}/${PG_DB} on ${PG_HOST_PORT}"
    log_info "Backend: ${BACKEND_HEALTH_URL}"
    log_info "Frontend: ${FRONTEND_NETWORK_URL}"
    log_info "Runtime state: ${RUN_STATE_FILE}"
    log_info "Safe Dish-Chat restart: PASS"
}

# Allow test harnesses to source this script and call individual functions
# without executing a restart.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
