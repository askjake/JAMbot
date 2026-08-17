#!/usr/bin/env bash
# Safe Dish-Chat stop.
#
# This script never stops PostgreSQL, removes a container, invokes Docker
# Compose, deletes a volume, or alters a database.

set -Eeuo pipefail
umask 077

INSTALL_DIR="/home/jakebot/Jakes-agent"
FRONTEND_DIR="/home/jakebot/Jakes-agent-fe/apps/chats"
BACKEND_PID_FILE="${INSTALL_DIR}/backend.pid"
FRONTEND_PID_FILE="${FRONTEND_DIR}/frontend.pid"
BACKEND_PORT=8002
FRONTEND_PORT=3002
STOP_OPTIONAL=0

usage() {
    cat <<'EOF'
Usage:
  stop-dishchat.sh [--all]

Default:
  Stop only the exact backend on 8002 and verified frontend on 3002.

--all:
  Also stop known optional helper processes by their PID files.
  PostgreSQL is never stopped.
EOF
}

while (($#)); do
    case "$1" in
        --all) STOP_OPTIONAL=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 64 ;;
    esac
    shift
done

listener_pids() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | sort -nu || true
    else
        ss -ltnp 2>/dev/null |
            awk -v needle=":${port}" '
                index($0,needle) {
                    s=$0
                    while(match(s,/pid=[0-9]+/)) {
                        print substr(s,RSTART+4,RLENGTH-4)
                        s=substr(s,RSTART+RLENGTH)
                    }
                }' | sort -nu
    fi
}

pid_cmd(){ ps -p "$1" -o args= 2>/dev/null || true; }
pid_cwd(){ readlink -f "/proc/$1/cwd" 2>/dev/null || true; }
pid_group(){ ps -p "$1" -o pgid= 2>/dev/null | tr -d ' ' || true; }

stop_group() {
    local pid="$1" label="$2" pgid current
    kill -0 "$pid" 2>/dev/null || return 0
    pgid="$(pid_group "$pid")"
    current="$(pid_group "$$")"
    echo "[INFO] Stopping ${label} PID ${pid}"
    if [[ -n "$pgid" && "$pgid" != "$current" ]]; then
        kill -TERM -- "-$pgid" 2>/dev/null || true
    else
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill -TERM "$pid" 2>/dev/null || true
    fi
    sleep 3
    kill -0 "$pid" 2>/dev/null || return 0
    if [[ -n "$pgid" && "$pgid" != "$current" ]]; then
        kill -KILL -- "-$pgid" 2>/dev/null || true
    else
        pkill -KILL -P "$pid" 2>/dev/null || true
        kill -KILL "$pid" 2>/dev/null || true
    fi
}

backend_owned() {
    local pid="$1" cmd cwd
    cmd="$(pid_cmd "$pid")"; cwd="$(pid_cwd "$pid")"
    [[ "$cwd" == "$INSTALL_DIR" &&
       "$cmd" == *"uvicorn app.main:app"* &&
       "$cmd" == *"--port ${BACKEND_PORT}"* ]]
}

frontend_owned() {
    local pid="$1" cmd cwd
    cmd="$(pid_cmd "$pid")"; cwd="$(pid_cwd "$pid")"
    [[ "$cwd" == "$FRONTEND_DIR" &&
       ( "$cmd" == *"next dev"* ||
         "$cmd" == *"next start"* ||
         "$cmd" == *"next-server"* ) ]]
}

stop_pid_file() {
    local file="$1" label="$2" matcher="$3" pid
    [[ -f "$file" ]] || return 0
    pid="$(tr -dc '0-9' < "$file")"
    if [[ -n "$pid" && -d "/proc/$pid" ]] && "$matcher" "$pid"; then
        stop_group "$pid" "$label"
    elif [[ -n "$pid" && -d "/proc/$pid" ]]; then
        echo "[WARN] Refusing unowned PID ${pid} from ${file}"
    fi
    rm -f "$file"
}

stop_port_owned() {
    local port="$1" label="$2" matcher="$3" pids=() pid
    mapfile -t pids < <(listener_pids "$port")
    for pid in "${pids[@]}"; do
        if "$matcher" "$pid"; then
            stop_group "$pid" "$label"
        else
            echo "[ERROR] Refusing unrelated PID ${pid} on port ${port}" >&2
            echo "  cwd=$(pid_cwd "$pid")" >&2
            echo "  cmd=$(pid_cmd "$pid")" >&2
            exit 1
        fi
    done
}

stop_pid_file "$BACKEND_PID_FILE" backend backend_owned
stop_pid_file "$FRONTEND_PID_FILE" frontend frontend_owned
stop_port_owned "$BACKEND_PORT" backend backend_owned
stop_port_owned "$FRONTEND_PORT" frontend frontend_owned

if [[ "$STOP_OPTIONAL" -eq 1 ]]; then
    for file in \
        "${INSTALL_DIR}/gateway.pid" \
        "${INSTALL_DIR}/qodo-portforward.pid" \
        "/home/montjac/dish-code-tools/dish-code-tools.pid"; do
        [[ -f "$file" ]] || continue
        pid="$(tr -dc '0-9' < "$file")"
        [[ -n "$pid" && -d "/proc/$pid" ]] && stop_group "$pid" "optional helper" || true
        rm -f "$file"
    done
fi

echo "[INFO] Backend/frontend stopped."
echo "[INFO] PostgreSQL dishchat-postgres was left running and unchanged."
