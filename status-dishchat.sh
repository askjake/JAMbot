#!/usr/bin/env bash
set -Eeuo pipefail

INSTALL_DIR="/home/jakebot/Jakes-agent"
BACKEND_PY="${INSTALL_DIR}/.venv/bin/python"
PG_CONTAINER="dishchat-postgres"
PG_USER="dev_user"
PG_DB="dishchat"

if docker info >/dev/null 2>&1; then
    D=(docker)
elif sudo -n docker info >/dev/null 2>&1; then
    D=(sudo docker)
else
    echo "Docker is unavailable" >&2
    exit 3
fi

echo "=== Docker/PostgreSQL ==="
"${D[@]}" ps -a --filter "name=^${PG_CONTAINER}$" \
    --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

if "${D[@]}" exec "${PG_CONTAINER}" \
    pg_isready -U "${PG_USER}" -d "${PG_DB}" >/dev/null 2>&1; then
    echo "PostgreSQL readiness: PASS"
    "${D[@]}" exec "${PG_CONTAINER}" psql -X --no-psqlrc \
        -U "${PG_USER}" -d postgres -P pager=off -c "
SELECT datname,datallowconn,pg_size_pretty(pg_database_size(oid)) AS size
FROM pg_database
WHERE datname='dishchat'
   OR datname LIKE 'dishchat_old_history_%'
ORDER BY datname;"
else
    echo "PostgreSQL readiness: FAIL"
fi

echo
echo "=== LangGraph migration ledger ==="
if "${D[@]}" exec "${PG_CONTAINER}" \
    pg_isready -U "${PG_USER}" -d "${PG_DB}" >/dev/null 2>&1; then
    installed_count="$(
        cd "${INSTALL_DIR}"
        PYTHONWARNINGS=ignore "${BACKEND_PY}" - <<'PY' 2>/dev/null
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
print(len(AsyncPostgresSaver.MIGRATIONS))
PY
    )"
    "${D[@]}" exec "${PG_CONTAINER}" psql -X --no-psqlrc \
        -U "${PG_USER}" -d "${PG_DB}" -P pager=off -c "
SELECT count(*) AS rows,min(v),max(v)
FROM public.checkpoint_migrations;
TABLE public.checkpoint_migrations;"
    echo "Installed migration count: ${installed_count}"
fi

echo
echo "=== Listeners ==="
ss -ltnp 2>/dev/null |
    grep -E ':(5434|8002|3002|3012|5001|5006|8087|18443)\b' || true

echo
echo "=== Application processes ==="
pgrep -af \
    'uvicorn app.main:app|next (dev|start)|next-server|coverity_assist_gateway|qodo-ssh-proxy|dish-code-tools' \
    || true

echo
echo "=== Backend health ==="
curl -sS --connect-timeout 2 --max-time 5 \
    http://127.0.0.1:8002/rest/api/v1/health || true
echo

echo
echo "=== Frontend health ==="
curl -sS --connect-timeout 2 --max-time 5 \
    -o /dev/null -w 'HTTP %{http_code}\n' \
    http://127.0.0.1:3002/ || true

echo
echo "=== Legacy service ==="
systemctl is-enabled dishchat-app.service 2>/dev/null || true
systemctl is-active dishchat-app.service 2>/dev/null || true
