#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_PARENT="$(dirname "$PACKAGE_DIR")"
REPO_ROOT="$(cd "$PACKAGE_DIR/../.." && pwd)"
PYTHON="${NIGHTLY_RCA_VENV:-${VIRTUAL_ENV:-}}"
if [[ -n "$PYTHON" && -x "$PYTHON/bin/python" ]]; then
  PYTHON="$PYTHON/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

export PYTHONPATH="$PACKAGE_PARENT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m compileall -q "$PACKAGE_DIR"
bash -n "$PACKAGE_DIR/run_nightly.sh"
"$PYTHON" -m pytest --rootdir="$PACKAGE_DIR" -q "$PACKAGE_DIR/tests"

"$PYTHON" - "$REPO_ROOT" <<'PY'
from pathlib import Path
import re
import sys
root = Path(sys.argv[1])
skip = {".git", ".venv", "__pycache__", "node_modules", "var", "logs"}
allowed_suffixes = {".py", ".sh", ".env", ".example", ".toml", ".yaml", ".yml", ".json", ".md", ".txt"}
parts = []
for path in root.rglob("*"):
    try:
        is_file = path.is_file()
    except PermissionError:
        continue
    if not is_file or any(part in skip for part in path.parts):
        continue
    if path.suffix.lower() not in allowed_suffixes and path.name != "nightly_rca.env":
        continue
    try:
        parts.append(path.read_text(errors="ignore"))
    except PermissionError:
        continue
text = "\n".join(parts)
checks = {
    "Google API key": r"AIza[0-9A-Za-z_-]{30,}",
    "Chat webhook path": r"chat\.googleapis\.com/v1/spaces/[A-Za-z0-9_-]{8,}",
    "JIRA token marker": "AT" + "ATT",
}
found = [name for name, pattern in checks.items() if re.search(pattern, text)]
if found:
    raise SystemExit("credential marker(s) found: " + ", ".join(found))
print("repository credential scan: PASS")
PY

echo "nightly_rca verification: PASS"
