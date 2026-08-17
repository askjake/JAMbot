"""Health-qualified MCP registry state and persistent last-known-good inventory.

D3B1 separates three previously conflated ideas:

``content``
    the semantic identity of the effective tool inventory;
``health``
    the current discovery and transport status of one family;
``lifecycle``
    a process-local refresh counter and diagnostic timestamps.

Only content identity may drive tool-profile signatures, model-facing schema
cache identity, or child-snapshot comparison.  A transient timeout or
cancellation must never silently convert a known family into an empty or absent
semantic inventory, so a validated last-known-good snapshot is persisted to disk
and reloaded on startup.

Nothing written here contains callables, connection objects, transport URLs,
auth headers, tokens, exception bodies, latency, process ids, or memory
addresses.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.agent.registry_canonical import (
    CANONICALIZATION_VERSION,
    UNAVAILABLE_SCHEMA_TOKEN,
    canonicalize,
    content_signature,
    json_schema_of,
    safe_content_signature,
)

# --------------------------------------------------------------------------
# State model
# --------------------------------------------------------------------------
FAMILY_HEALTHY = "HEALTHY"
FAMILY_DEGRADED_LAST_KNOWN_GOOD = "DEGRADED_LAST_KNOWN_GOOD"
FAMILY_UNAVAILABLE_NO_BASELINE = "UNAVAILABLE_NO_BASELINE"
FAMILY_DISABLED = "DISABLED"
FAMILY_INVALID_SCHEMA = "INVALID_SCHEMA"

FAMILY_HEALTH_STATES = frozenset({
    FAMILY_HEALTHY,
    FAMILY_DEGRADED_LAST_KNOWN_GOOD,
    FAMILY_UNAVAILABLE_NO_BASELINE,
    FAMILY_DISABLED,
    FAMILY_INVALID_SCHEMA,
})

# Health states in which a retained schema must not be executed blindly.
NON_EXECUTABLE_HEALTH_STATES = frozenset({
    FAMILY_DEGRADED_LAST_KNOWN_GOOD,
    FAMILY_UNAVAILABLE_NO_BASELINE,
    FAMILY_DISABLED,
    FAMILY_INVALID_SCHEMA,
})

SOURCE_LIVE_DISCOVERY = "LIVE_DISCOVERY"
SOURCE_PERSISTED_LKG = "PERSISTED_LKG"
SOURCE_IN_MEMORY_LKG = "IN_MEMORY_LKG"
SOURCE_NONE = "NONE"

ERROR_TIMEOUT = "TIMEOUT"
ERROR_CANCELLED = "CANCELLED"
ERROR_TRANSPORT = "TRANSPORT_ERROR"
ERROR_INVALID_SCHEMA = "INVALID_SCHEMA"
ERROR_CACHE_CORRUPT = "CACHE_CORRUPT"

SAFE_ERROR_CLASSES = frozenset({
    "",
    ERROR_TIMEOUT,
    ERROR_CANCELLED,
    ERROR_TRANSPORT,
    ERROR_INVALID_SCHEMA,
    ERROR_CACHE_CORRUPT,
})

DEGRADED_RESULT_CODE = "BLOCKED_UPSTREAM_UNAVAILABLE"

# --------------------------------------------------------------------------
# Persistent cache bounds
# --------------------------------------------------------------------------
LKG_SCHEMA_VERSION = "mcp_registry_lkg.v1"
CACHE_DIR_MODE = 0o700
CACHE_FILE_MODE = 0o600
MAX_SNAPSHOT_BYTES = 4_000_000
MAX_TOOLS_PER_FAMILY = 512
MAX_SCHEMA_CHARS = 400_000
MAX_DESCRIPTION_CHARS = 8_000
MAX_FAMILY_NAME_CHARS = 96

_SAFE_FAMILY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")

_STATE_LOCK = threading.RLock()
_FAMILY_STATE: dict[str, dict[str, Any]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_family(family: str) -> str:
    name = str(family or "").strip().lower()
    if not _SAFE_FAMILY_RE.match(name):
        return ""
    return name


# --------------------------------------------------------------------------
# Semantic tool material
# --------------------------------------------------------------------------
def tool_material(tool: Any, family: str = "") -> dict[str, Any]:
    """Project one tool to the semantic fields required to trust its schema.

    Only model-facing and gate-relevant fields are retained.  A schema that
    cannot be represented deterministically becomes an explicit token rather
    than an arbitrary object repr.
    """
    args_schema = getattr(tool, "args_schema", None)
    schema: Any = None
    if args_schema is not None:
        if isinstance(args_schema, Mapping):
            schema = dict(args_schema)
        else:
            schema = json_schema_of(args_schema)
        if schema is None:
            schema = dict(UNAVAILABLE_SCHEMA_TOKEN)
    description = str(getattr(tool, "description", "") or "")[:MAX_DESCRIPTION_CHARS]
    return {
        "name": str(getattr(tool, "name", "") or ""),
        "description": description,
        "args_schema": schema,
        "family": _safe_family(family) or str(family or ""),
    }


def family_materials(tools: Iterable[Any], family: str = "") -> list[dict[str, Any]]:
    """Return semantic materials sorted by fully qualified tool name."""
    materials = [tool_material(tool, family) for tool in tools or ()]
    materials.sort(key=lambda item: str(item.get("name") or ""))
    return materials


def materials_signature(materials: Iterable[Mapping[str, Any]]) -> str:
    """Content signature over one family inventory."""
    return content_signature(list(materials or ()))


def safe_materials_signature(materials: Iterable[Mapping[str, Any]]) -> tuple[str, str]:
    return safe_content_signature(list(materials or ()))


# --------------------------------------------------------------------------
# Persistent last-known-good cache
# --------------------------------------------------------------------------
def default_cache_dir() -> Path:
    override = os.environ.get("MCP_REGISTRY_CACHE_DIR", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "var" / "mcp_registry_cache"


def cache_dir(create: bool = True) -> Path:
    path = default_cache_dir()
    if create:
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(path, CACHE_DIR_MODE)
        except OSError:
            pass
    return path


def snapshot_path(family: str) -> Path | None:
    name = _safe_family(family)
    if not name:
        return None
    return cache_dir() / (name + ".json")


def build_snapshot(family: str, materials: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    name = _safe_family(family)
    if not name:
        return None
    items = [dict(item) for item in materials or ()]
    if len(items) > MAX_TOOLS_PER_FAMILY:
        return None
    signature, _error = safe_materials_signature(items)
    if not signature:
        return None
    return {
        "schema_version": LKG_SCHEMA_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "family": name,
        "content_signature": signature,
        "fetched_at": _now(),
        "tool_count": len(items),
        "tools": items,
    }


def write_snapshot(family: str, materials: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """Atomically persist one validated family snapshot.

    Returns ``(snapshot, error_class)``.  A snapshot that cannot be represented
    deterministically is never written, so a valid baseline is never replaced by
    an invalid one.
    """
    snapshot = build_snapshot(family, materials)
    if snapshot is None:
        return None, ERROR_INVALID_SCHEMA
    target = snapshot_path(snapshot["family"])
    if target is None:
        return None, ERROR_INVALID_SCHEMA
    try:
        body = json.dumps(
            canonicalize(snapshot),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except Exception:
        return None, ERROR_INVALID_SCHEMA
    encoded = body.encode("utf-8")
    if len(encoded) > MAX_SNAPSHOT_BYTES:
        return None, ERROR_INVALID_SCHEMA
    directory = target.parent
    handle = None
    tmp_name = ""
    try:
        descriptor, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(directory))
        handle = os.fdopen(descriptor, "wb")
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None
        os.chmod(tmp_name, CACHE_FILE_MODE)
        os.replace(tmp_name, target)
        tmp_name = ""
    except OSError:
        return None, ERROR_TRANSPORT
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        if tmp_name and os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    return snapshot, ""


def load_snapshot(family: str) -> tuple[dict[str, Any] | None, str]:
    """Load and fully validate one persisted snapshot.

    A malformed, oversized, wrong-version, or signature-mismatched snapshot is
    ignored safely and reported as CACHE_CORRUPT.  A missing snapshot is not an
    error condition.
    """
    target = snapshot_path(family)
    if target is None:
        return None, ERROR_CACHE_CORRUPT
    if not target.exists():
        return None, ""
    try:
        if target.stat().st_size > MAX_SNAPSHOT_BYTES:
            return None, ERROR_CACHE_CORRUPT
        raw = target.read_text(encoding="utf-8")
    except OSError:
        return None, ERROR_CACHE_CORRUPT
    try:
        payload = json.loads(raw)
    except Exception:
        return None, ERROR_CACHE_CORRUPT
    if not isinstance(payload, Mapping):
        return None, ERROR_CACHE_CORRUPT
    if payload.get("schema_version") != LKG_SCHEMA_VERSION:
        return None, ERROR_CACHE_CORRUPT
    if payload.get("canonicalization_version") != CANONICALIZATION_VERSION:
        return None, ERROR_CACHE_CORRUPT
    if _safe_family(str(payload.get("family") or "")) != _safe_family(family):
        return None, ERROR_CACHE_CORRUPT
    tools = payload.get("tools")
    if not isinstance(tools, list) or len(tools) > MAX_TOOLS_PER_FAMILY:
        return None, ERROR_CACHE_CORRUPT
    for item in tools:
        if not isinstance(item, Mapping) or not str(item.get("name") or ""):
            return None, ERROR_CACHE_CORRUPT
    recomputed, _error = safe_materials_signature([dict(item) for item in tools])
    if not recomputed or recomputed != str(payload.get("content_signature") or ""):
        return None, ERROR_CACHE_CORRUPT
    return dict(payload), ""


def load_all_snapshots(families: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
    """Load every valid persisted snapshot, skipping corrupt families safely."""
    names: list[str] = []
    if families is not None:
        names = [str(item) for item in families]
    else:
        directory = default_cache_dir()
        if directory.exists():
            names = [path.stem for path in sorted(directory.glob("*.json"))]
    loaded: dict[str, dict[str, Any]] = {}
    for name in names:
        snapshot, error = load_snapshot(name)
        if snapshot is not None and not error:
            loaded[snapshot["family"]] = snapshot
    return loaded


def cache_modes() -> dict[str, str]:
    """Report the actual on-disk modes for audit evidence."""
    directory = default_cache_dir()
    result = {"directory": "", "files": ""}
    if directory.exists():
        result["directory"] = oct(directory.stat().st_mode & 0o777)
    modes = set()
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            modes.add(oct(path.stat().st_mode & 0o777))
    result["files"] = ",".join(sorted(modes))
    return result


# --------------------------------------------------------------------------
# Effective per-family state
# --------------------------------------------------------------------------
def _blank_state(family: str) -> dict[str, Any]:
    return {
        "family": family,
        "health": FAMILY_UNAVAILABLE_NO_BASELINE,
        "source": SOURCE_NONE,
        "content_signature": "",
        "refresh_epoch": 0,
        "loaded_at": "",
        "last_success_at": "",
        "tool_count": 0,
        "error_class": "",
    }


def set_family_state(
    family: str,
    *,
    health: str,
    source: str,
    content_signature_value: str = "",
    refresh_epoch: int = 0,
    tool_count: int = 0,
    error_class: str = "",
    last_success_at: str | None = None,
) -> dict[str, Any]:
    """Record the health-qualified effective state for one family.

    ``error_class`` is restricted to a bounded safe vocabulary so no transport
    exception body can reach status output or audit records.
    """
    name = _safe_family(family) or str(family or "")[:MAX_FAMILY_NAME_CHARS]
    if health not in FAMILY_HEALTH_STATES:
        health = FAMILY_UNAVAILABLE_NO_BASELINE
    if error_class not in SAFE_ERROR_CLASSES:
        error_class = ERROR_TRANSPORT
    with _STATE_LOCK:
        state = dict(_FAMILY_STATE.get(name) or _blank_state(name))
        state.update({
            "family": name,
            "health": health,
            "source": source,
            "content_signature": str(content_signature_value or ""),
            "refresh_epoch": int(refresh_epoch),
            "loaded_at": _now(),
            "tool_count": int(tool_count),
            "error_class": error_class,
        })
        if last_success_at is not None:
            state["last_success_at"] = str(last_success_at)
        elif health == FAMILY_HEALTHY:
            state["last_success_at"] = state["loaded_at"]
        _FAMILY_STATE[name] = state
        return dict(state)


def get_family_state(family: str) -> dict[str, Any]:
    name = _safe_family(family) or str(family or "")
    with _STATE_LOCK:
        state = _FAMILY_STATE.get(name)
        return dict(state) if state else {}


def family_health(family: str) -> str:
    return str(get_family_state(family).get("health") or "")


def family_content_signature(family: str) -> str:
    return str(get_family_state(family).get("content_signature") or "")


def known_families() -> list[str]:
    with _STATE_LOCK:
        return sorted(_FAMILY_STATE)


def effective_registry_model() -> dict[str, dict[str, Any]]:
    """Return the per-family effective model without any raw error text."""
    with _STATE_LOCK:
        return {name: dict(state) for name, state in sorted(_FAMILY_STATE.items())}


def registry_health_signature() -> str:
    """Deterministic summary of current family health states.

    This value must never drive a tool-profile signature.  It exists so health
    transitions are observable and auditable on their own terms.
    """
    material = [
        {
            "family": name,
            "health": state.get("health"),
            "source": state.get("source"),
            "error_class": state.get("error_class"),
        }
        for name, state in sorted(effective_registry_model().items())
    ]
    signature, _error = safe_content_signature(material)
    return signature


def reset_family_states() -> None:
    """Test-only helper; production code never clears health state."""
    with _STATE_LOCK:
        _FAMILY_STATE.clear()


# --------------------------------------------------------------------------
# Execution health gate input
# --------------------------------------------------------------------------
def family_execution_block(family: str) -> tuple[bool, dict[str, Any]]:
    """Return ``(blocked, detail)`` for the current health of one family.

    A family with no recorded health state is not blocked: local, non-MCP tool
    families are outside this state model and must keep working normally.
    """
    state = get_family_state(family)
    if not state:
        return False, {}
    health = str(state.get("health") or "")
    detail = {
        "upstream_family": state.get("family", ""),
        "upstream_health": health,
        "upstream_source": state.get("source", ""),
        "upstream_error_class": state.get("error_class", ""),
    }
    return health in NON_EXECUTABLE_HEALTH_STATES, detail
