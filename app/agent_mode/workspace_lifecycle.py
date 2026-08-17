import os, shutil, logging
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
BASE = os.environ.get("AGENT_MODE_WORKDIR", "/tmp/home_agent")
TTL_HOURS = int(os.environ.get("WORKSPACE_TTL_HOURS", "168"))  # 7 days default
MAX_MB = float(os.environ.get("WORKSPACE_MAX_SIZE_MB", "500"))
# Grace period: workspace must be stale (no real file activity) for this long
# before cleanup is eligible. Set to 0 to disable activity checks.
STALE_HOURS = int(os.environ.get("WORKSPACE_STALE_HOURS", "168"))  # 7 days

# Metadata files excluded from activity scanning (not user-generated content)
_METADATA_FILES = {".last_access"}


def touch_workspace(chat_id: str):
    ws = Path(BASE) / chat_id
    ws.mkdir(parents=True, exist_ok=True)
    (ws / ".last_access").write_text(datetime.now().isoformat())
    return ws


def get_size_mb(path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1024 / 1024


def _latest_file_activity(ws: Path) -> datetime:
    """Scan workspace for the most recent file modification time.

    This provides an activity-aware check: even if .last_access was not
    explicitly updated, any file written (code output, logs, artifacts)
    counts as activity and prevents premature cleanup.

    Excludes internal metadata files (.last_access) from the scan.
    """
    latest = datetime.min
    try:
        for f in ws.rglob("*"):
            if not f.is_file():
                continue
            if f.name in _METADATA_FILES:
                continue
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime > latest:
                    latest = mtime
            except (OSError, ValueError):
                continue
    except (OSError, PermissionError):
        pass
    return latest


def _is_workspace_stale(ws: Path, cutoff: datetime) -> bool:
    """Determine if a workspace is truly stale (no activity past cutoff).

    Checks both the explicit .last_access marker AND actual file modification
    times within the workspace. The workspace is only considered stale if BOTH
    the marker and real file activity are older than the cutoff.
    """
    # Check explicit access marker
    af = ws / ".last_access"
    if af.exists():
        try:
            marker_time = datetime.fromisoformat(af.read_text().strip())
            if marker_time >= cutoff:
                return False
        except (ValueError, OSError):
            pass

    # Check actual file activity (the real staleness signal)
    latest_activity = _latest_file_activity(ws)
    if latest_activity >= cutoff:
        return False

    return True


async def cleanup_expired():
    """Remove workspaces that have been stale for longer than the threshold.

    A workspace is only removed if:
    1. Its .last_access marker is older than STALE_HOURS, AND
    2. No user file within the workspace has been modified within STALE_HOURS.

    This prevents premature cleanup of workspaces that are actively being
    used but whose .last_access marker wasn't explicitly refreshed.
    """
    base = Path(BASE)
    if not base.exists():
        return 0
    cutoff = datetime.now() - timedelta(hours=STALE_HOURS)
    cleaned = 0
    for ws in base.iterdir():
        if not ws.is_dir():
            continue
        if _is_workspace_stale(ws, cutoff):
            logger.info(
                f"Cleaned workspace {ws.name} (no activity in {STALE_HOURS}h)"
            )
            shutil.rmtree(ws, ignore_errors=True)
            cleaned += 1
        else:
            # Log workspaces approaching threshold for observability
            af = ws / ".last_access"
            if af.exists():
                try:
                    marker_time = datetime.fromisoformat(af.read_text().strip())
                    age_h = (datetime.now() - marker_time).total_seconds() / 3600
                    if age_h > (STALE_HOURS * 0.75):
                        logger.debug(
                            f"Workspace {ws.name} approaching stale threshold "
                            f"(marker age: {age_h:.1f}h / {STALE_HOURS}h)"
                        )
                except (ValueError, OSError):
                    pass
    return cleaned


def get_stats() -> dict:
    base = Path(BASE)
    if not base.exists():
        return {}
    rows = []
    for ws in base.iterdir():
        if not ws.is_dir():
            continue
        latest_activity = _latest_file_activity(ws)
        af = ws / ".last_access"
        if af.exists():
            try:
                marker_time = datetime.fromisoformat(af.read_text().strip())
            except (ValueError, OSError):
                marker_time = datetime.fromtimestamp(ws.stat().st_mtime)
        else:
            marker_time = datetime.fromtimestamp(ws.stat().st_mtime)

        rows.append({
            "id": ws.name,
            "size_mb": round(get_size_mb(ws), 2),
            "marker_age_h": round(
                (datetime.now() - marker_time).total_seconds() / 3600, 1
            ),
            "last_file_activity_h": round(
                (datetime.now() - latest_activity).total_seconds() / 3600, 1
            ) if latest_activity != datetime.min else None,
        })
    rows.sort(key=lambda x: x["size_mb"], reverse=True)
    return {
        "total": len(rows),
        "total_mb": round(sum(r["size_mb"] for r in rows), 2),
        "stale_hours": STALE_HOURS,
        "ttl_hours": TTL_HOURS,
        "top": rows[:10],
    }
