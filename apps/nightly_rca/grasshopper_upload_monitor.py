#!/usr/bin/env python3
"""Grasshopper upload monitor — polls S3 for AWAITING_LOGS receivers and re-triggers triage.

This script reads the latest nightly RCA run state, identifies receivers that are
AWAITING_LOGS (Grasshopper upload requested but not yet landed), and polls S3 via
MCP to detect when logs arrive. Once detected, it re-runs the triage phases (5+6)
for those receivers.

Usage:
    python3 grasshopper_upload_monitor.py [--poll-interval 1800] [--max-duration 43200] [--once]

Environment:
    Reads from NIGHTLY_RCA_ENV_FILE or /home/jakebot/Jakes-agent/config/nightly_rca.env
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ─── Configuration ───────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = Path(os.environ.get(
    "NIGHTLY_RCA_ENV_FILE",
    REPO_ROOT / "config" / "nightly_rca.env"
))

DEFAULT_POLL_INTERVAL = 1800  # 30 minutes
DEFAULT_MAX_DURATION = 43200  # 12 hours
LOCK_FILE = "/tmp/nightly_rca_grasshopper_monitor.lock"

LOG_DIR = Path(os.environ.get(
    "NIGHTLY_RCA_LOG_DIR",
    "/home/jakebot/Jakes-agent/logs/nightly_rca"
))
LOG_FILE = LOG_DIR / "grasshopper_monitor.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("grasshopper_monitor")


def load_env() -> dict[str, str]:
    """Load the nightly RCA env file."""
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip()
    return env


def find_latest_run(output_dir: Path) -> Path | None:
    """Find the most recent run directory."""
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return None
    run_dirs = sorted(runs_dir.iterdir(), reverse=True)
    for d in run_dirs:
        state_file = d / "state.json"
        if state_file.exists():
            return d
    return None


def get_awaiting_receivers(state: dict) -> list[dict[str, Any]]:
    """Extract receivers that are AWAITING_LOGS from state."""
    awaiting = []
    validations = state.get("data", {}).get("profile_validations", [])
    for v in validations:
        if v.get("classification") == "AWAITING_LOGS":
            candidates = v.get("candidate_receivers", [])
            awaiting.append({
                "profile_id": v.get("profile_id"),
                "best_receiver": v.get("best_receiver", ""),
                "candidate_receivers": candidates,
                "required_logs_present": v.get("required_logs_present", False),
            })
    return awaiting


async def check_receiver_logs(receiver_id: str, mcp_url: str) -> bool:
    """Check if a receiver has logs in S3 via the MCP list_dates tool."""
    try:
        # Use the MCP streamable HTTP transport
        sys.path.insert(0, str(REPO_ROOT))
        from apps.nightly_rca.transport import McpToolClient
        
        client = McpToolClient()
        await client.connect("s3_stb_logs", mcp_url)
        result = await client.call("s3_stb_logs", "list_dates", {
            "receiver_id": receiver_id,
            "limit": 3,
        })
        await client.close("s3_stb_logs")
        
        if isinstance(result, dict):
            if result.get("ok") is False:
                return False
            # If we get dates back, logs have landed
            dates = result.get("dates", [])
            return len(dates) > 0
        return False
    except Exception as exc:
        if "No logs found" in str(exc) or "FileNotFound" in str(exc):
            return False
        log.warning("Error checking receiver %s: %s", receiver_id, exc)
        return False


async def trigger_triage_rerun(
    receiver_id: str,
    profile_id: str,
    env: dict[str, str],
) -> bool:
    """Trigger a targeted re-run of the nightly RCA pipeline for a specific receiver."""
    log.info("TRIGGER: Logs detected for %s (profile: %s) — initiating triage re-run",
             receiver_id, profile_id)
    
    venv_python = Path(env.get("NIGHTLY_RCA_VENV", "")) / "bin" / "python"
    if not venv_python.exists():
        log.error("Cannot trigger re-run: venv python not found at %s", venv_python)
        return False
    
    # Run the pipeline with targeted parameters
    import subprocess
    cmd = [
        str(venv_python), "-m", "apps.nightly_rca.run",
        "--dry-run",  # Safety: always dry-run from monitor unless commit enabled
        "--log-level", "INFO",
        "--targeted-receiver", receiver_id,
        "--targeted-profile", profile_id,
    ]
    
    # Check if commit mode is enabled
    commit = env.get("NIGHTLY_RCA_COMMIT", "false").lower() in ("true", "1", "yes", "on")
    write_auth = env.get("NIGHTLY_RCA_WRITE_AUTHORIZED", "false").lower() in ("true", "1", "yes", "on")
    if commit and write_auth:
        cmd[cmd.index("--dry-run")] = "--commit"
    
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = str(REPO_ROOT)
    env_vars["NIGHTLY_RCA_OUTPUT_DIR"] = env.get("NIGHTLY_RCA_OUTPUT_DIR", "/tmp/nightly_rca_v6")
    
    log.info("Executing: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600,
            cwd=str(REPO_ROOT), env=env_vars,
        )
        if result.returncode == 0:
            log.info("Triage re-run completed successfully for %s", receiver_id)
            return True
        else:
            log.warning("Triage re-run failed (rc=%d) for %s: %s",
                       result.returncode, receiver_id, result.stderr[-500:])
            return False
    except subprocess.TimeoutExpired:
        log.error("Triage re-run timed out for %s", receiver_id)
        return False
    except Exception as exc:
        log.error("Triage re-run error for %s: %s", receiver_id, exc)
        return False


async def monitor_loop(
    poll_interval: int,
    max_duration: int,
    once: bool,
    env: dict[str, str],
) -> None:
    """Main monitoring loop."""
    output_dir = Path(env.get("NIGHTLY_RCA_OUTPUT_DIR", "/tmp/nightly_rca_v6"))
    s3_mcp_url = env.get("NIGHTLY_RCA_S3_MCP_URL", "")
    
    if not s3_mcp_url:
        log.error("FATAL: NIGHTLY_RCA_S3_MCP_URL not set")
        return
    
    start_time = time.monotonic()
    resolved: set[str] = set()
    poll_count = 0
    
    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > max_duration:
            log.info("Max duration reached (%ds). Exiting monitor.", max_duration)
            break
        
        poll_count += 1
        log.info("─── Poll %d (elapsed: %dm) ───", poll_count, int(elapsed / 60))
        
        # Find latest run state
        run_dir = find_latest_run(output_dir)
        if not run_dir:
            log.warning("No run directory found in %s", output_dir)
            if once:
                break
            await asyncio.sleep(poll_interval)
            continue
        
        state_file = run_dir / "state.json"
        with open(state_file) as f:
            state = json.load(f)
        
        awaiting = get_awaiting_receivers(state)
        if not awaiting:
            log.info("No AWAITING_LOGS receivers found. Nothing to monitor.")
            break
        
        all_resolved = True
        for entry in awaiting:
            profile_id = entry["profile_id"]
            receivers_to_check = [entry["best_receiver"]] + entry["candidate_receivers"][:5]
            receivers_to_check = [r for r in receivers_to_check if r and r not in resolved]
            
            if not receivers_to_check:
                continue
            
            all_resolved = False
            for receiver_id in receivers_to_check:
                log.info("Checking %s for profile %s...", receiver_id, profile_id)
                has_logs = await check_receiver_logs(receiver_id, s3_mcp_url)
                
                if has_logs:
                    log.info("LOGS DETECTED for %s!", receiver_id)
                    resolved.add(receiver_id)
                    success = await trigger_triage_rerun(receiver_id, profile_id, env)
                    if success:
                        log.info("Successfully re-ran triage for %s/%s", receiver_id, profile_id)
                    else:
                        log.warning("Re-run failed for %s/%s — will retry next poll", receiver_id, profile_id)
                        resolved.discard(receiver_id)  # Retry next time
                else:
                    log.info("No logs yet for %s", receiver_id)
        
        if all_resolved:
            log.info("All AWAITING_LOGS receivers resolved. Monitor complete.")
            break
        
        if once:
            log.info("--once mode: exiting after single poll.")
            break
        
        log.info("Sleeping %ds until next poll...", poll_interval)
        await asyncio.sleep(poll_interval)
    
    # Summary
    log.info("Monitor session complete. Resolved receivers: %s", sorted(resolved) or "none")


def main():
    parser = argparse.ArgumentParser(description="Grasshopper upload monitor")
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL,
                       help="Seconds between polls (default: 1800)")
    parser.add_argument("--max-duration", type=int, default=DEFAULT_MAX_DURATION,
                       help="Max monitoring duration in seconds (default: 43200)")
    parser.add_argument("--once", action="store_true",
                       help="Run a single poll and exit")
    args = parser.parse_args()
    
    # Lock file to prevent concurrent monitors
    try:
        lock_fd = open(LOCK_FILE, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        log.error("Another grasshopper monitor is already running (lock: %s)", LOCK_FILE)
        sys.exit(0)
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    env = load_env()
    log.info("Grasshopper upload monitor starting (poll=%ds, max=%ds, once=%s)",
             args.poll_interval, args.max_duration, args.once)
    
    try:
        asyncio.run(monitor_loop(
            poll_interval=args.poll_interval,
            max_duration=args.max_duration,
            once=args.once,
            env=env,
        ))
    except KeyboardInterrupt:
        log.info("Monitor interrupted by user.")
    finally:
        lock_fd.close()
        try:
            os.unlink(LOCK_FILE)
        except OSError:
            pass


if __name__ == "__main__":
    main()
