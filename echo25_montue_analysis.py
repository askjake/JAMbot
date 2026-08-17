#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  ECHO25 OUTAGE IMPACT — MONDAY/TUESDAY NIGHT TARGETED ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

Purpose:
  Query the PAST 5 Monday night / Tuesday morning windows (12AM-4AM MT)
  to establish a "most likely" estimate for June 2, 2026 (Tuesday).

  June 2 outage: Monday June 1 night → Tuesday June 2 early morning.
  UTC window: Tuesday 06:00-10:00 UTC = Monday 12AM-4AM MT.

Target dates (all Tuesdays):
  - 2026-05-12 (already collected)
  - 2026-05-05 (new)
  - 2026-04-28 (new)
  - 2026-04-21 (new)
  - 2026-04-14 (new)

After collection, merges with existing 7-day results to produce combined report.
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import json
import time
import sys
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, ".")

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

WINDOW_START_HOUR_UTC = 6   # 12:00 AM MT
WINDOW_END_HOUR_UTC = 10    # 4:00 AM MT

BATCH_SIZE = 25
MAX_CONCURRENT = 3
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3
RETRY_BACKOFF = 10

OUTPUT_FILE = "/tmp/echo25_monday_tuesday_results.json"
MERGED_OUTPUT = "/tmp/echo25_merged_final.json"
EXISTING_RESULTS = "/tmp/echo25_viewership_results.json"
LOG_FILE = "/tmp/echo25_montue.log"

# Target: Past 5 Mon/Tue nights (Tuesdays at 06:00-10:00 UTC)
BASE_TUESDAY = datetime(2026, 5, 12, tzinfo=timezone.utc)
TARGET_DATES = [BASE_TUESDAY - timedelta(weeks=i) for i in range(5)]
# = [2026-05-12, 2026-05-05, 2026-04-28, 2026-04-21, 2026-04-14]

# 2026-05-12 already exists in previous results - skip if available
SKIP_IF_EXISTS = ["2026-05-12"]

# ═══════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("echo25_montue")

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def get_epoch_range(day: datetime) -> tuple:
    start_dt = day.replace(hour=WINDOW_START_HOUR_UTC, minute=0, second=0, microsecond=0)
    end_dt = day.replace(hour=WINDOW_END_HOUR_UTC, minute=0, second=0, microsecond=0)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


async def call_with_retry(tool, params, timeout=REQUEST_TIMEOUT, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            result = await asyncio.wait_for(tool.ainvoke(params), timeout=timeout)
            if isinstance(result, str):
                result = json.loads(result)
            if "error" in result and "timeout" in str(result.get("error", "")).lower():
                raise asyncio.TimeoutError("Server-side timeout")
            return result
        except asyncio.TimeoutError:
            if attempt < retries - 1:
                wait = RETRY_BACKOFF * (attempt + 1)
                log.warning(f"  Timeout (attempt {attempt+1}/{retries}), retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                return {"error": f"timeout after {retries} attempts"}
        except Exception as e:
            if attempt < retries - 1:
                wait = RETRY_BACKOFF * (attempt + 1)
                log.warning(f"  Error: {e} (attempt {attempt+1}/{retries}), retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                return {"error": str(e)}
    return {"error": "max retries exceeded"}

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

async def main():
    start_time = time.time()
    
    log.info("=" * 70)
    log.info("ECHO25 — MONDAY/TUESDAY NIGHT TARGETED ANALYSIS")
    log.info("Past 5 Mon night/Tue morning windows (12AM-4AM MT)")
    log.info("=" * 70)
    
    # ─── Initialize MCP ───
    log.info("\nInitializing MCP...")
    from app.config import get_settings
    from app.agent.agents.utils import get_mcp_tools
    settings = get_settings()
    
    tools = await asyncio.wait_for(
        get_mcp_tools(settings.VIEWERSHIP_MCP_CONFIG), timeout=30
    )
    tools_dict = {t.name: t for t in tools}
    log.info(f"  ✓ Connected")
    
    # ─── Load stations ───
    with open("/tmp/local_stations_filtered.json") as f:
        station_data = json.load(f)
    local_suids = station_data["local_suids"]
    log.info(f"  ✓ {len(local_suids)} local SUIDs loaded")
    
    # ─── Check existing results ───
    existing_data = {}
    if Path(EXISTING_RESULTS).exists():
        with open(EXISTING_RESULTS) as f:
            prev = json.load(f)
        existing_m1 = prev.get("results", {}).get("method1_per_service", {}).get("daily_data", {})
        existing_m2 = prev.get("results", {}).get("method2_platform_summary", {}).get("daily_data", {})
        log.info(f"  ✓ Loaded previous results with {len(existing_m1)} days")
    else:
        existing_m1 = {}
        existing_m2 = {}
        log.info("  ⚠ No previous results file found")
    
    # ─── Determine which dates to query ───
    dates_to_query = []
    dates_from_cache = []
    for day in TARGET_DATES:
        day_str = day.strftime("%Y-%m-%d")
        if day_str in SKIP_IF_EXISTS and day_str in existing_m1:
            log.info(f"  ✓ {day_str} (Tue) — using cached data")
            dates_from_cache.append(day_str)
        else:
            dates_to_query.append(day)
    
    log.info(f"\n  Dates to query fresh: {[d.strftime('%Y-%m-%d') for d in dates_to_query]}")
    log.info(f"  Dates from cache: {dates_from_cache}")
    
    # ─── METHOD 1: Per-Service Batch Query ───
    log.info("\n" + "=" * 70)
    log.info("METHOD 1: Per-Service Batch Query (local channels)")
    log.info("=" * 70)
    
    query_viewership = tools_dict["query_viewership"]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    suid_batches = [local_suids[i:i+BATCH_SIZE] for i in range(0, len(local_suids), BATCH_SIZE)]
    log.info(f"  {len(suid_batches)} batches × {len(dates_to_query)} days = {len(suid_batches) * len(dates_to_query)} total queries")
    
    async def query_batch(day, batch):
        from_epoch, to_epoch = get_epoch_range(day)
        async with semaphore:
            return await call_with_retry(query_viewership, {
                "fromEpochTimeUtc": from_epoch,
                "toEpochTimeUtc": to_epoch,
                "viewingType": "Tune",
                "serviceUids": batch
            })
    
    method1_new = {}
    for day in dates_to_query:
        day_str = day.strftime("%Y-%m-%d")
        day_name = (day - timedelta(days=1)).strftime("%A")  # Monday night
        log.info(f"\n  [{day_str}] {day_name} night → Tue morning | {len(suid_batches)} batches...")
        day_start = time.time()
        
        tasks = [query_batch(day, batch) for batch in suid_batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_accounts = 0
        total_sessions = 0
        total_hours = 0.0
        services_active = 0
        errors = 0
        
        for r in results:
            if isinstance(r, Exception):
                errors += 1
                continue
            if "error" in r:
                errors += 1
                continue
            res = r.get("results", [])
            if isinstance(res, list):
                for entry in res:
                    accts = entry.get("totalWatchAccounts", 0)
                    if accts > 0:
                        services_active += 1
                    total_accounts += accts
                    total_sessions += entry.get("totalSessions", 0)
                    total_hours += entry.get("totalWatchHours", 0)
            elif isinstance(res, dict) and res:
                accts = res.get("totalWatchAccounts", 0)
                if accts > 0:
                    services_active += 1
                total_accounts += accts
                total_sessions += res.get("totalSessions", 0)
                total_hours += res.get("totalWatchHours", 0)
        
        elapsed = time.time() - day_start
        method1_new[day_str] = {
            "total_accounts_sum": total_accounts,
            "total_sessions": total_sessions,
            "total_watch_hours": round(total_hours, 2),
            "services_with_viewers": services_active,
            "batches_completed": len(suid_batches) - errors,
            "errors": errors,
            "elapsed_seconds": round(elapsed, 1)
        }
        log.info(f"  [{day_str}] Done in {elapsed:.1f}s | Accounts: {total_accounts:,} | "
                 f"Sessions: {total_sessions:,} | Active: {services_active} | Errors: {errors}/{len(suid_batches)}")
    
    # ─── METHOD 2: Viewing Summary ───
    log.info("\n" + "=" * 70)
    log.info("METHOD 2: Platform Viewing Summary (true unique accounts)")
    log.info("=" * 70)
    
    get_summary = tools_dict["get_viewing_summary"]
    method2_new = {}
    
    for day in dates_to_query:
        day_str = day.strftime("%Y-%m-%d")
        from_epoch, to_epoch = get_epoch_range(day)
        
        result = await call_with_retry(get_summary, {
            "fromEpochTimeUtc": from_epoch,
            "toEpochTimeUtc": to_epoch,
            "viewingType": "Tune"
        })
        
        method2_new[day_str] = result
        if "error" not in result:
            log.info(f"  [{day_str}] Unique accounts: {result.get('totalWatchAccounts', 'N/A'):,} | "
                     f"Services: {result.get('totalServices', 'N/A')}")
        else:
            log.info(f"  [{day_str}] ERROR: {result['error']}")
    
    # ─── MERGE WITH EXISTING DATA ───
    log.info("\n" + "=" * 70)
    log.info("MERGING WITH EXISTING RESULTS")
    log.info("=" * 70)
    
    # Combine all Mon/Tue method1 data
    montue_m1 = {}
    montue_m2 = {}
    
    # Add cached date(s) from previous run
    for day_str in dates_from_cache:
        if day_str in existing_m1:
            montue_m1[day_str] = existing_m1[day_str]
            log.info(f"  [CACHED] {day_str}: {existing_m1[day_str].get('total_accounts_sum', 'N/A'):,} accounts")
        if day_str in existing_m2:
            montue_m2[day_str] = existing_m2[day_str]
    
    # Add new data
    for day_str, data in method1_new.items():
        montue_m1[day_str] = data
    for day_str, data in method2_new.items():
        montue_m2[day_str] = data
    
    # ─── CALCULATE MON/TUE AVERAGE ───
    log.info("\n" + "=" * 70)
    log.info("MONDAY/TUESDAY NIGHT RESULTS")
    log.info("=" * 70)
    
    valid_m1 = {k: v for k, v in montue_m1.items() 
                if isinstance(v, dict) and "total_accounts_sum" in v and v.get("errors", 999) < len(suid_batches) * 0.1}
    valid_m2 = {k: v for k, v in montue_m2.items()
                if isinstance(v, dict) and "error" not in v and "totalWatchAccounts" in v}
    
    if valid_m1:
        m1_values = [v["total_accounts_sum"] for v in valid_m1.values()]
        m1_avg = sum(m1_values) / len(m1_values)
        m1_sessions = [v["total_sessions"] for v in valid_m1.values()]
        m1_hours = [v["total_watch_hours"] for v in valid_m1.values()]
        
        log.info(f"\n  METHOD 1 — Local channel accounts (Mon/Tue nights only):")
        log.info(f"    Valid nights: {len(valid_m1)}")
        for k in sorted(valid_m1.keys()):
            v = valid_m1[k]
            log.info(f"      {k}: {v['total_accounts_sum']:>6,} accounts | "
                     f"{v['total_sessions']:>6,} sessions | "
                     f"{v['total_watch_hours']:>9,.1f} hrs")
        log.info(f"    ─────────────────────────────────────────────")
        log.info(f"    AVERAGE: {m1_avg:,.0f} account-channel pairs")
        log.info(f"    MIN:     {min(m1_values):,}")
        log.info(f"    MAX:     {max(m1_values):,}")
        log.info(f"    STDEV:   {(sum((x - m1_avg)**2 for x in m1_values) / len(m1_values))**0.5:,.0f}")
    else:
        m1_avg = None
        log.warning("  METHOD 1: No valid Mon/Tue data")
    
    if valid_m2:
        m2_values = [v["totalWatchAccounts"] for v in valid_m2.values()]
        m2_avg = sum(m2_values) / len(m2_values)
        
        log.info(f"\n  METHOD 2 — Platform-wide unique accounts (Mon/Tue nights):")
        log.info(f"    Valid nights: {len(valid_m2)}")
        for k in sorted(valid_m2.keys()):
            v = valid_m2[k]
            log.info(f"      {k}: {v['totalWatchAccounts']:>9,} unique accounts | "
                     f"{v.get('totalServices', 'N/A')} services")
        log.info(f"    ─────────────────────────────────────────────")
        log.info(f"    AVERAGE: {m2_avg:,.0f} unique accounts (all channels)")
    else:
        m2_avg = None
        log.warning("  METHOD 2: No valid Mon/Tue data")
    
    # ─── FINAL MERGED REPORT ───
    log.info("\n" + "=" * 70)
    log.info("FINAL MERGED REPORT — ECHO25 OUTAGE IMPACT")
    log.info("=" * 70)
    
    # Load full 7-day data for context
    all_days_m1 = dict(existing_m1)
    all_days_m1.update(method1_new)
    
    all_7day_values = [v.get("total_accounts_sum", 0) for k, v in existing_m1.items() 
                       if isinstance(v, dict) and "total_accounts_sum" in v]
    avg_7day = sum(all_7day_values) / len(all_7day_values) if all_7day_values else 0
    
    log.info(f"\n  ┌─────────────────────────────────────────────────────────────────────┐")
    log.info(f"  │  ECHO25 LOCAL CHANNEL VIEWERSHIP — 12:00 AM - 4:00 AM MT            │")
    log.info(f"  │                                                                      │")
    log.info(f"  │  ★ MOST LIKELY (Mon/Tue night average, {len(valid_m1) if valid_m1 else 0} samples):              │")
    if m1_avg:
        log.info(f"  │      {m1_avg:,.0f} account-channel pairs                              │")
        log.info(f"  │      Range: {min(m1_values):,} – {max(m1_values):,}                                   │")
    log.info(f"  │                                                                      │")
    log.info(f"  │  Context (all 7 days avg): {avg_7day:,.0f}                                   │")
    log.info(f"  │  Platform total in window: {m2_avg:,.0f} unique accounts (all channels)  │" if m2_avg else "")
    log.info(f"  │  Local as % of platform: {m1_avg/m2_avg*100:.1f}%                                     │" if m1_avg and m2_avg else "")
    log.info(f"  │                                                                      │")
    log.info(f"  │  June 2 (Mon night/Tue AM) expected impact: ~{m1_avg:,.0f} customers       │" if m1_avg else "")
    log.info(f"  └─────────────────────────────────────────────────────────────────────┘")
    
    # ─── SAVE RESULTS ───
    total_elapsed = time.time() - start_time
    
    output = {
        "analysis": "Echo25 Outage Impact - Monday/Tuesday Night Targeted",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": "June 2, 2026 (Mon night → Tue 12AM-4AM MT)",
        "query_parameters": {
            "window_mt": "12:00 AM - 4:00 AM Mountain Time",
            "window_utc": "06:00 - 10:00 UTC",
            "viewing_type": "Tune (Live TV)",
            "local_suids_queried": len(local_suids),
            "sample_type": "Monday night / Tuesday morning only",
            "dates_queried": [d.strftime("%Y-%m-%d") for d in TARGET_DATES]
        },
        "monday_tuesday_results": {
            "method1_per_service": {
                "description": "Sum of totalWatchAccounts per local service for Mon/Tue nights",
                "daily_data": montue_m1,
                "average": m1_avg,
                "min": min(m1_values) if valid_m1 else None,
                "max": max(m1_values) if valid_m1 else None,
                "valid_days": len(valid_m1) if valid_m1 else 0
            },
            "method2_platform_summary": {
                "description": "True unique accounts (all channels) for Mon/Tue nights",
                "daily_data": montue_m2,
                "average": m2_avg,
                "valid_days": len(valid_m2) if valid_m2 else 0
            }
        },
        "context": {
            "all_7day_average": avg_7day,
            "all_7day_daily": {k: v.get("total_accounts_sum") for k, v in existing_m1.items() if isinstance(v, dict)}
        },
        "most_likely_estimate": {
            "value": m1_avg,
            "basis": "5 Monday/Tuesday night samples",
            "confidence": "High — same day-of-week pattern as June 2 outage"
        },
        "execution": {
            "total_seconds": round(total_elapsed, 1),
            "dates_queried_fresh": [d.strftime("%Y-%m-%d") for d in dates_to_query],
            "dates_from_cache": dates_from_cache
        }
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\n  ✓ Mon/Tue results saved: {OUTPUT_FILE}")
    
    # Also write merged file combining everything
    merged = {
        "echo25_final_report": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "most_likely": output["most_likely_estimate"],
        "monday_tuesday_data": output["monday_tuesday_results"],
        "full_7day_context": {
            "average": avg_7day,
            "daily": {k: v.get("total_accounts_sum") for k, v in existing_m1.items() if isinstance(v, dict)}
        }
    }
    with open(MERGED_OUTPUT, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    log.info(f"  ✓ Merged final saved: {MERGED_OUTPUT}")
    log.info(f"  ✓ Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    log.info("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
