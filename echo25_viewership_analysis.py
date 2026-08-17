#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  ECHO25 OUTAGE IMPACT ANALYSIS
  Local Channel Viewership: 12:00 AM - 4:00 AM Mountain Time
═══════════════════════════════════════════════════════════════════════════════

Purpose:
  Determine the average count of customers viewing local channels between 
  12:00 AM (MT) and 4:00 AM (MT) to assess Echo25 satellite transition impact 
  on June 2nd.

Methodology:
  - Query window: 12:00 AM - 4:00 AM MT = 06:00 - 10:00 UTC
  - Viewing type: "Tune" (Live satellite TV)
  - Stations: 3,218 FCC-licensed local broadcast SUIDs (K/W call signs)
  - Sample: Last 7 days of data to establish reliable average
  - Three independent measurement methods for cross-validation

Time Conversion:
  Mountain Time (MT) = UTC - 6 hours
  12:00 AM MT = 06:00 UTC
  4:00 AM MT  = 10:00 UTC

Run: .venv/bin/python echo25_viewership_analysis.py
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import json
import time
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, ".")

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Time window (UTC equivalent of 12AM-4AM MT)
WINDOW_START_HOUR_UTC = 6
WINDOW_END_HOUR_UTC = 10

# Number of days to sample for average
SAMPLE_DAYS = 7

# Batch size for serviceUids per API call (conservative to avoid timeouts)
BATCH_SIZE = 25

# Max concurrent API requests (throttled to prevent Lambda saturation)
MAX_CONCURRENT = 3

# Per-request timeout (Lambda max is typically 900s, Athena can be slow)
REQUEST_TIMEOUT = 120

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF = 10  # seconds between retries

# Output file
OUTPUT_FILE = "/tmp/echo25_viewership_results.json"
LOG_FILE = "/tmp/echo25_viewership.log"

# ═══════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("echo25")

# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def get_epoch_range(day: datetime) -> tuple:
    """Get epoch range for 12AM-4AM MT on given day."""
    start_dt = day.replace(hour=WINDOW_START_HOUR_UTC, minute=0, second=0, microsecond=0)
    end_dt = day.replace(hour=WINDOW_END_HOUR_UTC, minute=0, second=0, microsecond=0)
    return int(start_dt.timestamp()), int(end_dt.timestamp())


async def call_with_retry(tool, params, timeout=REQUEST_TIMEOUT, retries=MAX_RETRIES):
    """Call an MCP tool with retry logic and exponential backoff."""
    for attempt in range(retries):
        try:
            result = await asyncio.wait_for(
                tool.ainvoke(params),
                timeout=timeout
            )
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
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

async def main():
    start_time = time.time()
    
    log.info("=" * 70)
    log.info("ECHO25 OUTAGE IMPACT ANALYSIS")
    log.info("Average Local Channel Viewers: 12:00 AM - 4:00 AM MT")
    log.info("=" * 70)
    
    # ─── PHASE 1: Initialize MCP ───────────────────────────────────────
    log.info("\nPHASE 1: Initializing MCP connection...")
    from app.config import get_settings
    from app.agent.agents.utils import get_mcp_tools
    settings = get_settings()
    
    tools = await asyncio.wait_for(
        get_mcp_tools(settings.VIEWERSHIP_MCP_CONFIG),
        timeout=30
    )
    tools_dict = {t.name: t for t in tools}
    log.info(f"  ✓ Connected. Tools: {list(tools_dict.keys())}")
    
    # ─── PHASE 2: Load Local Station Universe ──────────────────────────
    log.info("\nPHASE 2: Loading local station universe...")
    
    stations_file = Path("/tmp/local_stations_filtered.json")
    if stations_file.exists():
        with open(stations_file) as f:
            station_data = json.load(f)
        local_suids = station_data["local_suids"]
        log.info(f"  ✓ Loaded {len(local_suids)} local SUIDs from cache")
    else:
        log.info("  Querying station inventory (no cache found)...")
        import re
        query_stations = tools_dict["query_unique_suid_stations"]
        
        all_stations = []
        for pattern in ["K", "W"]:
            result = await call_with_retry(query_stations, {
                "event_date": "2025-05-12",
                "station": pattern
            })
            all_stations.extend(result.get("results", []))
        
        # Filter to FCC local patterns
        fcc_pattern = re.compile(r"^[KW][A-Z]{2,4}[0-9]?$")
        seen = set()
        locals_only = []
        for s in all_stations:
            key = s["suid"]
            if key not in seen and fcc_pattern.match(s["stationCallSign"]) and 3 <= len(s["stationCallSign"]) <= 6:
                seen.add(key)
                locals_only.append(s)
        
        local_suids = sorted(set(s["suid"] for s in locals_only))
        
        # Cache for reuse
        with open(stations_file, "w") as f:
            json.dump({"local_suids": local_suids, "stations": locals_only}, f)
        log.info(f"  ✓ Found and cached {len(local_suids)} local SUIDs")
    
    # ─── PHASE 3: Determine Query Dates ───────────────────────────────
    log.info("\nPHASE 3: Determining query dates...")
    
    # Use recent dates (adjust based on data availability)
    # Current date: 2026-05-18, but data was available for 2025-05-12 station queries
    # Try both recent ranges
    today = datetime(2026, 5, 18, tzinfo=timezone.utc)
    
    dates_to_query = []
    for i in range(1, SAMPLE_DAYS + 1):
        dates_to_query.append(today - timedelta(days=i))
    
    log.info(f"  Dates: {[d.strftime('%Y-%m-%d') for d in dates_to_query]}")
    log.info(f"  Window: 06:00-10:00 UTC (12:00 AM - 4:00 AM MT)")
    
    # ─── PHASE 4: Method 1 - Per-Service Batch Query ──────────────────
    log.info("\nPHASE 4: METHOD 1 - Per-Service Batch Query")
    log.info(f"  Querying {len(local_suids)} SUIDs in batches of {BATCH_SIZE}")
    log.info(f"  Concurrency: {MAX_CONCURRENT}, Timeout: {REQUEST_TIMEOUT}s")
    
    query_viewership = tools_dict["query_viewership"]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # Split into batches
    suid_batches = [local_suids[i:i+BATCH_SIZE] for i in range(0, len(local_suids), BATCH_SIZE)]
    log.info(f"  Total batches per day: {len(suid_batches)}")
    
    method1_results = {}
    
    async def query_batch(day, batch, batch_idx):
        from_epoch, to_epoch = get_epoch_range(day)
        async with semaphore:
            return await call_with_retry(query_viewership, {
                "fromEpochTimeUtc": from_epoch,
                "toEpochTimeUtc": to_epoch,
                "viewingType": "Tune",
                "serviceUids": batch
            })
    
    for day in dates_to_query:
        day_str = day.strftime("%Y-%m-%d")
        log.info(f"\n  [{day_str}] Launching {len(suid_batches)} batch queries...")
        day_start = time.time()
        
        tasks = [query_batch(day, batch, i) for i, batch in enumerate(suid_batches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate
        total_accounts = 0
        total_sessions = 0
        total_hours = 0
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
        method1_results[day_str] = {
            "total_accounts_sum": total_accounts,
            "total_sessions": total_sessions,
            "total_watch_hours": round(total_hours, 2),
            "services_with_viewers": services_active,
            "batches_completed": len(suid_batches) - errors,
            "errors": errors,
            "elapsed_seconds": round(elapsed, 1)
        }
        
        log.info(f"  [{day_str}] Done in {elapsed:.1f}s | "
                 f"Accounts: {total_accounts:,} | Sessions: {total_sessions:,} | "
                 f"Active services: {services_active} | Errors: {errors}/{len(suid_batches)}")
    
    # ─── PHASE 5: Method 2 - Viewing Summary (True Unique) ────────────
    log.info("\nPHASE 5: METHOD 2 - Platform Viewing Summary")
    log.info("  (True unique accounts across ALL services in time window)")
    
    get_summary = tools_dict["get_viewing_summary"]
    method2_results = {}
    
    for day in dates_to_query:
        day_str = day.strftime("%Y-%m-%d")
        from_epoch, to_epoch = get_epoch_range(day)
        
        result = await call_with_retry(get_summary, {
            "fromEpochTimeUtc": from_epoch,
            "toEpochTimeUtc": to_epoch,
            "viewingType": "Tune"
        })
        
        method2_results[day_str] = result
        if "error" not in result:
            log.info(f"  [{day_str}] Unique accounts: {result.get('totalWatchAccounts', 'N/A'):,} | "
                     f"Total services: {result.get('totalServices', 'N/A')} | "
                     f"Watch hours: {result.get('totalWatchHours', 0):,.1f}")
        else:
            log.info(f"  [{day_str}] ERROR: {result['error']}")
    
    # ─── PHASE 6: Method 3 - Hourly Breakdown (Verification) ──────────
    log.info("\nPHASE 6: METHOD 3 - Hourly Breakdown Verification")
    log.info("  (Sample top local stations for hour-by-hour pattern)")
    
    get_hourly = tools_dict["get_hourly_breakdown"]
    
    # Sample 5 major market stations
    sample_calls = ["KABC", "WABC", "WBBM", "KTTV", "WNBC"]
    sample_suids = []
    for call in sample_calls:
        for s in station_data.get("stations", []):
            if s["stationCallSign"] == call:
                sample_suids.append((call, s["suid"]))
                break
    
    method3_results = {}
    sample_day = dates_to_query[0]
    from_epoch_full = int(sample_day.replace(hour=0).timestamp())
    to_epoch_full = int(sample_day.replace(hour=23, minute=59, second=59).timestamp())
    
    for call_sign, suid in sample_suids[:5]:
        result = await call_with_retry(get_hourly, {
            "fromEpochTimeUtc": from_epoch_full,
            "toEpochTimeUtc": to_epoch_full,
            "serviceUid": suid,
            "viewingType": "Tune"
        })
        
        method3_results[call_sign] = result
        if "error" not in result:
            hourly = result.get("hourly", [])
            midnight_hours = [h for h in hourly if h.get("hour") in (6, 7, 8, 9)]
            total_accts = sum(h.get("totalWatchAccounts", 0) for h in midnight_hours)
            log.info(f"  {call_sign} (SUID {suid}): {total_accts:,} accounts in 06-10 UTC window")
        else:
            log.info(f"  {call_sign}: ERROR: {result.get('error', 'unknown')}")
    
    # ─── PHASE 7: Calculate Final Average ─────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("FINAL RESULTS")
    log.info("=" * 70)
    
    # Method 1 average
    m1_valid = {k: v for k, v in method1_results.items() if v["errors"] < len(suid_batches) * 0.5}
    if m1_valid:
        m1_avg = sum(v["total_accounts_sum"] for v in m1_valid.values()) / len(m1_valid)
        m1_sessions = sum(v["total_sessions"] for v in m1_valid.values()) / len(m1_valid)
        log.info(f"\n  METHOD 1 (per-channel account sum, overcounts multi-channel viewers):")
        log.info(f"    Valid days: {len(m1_valid)}")
        log.info(f"    Average account-channel pairs: {m1_avg:,.0f}")
        log.info(f"    Average sessions: {m1_sessions:,.0f}")
        log.info(f"    Daily breakdown: {json.dumps({k: v['total_accounts_sum'] for k, v in m1_valid.items()})}")
    else:
        m1_avg = None
        log.warning("  METHOD 1: No valid data (all queries failed)")
    
    # Method 2 average
    m2_valid = {k: v for k, v in method2_results.items() if "error" not in v}
    if m2_valid:
        m2_avg = sum(v.get("totalWatchAccounts", 0) for v in m2_valid.values()) / len(m2_valid)
        m2_services = sum(v.get("totalServices", 0) for v in m2_valid.values()) / len(m2_valid)
        log.info(f"\n  METHOD 2 (platform-wide true unique accounts, ALL channels):")
        log.info(f"    Valid days: {len(m2_valid)}")
        log.info(f"    Average unique accounts (all TV): {m2_avg:,.0f}")
        log.info(f"    Average active services: {m2_services:,.0f}")
        log.info(f"    Daily breakdown: {json.dumps({k: v.get('totalWatchAccounts', 0) for k, v in m2_valid.items()})}")
    else:
        m2_avg = None
        log.warning("  METHOD 2: No valid data (all queries failed)")
    
    # Cross-validation estimate
    log.info(f"\n  CROSS-VALIDATION:")
    if m1_avg and m2_avg:
        # Method 1 overcounts (one user watching 3 local channels = 3 account entries)
        # Method 2 includes non-local channels
        # The ratio of local accounts to total gives us an estimate
        local_fraction = m1_avg / m2_avg if m2_avg > 0 else 0
        log.info(f"    Local/Total ratio: {local_fraction:.2%}")
        log.info(f"    Estimated unique local viewers: {m2_avg * min(local_fraction, 1.0):,.0f} (lower bound)")
        log.info(f"    Upper bound (accounts watching ≥1 local): {min(m1_avg, m2_avg):,.0f}")
    
    # ─── SAVE RESULTS ─────────────────────────────────────────────────
    total_elapsed = time.time() - start_time
    
    output = {
        "analysis": "Echo25 Outage Impact - Local Channel Viewership",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query_parameters": {
            "window_mt": "12:00 AM - 4:00 AM Mountain Time",
            "window_utc": "06:00 - 10:00 UTC",
            "viewing_type": "Tune (Live TV)",
            "sample_days": SAMPLE_DAYS,
            "local_suids_queried": len(local_suids),
            "dates_queried": [d.strftime("%Y-%m-%d") for d in dates_to_query]
        },
        "results": {
            "method1_per_service": {
                "description": "Sum of totalWatchAccounts per local service (overcounts multi-channel viewers)",
                "daily_data": method1_results,
                "average": m1_avg
            },
            "method2_platform_summary": {
                "description": "True unique accounts across ALL services (includes non-local)",
                "daily_data": method2_results,
                "average": m2_avg
            },
            "method3_hourly_samples": {
                "description": "Hourly breakdown for sample major-market stations",
                "data": method3_results
            }
        },
        "execution": {
            "total_seconds": round(total_elapsed, 1),
            "batch_size": BATCH_SIZE,
            "concurrency": MAX_CONCURRENT,
            "request_timeout": REQUEST_TIMEOUT
        }
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    log.info(f"\n  Results saved to: {OUTPUT_FILE}")
    log.info(f"  Log file: {LOG_FILE}")
    log.info(f"  Total execution time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    log.info("=" * 70)
    log.info("ANALYSIS COMPLETE")
    log.info("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
