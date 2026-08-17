#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Echo25 Viewership Analysis - Health Check & Launcher
# ═══════════════════════════════════════════════════════════════════════
# 
# This script:
# 1. Tests if the viewership MCP data queries are working
# 2. If healthy, launches the full analysis
# 3. If unhealthy, reports status and exits
#
# Usage: bash /home/jakebot/Jakes-agent/echo25_launch.sh
# ═══════════════════════════════════════════════════════════════════════

cd /home/jakebot/Jakes-agent

echo "════════════════════════════════════════════════════════════"
echo "  ECHO25 VIEWERSHIP ANALYSIS - HEALTH CHECK"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Testing Viewership MCP backend responsiveness..."
echo ""

# Run quick health check (30s timeout for a single query)
RESULT=$(.venv/bin/python -c "
import asyncio, json, sys, time
sys.path.insert(0, '.')

async def health_check():
    from app.config import get_settings
    from app.agent.agents.utils import get_mcp_tools
    settings = get_settings()
    tools = await asyncio.wait_for(get_mcp_tools(settings.VIEWERSHIP_MCP_CONFIG), timeout=15)
    td = {t.name: t for t in tools}
    
    # Quick test: get_top_services with 1hr window, limit 1
    from datetime import datetime, timezone
    day = datetime(2026, 5, 15, tzinfo=timezone.utc)
    start = int(day.replace(hour=6).timestamp())
    end = int(day.replace(hour=7).timestamp())
    
    t = time.time()
    result = await asyncio.wait_for(
        td['get_top_services'].ainvoke({
            'viewingType': 'Tune',
            'fromEpochTimeUtc': start,
            'toEpochTimeUtc': end,
            'limit': 1
        }),
        timeout=30
    )
    if isinstance(result, str):
        result = json.loads(result)
    elapsed = time.time() - t
    
    if 'error' in result:
        print(f'ERROR:{result["error"]}')
    else:
        print(f'OK:{elapsed:.1f}s:{json.dumps(result)}')

try:
    asyncio.run(health_check())
except asyncio.TimeoutError:
    print('TIMEOUT:30s')
except Exception as e:
    print(f'EXCEPTION:{type(e).__name__}:{e}')
" 2>/dev/null)

echo "Health check result: $RESULT"
echo ""

if [[ "$RESULT" == OK:* ]]; then
    echo "✅ BACKEND IS HEALTHY! Launching full analysis..."
    echo ""
    nohup .venv/bin/python echo25_viewership_analysis.py > /tmp/echo25_live.log 2>&1 &
    echo $! > /tmp/echo25_pid.txt
    echo "  PID: $(cat /tmp/echo25_pid.txt)"
    echo "  Log: /tmp/echo25_live.log"
    echo "  Results: /tmp/echo25_viewership_results.json"
    echo ""
    echo "  Monitor: tail -f /tmp/echo25_live.log"
    echo "  Status:  ps -p $(cat /tmp/echo25_pid.txt) -o etime"
    echo ""
    echo "  Expected runtime: ~30-60 min (if backend responsive)"
else
    echo "❌ BACKEND NOT HEALTHY"
    echo ""
    echo "  The Viewership MCP Lambda's data backend (likely Athena) is"
    echo "  not returning results within 30 seconds. This is an infrastructure"
    echo "  issue with the viewership-measurement Lambda function."
    echo ""
    echo "  Action items:"
    echo "    1. Contact the viewership MCP team to check backend status"
    echo "    2. Check if Athena workgroup is throttled or if the S3 data"
    echo "       partition for the queried dates exists"
    echo "    3. Try again later: bash /home/jakebot/Jakes-agent/echo25_launch.sh"
    echo ""
    echo "  Lambda URL: https://cy4h556zxlhqyjju5psohdr6ou0scrxj.lambda-url.us-west-2.on.aws/mcp"
fi
