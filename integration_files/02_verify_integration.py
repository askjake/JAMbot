#!/usr/bin/env python3
"""
Verification script for Viewership MCP integration
Run this BEFORE implementing changes to verify current state
"""

import sys
import json
import requests
from pathlib import Path

# Add parent directory to path to import from app
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app.config import get_settings
    settings = get_settings()
except ImportError:
    print("❌ Cannot import settings. Make sure you're in the dish-chat directory.")
    sys.exit(1)

print("=" * 80)
print("VIEWERSHIP MCP INTEGRATION VERIFICATION")
print("=" * 80)
print()

# Check 1: ENABLE_VIEWERSHIP_MCP flag
print("1. Checking ENABLE_VIEWERSHIP_MCP flag...")
enable_flag = getattr(settings, "ENABLE_VIEWERSHIP_MCP", False)
print(f"   Current value: {enable_flag}")
if enable_flag:
    print("   ✅ Viewership MCP is ENABLED")
else:
    print("   ⚠️  Viewership MCP is DISABLED")
print()

# Check 2: Configuration exists
print("2. Checking VIEWERSHIP_MEASUREMENT_MCP_CONFIG...")
if hasattr(settings, "VIEWERSHIP_MEASUREMENT_MCP_CONFIG"):
    config = settings.VIEWERSHIP_MEASUREMENT_MCP_CONFIG
    print(f"   ✅ Configuration exists")
    print(f"   URL: {config.get('viewership_measurement', {}).get('url', 'NOT SET')}")
else:
    print("   ❌ Configuration NOT FOUND")
    sys.exit(1)
print()

# Check 3: Test Lambda endpoint
print("3. Testing Lambda endpoint...")
url = config.get('viewership_measurement', {}).get('url', '')
if url:
    try:
        response = requests.post(
            url,
            json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if 'result' in data and 'tools' in data['result']:
                print(f"   ✅ Lambda is working! Found {len(data['result']['tools'])} tools")
            elif 'errorType' in data:
                print(f"   ⚠️  Lambda error: {data.get('errorMessage', 'Unknown error')}")
            else:
                print(f"   ⚠️  Unexpected response format")
        else:
            print(f"   ❌ HTTP {response.status_code}: {response.text[:100]}")
    except Exception as e:
        print(f"   ❌ Connection failed: {str(e)}")
else:
    print("   ❌ No URL configured")
print()

# Check 4: Verify registry integration
print("4. Checking tool registry integration...")
try:
    from app.agent.agents.tools.registry import _ASYNC_TOOL_FACTORIES
    if "viewership_measurement" in _ASYNC_TOOL_FACTORIES:
        print("   ✅ Viewership tools registered in _ASYNC_TOOL_FACTORIES")
    else:
        print("   ⚠️  Viewership tools NOT in _ASYNC_TOOL_FACTORIES")
        print(f"   Available factories: {list(_ASYNC_TOOL_FACTORIES.keys())}")
except ImportError as e:
    print(f"   ❌ Cannot import registry: {e}")
print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("Next steps:")
if not enable_flag:
    print("❌ Set ENABLE_VIEWERSHIP_MCP=True in config.py or .env")
if 'errorType' in locals():
    print("⚠️  Lambda has runtime errors - contact Ilhyoung to fix")
print("✅ Once Lambda is working, restart backend and test in Dish-Chat")
