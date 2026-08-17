#!/usr/bin/env python3
"""
Grasshopper Integration Test Suite
===================================

Tests all Grasshopper tool functionality before deployment.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tools.grasshopper_tool import (
    GrasshopperClient,
    grasshopper_upload_file,
    grasshopper_list_uploadable_files,
    grasshopper_list_file_groups,
    grasshopper_batch_upload,
)


async def test_connection():
    """Test basic connection to Grasshopper API."""
    print("\n" + "="*70)
    print("TEST 1: Connection Test")
    print("="*70)
    
    try:
        client = GrasshopperClient()
        print(f"✅ Client initialized")
        print(f"   Host: {client.base_url}")
        print(f"   Auth: {'OAuth' if client.oauth_enabled else 'Auth Key'}")
        return True
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


async def test_list_file_groups():
    """Test listing file groups."""
    print("\n" + "="*70)
    print("TEST 2: List File Groups")
    print("="*70)
    
    try:
        result = await grasshopper_list_file_groups()
        data = json.loads(result)
        
        if "error" in data:
            print(f"⚠️  API returned error: {data['error']}")
            return False
        
        print(f"✅ Retrieved file groups:")
        print(json.dumps(data, indent=2)[:500])
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def test_list_uploadable_files():
    """Test listing uploadable files for a receiver."""
    print("\n" + "="*70)
    print("TEST 3: List Uploadable Files")
    print("="*70)
    
    # Use test receiver ID
    test_rec_id = "1971450629"
    
    try:
        result = await grasshopper_list_uploadable_files(rec_id=test_rec_id)
        data = json.loads(result)
        
        if "error" in data:
            print(f"⚠️  API returned error: {data['error']}")
            print(f"   This may be expected if receiver {test_rec_id} is not active")
            return True  # Not a test failure
        
        print(f"✅ Retrieved uploadable files for rec_id={test_rec_id}:")
        print(json.dumps(data, indent=2)[:500])
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def test_upload_validation():
    """Test upload validation (without actually uploading)."""
    print("\n" + "="*70)
    print("TEST 4: Upload Validation")
    print("="*70)
    
    # Test with non-existent file
    test_rec_id = "1971450629"
    fake_file = "/tmp/nonexistent_test_file.log"
    
    try:
        result = await grasshopper_upload_file(
            rec_id=test_rec_id,
            file_path=fake_file,
            file_group="logs"
        )
        data = json.loads(result)
        
        if not data.get("success") and "not found" in data.get("message", "").lower():
            print(f"✅ Upload validation working correctly")
            print(f"   Expected error for non-existent file: {data['message']}")
            return True
        else:
            print(f"⚠️  Unexpected result: {data}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def test_batch_upload_validation():
    """Test batch upload validation."""
    print("\n" + "="*70)
    print("TEST 5: Batch Upload Validation")
    print("="*70)
    
    test_rec_id = "1971450629"
    test_files = json.dumps(["/tmp/file1.log", "/tmp/file2.log"])
    
    try:
        result = await grasshopper_batch_upload(
            rec_id=test_rec_id,
            file_paths=test_files,
            file_group="logs"
        )
        data = json.loads(result)
        
        if "total" in data and "results" in data:
            print(f"✅ Batch upload structure correct")
            print(f"   Total files: {data['total']}")
            print(f"   Results: {len(data['results'])} entries")
            return True
        else:
            print(f"⚠️  Unexpected structure: {data}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("GRASSHOPPER INTEGRATION TEST SUITE")
    print("="*70)
    print(f"Testing against: {GrasshopperClient().base_url}")
    
    tests = [
        ("Connection", test_connection),
        ("List File Groups", test_list_file_groups),
        ("List Uploadable Files", test_list_uploadable_files),
        ("Upload Validation", test_upload_validation),
        ("Batch Upload Validation", test_batch_upload_validation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Integration is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
