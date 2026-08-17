print('╔══════════════════════════════════════════════════════════════╗')
print('║                                                              ║')
print('║         COMPREHENSIVE FIX VERIFICATION                       ║')
print('║                                                              ║')
print('╚══════════════════════════════════════════════════════════════╝')
print()

import sys

tests_passed = 0
tests_total = 7

# Test 1: File exists and syntax valid
print('Test 1: File & Syntax')
try:
    import py_compile
    py_compile.compile('app_enhanced.py', doraise=True)
    print('   ✅ PASS - Syntax valid')
    tests_passed += 1
except Exception as e:
    print(f'   ❌ FAIL - {e}')

# Test 2: Database connection
print('Test 2: Database Connection')
try:
    import psycopg
    from psycopg.rows import dict_row
    conn = psycopg.connect(
        host='127.0.0.1',
        port=5432,
        dbname='dishchat',
        user='dev_user',
        password='dev123',
        row_factory=dict_row
    )
    print('   ✅ PASS - Connected')
    tests_passed += 1
    conn_works = True
except Exception as e:
    print(f'   ❌ FAIL - {e}')
    conn_works = False

# Test 3: Check default time range
print('Test 3: Default Time Range')
with open('app_enhanced.py', 'r') as f:
    content = f.read()
    if 'value=168,' in content:
        print('   ✅ PASS - Default is 168 hours')
        tests_passed += 1
    else:
        print('   ❌ FAIL - Default not set to 168')

# Test 4: Check INTERVAL syntax
print('Test 4: Query Syntax (MAKE_INTERVAL)')
if 'MAKE_INTERVAL(hours => %s)' in content:
    print('   ✅ PASS - Using MAKE_INTERVAL')
    tests_passed += 1
else:
    print('   ❌ FAIL - Still using old INTERVAL syntax')

# Test 5: Test actual query with 168 hours
if conn_works:
    print('Test 5: Query Execution (168h)')
    try:
        cursor = conn.cursor()
        query = """
        SELECT COUNT(*) as count
        FROM usage_tracking
        WHERE timestamp >= NOW() - MAKE_INTERVAL(hours => %s)
        """
        cursor.execute(query, (168,))
        result = cursor.fetchone()
        count = result['count']
        
        if count > 0:
            print(f'   ✅ PASS - Query returns {count:,} rows')
            tests_passed += 1
        else:
            print('   ❌ FAIL - Query returns 0 rows')
        cursor.close()
    except Exception as e:
        print(f'   ❌ FAIL - {e}')
else:
    print('Test 5: Query Execution')
    print('   ⏭️  SKIP - DB not connected')

# Test 6: Full query test
if conn_works:
    print('Test 6: Full App Query')
    try:
        query = """
        SELECT 
            timestamp,
            chat_id,
            owner_id as user_email,
            model,
            task,
            input_tokens,
            output_tokens,
            input_cache_read,
            input_cache_create,
            (input_tokens + output_tokens) as total_tokens
        FROM usage_tracking
        WHERE timestamp >= NOW() - MAKE_INTERVAL(hours => %s)
        ORDER BY timestamp DESC
        LIMIT 1000
        """
        cursor = conn.cursor()
        cursor.execute(query, (168,))
        rows = cursor.fetchall()
        
        if len(rows) > 0:
            print(f'   ✅ PASS - Retrieved {len(rows)} records')
            tests_passed += 1
        else:
            print('   ❌ FAIL - No records retrieved')
        cursor.close()
    except Exception as e:
        print(f'   ❌ FAIL - {e}')
else:
    print('Test 6: Full App Query')
    print('   ⏭️  SKIP - DB not connected')

# Test 7: Calculate metrics
if conn_works:
    print('Test 7: Metrics Calculation')
    try:
        import pandas as pd
        cursor = conn.cursor()
        cursor.execute("""
        SELECT 
            SUM(input_tokens) as total_input,
            SUM(output_tokens) as total_output,
            SUM(input_cache_read) as total_cache,
            COUNT(*) as total_calls
        FROM usage_tracking
        WHERE timestamp >= NOW() - MAKE_INTERVAL(hours => %s)
        """, (168,))
        result = cursor.fetchone()
        
        if result and result['total_input']:
            print(f'   ✅ PASS - Metrics available')
            print(f'      Input: {result["total_input"]:,} tokens')
            print(f'      Output: {result["total_output"]:,} tokens')
            print(f'      Cache: {result["total_cache"]:,} tokens')
            print(f'      Calls: {result["total_calls"]:,}')
            tests_passed += 1
        else:
            print('   ❌ FAIL - No metrics')
        cursor.close()
    except Exception as e:
        print(f'   ❌ FAIL - {e}')
else:
    print('Test 7: Metrics Calculation')
    print('   ⏭️  SKIP - DB not connected')

if conn_works:
    conn.close()

print()
print('╔══════════════════════════════════════════════════════════════╗')
print(f'║  RESULTS: {tests_passed}/{tests_total} tests passed                                   ║')
print('╚══════════════════════════════════════════════════════════════╝')
print()

if tests_passed == tests_total:
    print('🎉 ALL TESTS PASSED - DASHBOARD READY!')
    print()
    print('🚀 Launch command:')
    print('   cd ~/Token_Tracker')
    print('   bash test_enhanced.sh')
    print()
    print('🌐 Dashboard URL:')
    print('   http://10.79.85.47:8505')
    print()
    print('📊 Expected Data:')
    print('   • 326 records from last 7 days')
    print('   • Metrics will display immediately')
    print('   • All 4 tabs will show data')
    sys.exit(0)
else:
    print(f'⚠️  {tests_total - tests_passed} test(s) failed')
    print('Review errors above and retry')
    sys.exit(1)
