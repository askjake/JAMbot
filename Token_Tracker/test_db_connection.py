import psycopg
from psycopg.rows import dict_row
from datetime import datetime

print('═══════════════════════════════════════════════════════')
print('🔍 Phase 1: Testing Database Connection & Data')
print('═══════════════════════════════════════════════════════')
print()

DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 5432,
    'dbname': 'dishchat',
    'user': 'dev_user',
    'password': 'dev123'
}

# Test 1: Basic connection
print('Test 1: Basic Connection')
try:
    conn = psycopg.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        dbname=DB_CONFIG['dbname'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        row_factory=dict_row
    )
    print('   ✅ Connection successful')
except Exception as e:
    print(f'   ❌ Failed: {e}')
    exit(1)

# Test 2: Count total records
print('Test 2: Total Records')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) as count FROM usage_tracking')
result = cursor.fetchone()
total = result['count']
print(f'   Total records: {total:,}')

# Test 3: Check latest timestamp
print('Test 3: Data Age Check')
cursor.execute('SELECT MAX(timestamp) as latest, MIN(timestamp) as oldest FROM usage_tracking')
result = cursor.fetchone()
latest = result['latest']
oldest = result['oldest']

print(f'   Latest record: {latest}')
print(f'   Oldest record: {oldest}')

if latest:
    now = datetime.now()
    latest_naive = latest.replace(tzinfo=None) if latest.tzinfo else latest
    age_hours = (now - latest_naive).total_seconds() / 3600
    age_days = age_hours / 24
    print(f'   Data age: {age_hours:.1f} hours ({age_days:.1f} days)')
    
    if age_hours > 24:
        print(f'   🔴 ISSUE: Data is {age_hours:.1f} hours old!')
        print(f'   Default app shows last 24h, so NO DATA displays')
    else:
        print(f'   ✅ Data is fresh (< 24 hours)')

# Test 4: Query with 24 hour window
print('Test 4: Query Last 24 Hours')
cursor.execute('''
    SELECT COUNT(*) as count
    FROM usage_tracking
    WHERE timestamp >= NOW() - INTERVAL '24 hours'
''')
result = cursor.fetchone()
count_24h = result['count']
print(f'   Records in last 24h: {count_24h:,}')

# Test 5: Query with 7 days
print('Test 5: Query Last 7 Days')
cursor.execute('''
    SELECT COUNT(*) as count
    FROM usage_tracking
    WHERE timestamp >= NOW() - INTERVAL '7 days'
''')
result = cursor.fetchone()
count_7d = result['count']
print(f'   Records in last 7 days: {count_7d:,}')

# Test 6: Query with 168 hours (max slider)
print('Test 6: Query Last 168 Hours (7 days)')
cursor.execute('''
    SELECT COUNT(*) as count
    FROM usage_tracking
    WHERE timestamp >= NOW() - INTERVAL '168 hours'
''')
result = cursor.fetchone()
count_168h = result['count']
print(f'   Records in last 168h: {count_168h:,}')

cursor.close()
conn.close()

print()
print('═══════════════════════════════════════════════════════')
print('DIAGNOSIS:')
if count_24h == 0:
    print('🔴 No data in last 24 hours (app default)')
    if count_168h > 0:
        print('✅ Solution: Increase time range slider to 168 hours')
    else:
        print('🔴 No data even in 7 days - database may be stale')
else:
    print('✅ Data should be visible in dashboard')
print('═══════════════════════════════════════════════════════')
