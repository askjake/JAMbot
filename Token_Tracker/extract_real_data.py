#!/usr/bin/env python3
"""
Real-Time Token Usage Data Extractor v2.0
Connects to DishChat PostgreSQL and generates structured log data.
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import sys
import os
import argparse

DB_CONFIG = {
    'host': os.environ.get('DB_HOST', '127.0.0.1'),
    'port': int(os.environ.get('DB_PORT', '5433')),
    'database': os.environ.get('DB_NAME', 'dishchat'),
    'user': os.environ.get('DB_USER', 'dev_user'),
    'password': os.environ.get('DB_PASSWORD', 'dev123'),
}


def test_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM usage_tracking")
        count = cursor.fetchone()[0]
        cursor.execute("SELECT MAX(timestamp) FROM usage_tracking")
        latest = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return True, count, latest
    except Exception as e:
        return False, 0, str(e)


def fetch_usage_data(limit=500, hours_back=None):
    try:
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        query = "SELECT ut.timestamp, ut.chat_id, ut.owner_id, ut.model, ut.task, "
        query += "ut.input_tokens, ut.output_tokens, ut.input_cache_read, ut.input_cache_create, "
        query += "ut.input_cost, ut.output_cost FROM usage_tracking ut "
        params = []
        if hours_back:
            query += "WHERE ut.timestamp > NOW() - INTERVAL '%s hours' "
            params.append(hours_back)
        query += "ORDER BY ut.timestamp DESC LIMIT %s"
        params.append(limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        print(f"Query failed: {e}", file=sys.stderr)
        return []


def generate_log_file(output_file, limit=500, hours_back=None):
    print(f"Querying database {DB_CONFIG['host']}:{DB_CONFIG['port']}...")
    rows = fetch_usage_data(limit=limit, hours_back=hours_back)
    if not rows:
        print("No data found")
        return False
    print(f"Found {len(rows)} records")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for row in reversed(rows):
            ts = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if row['timestamp'] else 'unknown'
            user = row['owner_id'].split('@')[0] if row.get('owner_id') and '@' in str(row['owner_id']) else 'unknown'
            total = (row.get('input_tokens', 0) or 0) + (row.get('output_tokens', 0) or 0)
            log_entry = (
                f"{ts} INFO "
                f"chat_id: {row.get('chat_id', 'N/A')}, "
                f"user: {user}, "
                f"model: {row.get('model', 'unknown')}, "
                f"task: {row.get('task', 'unknown')}, "
                f"input_tokens: {row.get('input_tokens', 0)}, "
                f"output_tokens: {row.get('output_tokens', 0)}, "
                f"total_tokens: {total}, "
                f"cache_read_tokens: {row.get('input_cache_read', 0) or 0}, "
                f"cache_create_tokens: {row.get('input_cache_create', 0) or 0}, "
                f"input_cost: {row.get('input_cost', 0.0):.6f}, "
                f"output_cost: {row.get('output_cost', 0.0):.6f}"
            )
            f.write(log_entry + '\n')
    print(f"Log file: {output_file} ({len(rows)} entries)")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract token usage from DishChat database')
    parser.add_argument('--output', '-o', default='/home/jakebot/Token_Tracker/agent_usage.log')
    parser.add_argument('--limit', '-l', type=int, default=500)
    parser.add_argument('--hours', type=int, default=None)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    
    if args.test:
        ok, count, info = test_connection()
        if ok:
            print(f"Connected! {count} records, latest: {info}")
        else:
            print(f"Connection failed: {info}")
        sys.exit(0 if ok else 1)
    
    success = generate_log_file(args.output, args.limit, args.hours)
    sys.exit(0 if success else 1)
