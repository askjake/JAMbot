#!/usr/bin/env python3
'''
🚀 DishChat Token Usage Tracker (Enhanced)
Real-time monitoring of token usage with cost analysis and advanced visualizations

Features:
  - Multi-tab interface (Timeline, Breakdown, Cache Stats, Raw Data)
  - Altair charts (faster, cleaner than Plotly)
  - Cost calculation with model-specific pricing
  - Cache efficiency metrics
  - Model and task breakdowns
  - Database integration (PostgreSQL)
  
Author: Dish-Chat AI Assistant (Enhanced Edition)
Run: streamlit run app.py --server.port 8503
'''

import streamlit as st
import pandas as pd
import altair as alt
import psycopg
from psycopg.rows import dict_row
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

MT = ZoneInfo('America/Denver')  # Mountain Time (auto DST: MST/MDT)
import os

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Page config
st.set_page_config(
    page_title="Jakes-agent Token Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CONVERSATION STATS PARSING (from streaming logs)
# ============================================================================

import os

# Possible log file locations
# Log paths for Jakes-agent on 10.79.85.35 (port 8002)
AGENT_DIR = '/home/jakebot/Jakes-agent'
LOG_PATHS = [
    '/home/jakebot/Jakes-agent/logs/backend.log',        # Primary — uvicorn log
    '/home/jakebot/Jakes-agent/logs/backend_8002.log',   # Port-specific variant
    '/home/jakebot/Jakes-agent/logs/backend_8000.log',   # Fallback port 8000
    '/home/jakebot/Jakes-agent/smplogs/backend.log',     # Alternate log dir
]

def find_log_file():
    """Find the most recent active log file"""
    for path in LOG_PATHS:
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 1000:  # At least 1KB
                return path
    return None

def parse_streaming_logs(log_file=None, lines_to_read=2000):
    """Parse conversation stats from streaming logs"""
    
    if log_file is None:
        log_file = find_log_file()
    
    if not log_file:
        return None
    
    stats = {
        'timestamp': None,
        'messages_total': None,
        'messages_kept': None,
        'messages_truncated': None,
        'tokens_before_truncation': None,
        'system_prompt_tokens': None,
        'tools_tokens': None,
        'tool_count': None,
        'final_token_count': None,
        'token_limit': None,
        'percentage_used': None,
        'safe_limit': None,
        'complexity': None,
        'model': None,
    }
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()[-lines_to_read:]
    except Exception as e:
        return None
    
    # Find the most recent full cycle
    for line in reversed(lines):
        # Extract timestamp
        ts_match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if ts_match and not stats['timestamp']:
            stats['timestamp'] = ts_match.group(1)
        
        # Pre-truncation overhead
        if 'Pre-truncation overhead' in line:
            match = re.search(r'system_prompt=(\d+) tokens.*tools=(\d+) tokens.*\((\d+) tools\)', line)
            if match and not stats['system_prompt_tokens']:
                stats['system_prompt_tokens'] = int(match.group(1))
                stats['tools_tokens'] = int(match.group(2))
                stats['tool_count'] = int(match.group(3))
        
        # Truncating message history
        if 'Truncating message history from' in line:
            match = re.search(r'from (\d+) to (\d+)', line)
            if match and not stats['messages_total']:
                stats['messages_total'] = int(match.group(1))
                stats['messages_kept'] = int(match.group(2))
                stats['messages_truncated'] = stats['messages_total'] - stats['messages_kept']
        
        # Message tokens before truncation
        if 'Message tokens before truncation:' in line:
            match = re.search(r': (\d+)', line)
            if match and not stats['tokens_before_truncation']:
                stats['tokens_before_truncation'] = int(match.group(1))
        
        # Complexity detection
        if 'detect_prompt_complexity' in line:
            if 'Simple prompt' in line and not stats['complexity']:
                stats['complexity'] = 'Simple'
                stats['model'] = 'Sonnet'
            elif 'Complex prompt' in line and not stats['complexity']:
                stats['complexity'] = 'Complex'
                stats['model'] = 'Opus' if 'Opus' in line else 'Sonnet'
        
        # Calling model with N messages
        if 'Calling model with' in line:
            match = re.search(r'with (\d+) messages', line)
            if match and not stats['messages_kept']:
                stats['messages_kept'] = int(match.group(1))
        
        # Final token count
        if 'Final token count:' in line:
            match = re.search(r'(\d+) / (\d+) \(([\d.]+)%\), safe_limit=(\d+)', line)
            if match:
                stats['final_token_count'] = int(match.group(1))
                stats['token_limit'] = int(match.group(2))
                stats['percentage_used'] = float(match.group(3))
                stats['safe_limit'] = int(match.group(4))
                break  # Found the latest, stop
    
    return stats



# ============================================================================
# TIMESTAMP FORMATTING HELPERS
# ============================================================================

import time

def format_timestamp_with_timezone(timestamp_str):
    """Convert UTC DB timestamp to Mountain Time (MST/MDT)"""
    if not timestamp_str or timestamp_str == 'N/A':
        return 'N/A'
    try:
        from zoneinfo import ZoneInfo
        UTC = ZoneInfo('UTC')
        dt_utc = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC)
        dt_mt  = dt_utc.astimezone(MT)
        tz_abbr = dt_mt.strftime('%Z')          # 'MST' or 'MDT'
        formatted = dt_mt.strftime('%B %-d, %Y %-I:%M:%S %p')
        return f"{formatted} {tz_abbr}"
    except Exception:
        return timestamp_str

def get_time_ago(timestamp_str):
    """Calculate how long ago a timestamp was"""
    if not timestamp_str or timestamp_str == 'N/A':
        return ''
    
    try:
        from zoneinfo import ZoneInfo
        UTC = ZoneInfo('UTC')
        dt  = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC).astimezone(MT)
        now = datetime.now(tz=MT)
        diff = now - dt
        seconds = diff.total_seconds()
        
        if seconds < 0:
            return 'just now'
        elif seconds < 60:
            return f"{int(seconds)} seconds ago"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"
    except Exception:
        return ''

def get_data_freshness_indicator(timestamp_str):
    """Get freshness indicator based on age"""
    if not timestamp_str or timestamp_str == 'N/A':
        return ('⚪', 'Unknown', 'gray')
    
    try:
        from zoneinfo import ZoneInfo
        UTC = ZoneInfo('UTC')
        dt  = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=UTC).astimezone(MT)
        now = datetime.now(tz=MT)
        diff = now - dt
        minutes = diff.total_seconds() / 60
        
        if minutes < 5:
            return ('🟢', 'Fresh', 'green')
        elif minutes < 30:
            return ('🟡', 'Recent', 'yellow')
        elif minutes < 60:
            return ('🟠', 'Stale', 'orange')
        else:
            return ('🔴', 'Old', 'red')
    except Exception:
        return ('⚪', 'Unknown', 'gray')


def calculate_conversation_derived_stats(stats):
    """Calculate derived statistics for conversation"""
    if not stats or not stats.get('final_token_count'):
        return {}
    
    derived = {}
    
    if stats['final_token_count'] and stats['safe_limit']:
        derived['percentage_of_safe_limit'] = (stats['final_token_count'] / stats['safe_limit']) * 100
        derived['tokens_until_safe_limit'] = stats['safe_limit'] - stats['final_token_count']
    
    if stats['final_token_count'] and stats['token_limit']:
        derived['tokens_until_hard_limit'] = stats['token_limit'] - stats['final_token_count']
    
    if stats['system_prompt_tokens'] and stats['final_token_count']:
        derived['message_tokens'] = stats['final_token_count'] - stats['system_prompt_tokens']
        derived['system_percentage'] = (stats['system_prompt_tokens'] / stats['final_token_count']) * 100
        derived['message_percentage'] = (derived['message_tokens'] / stats['final_token_count']) * 100
    
    if stats.get('messages_truncated') and stats.get('messages_total'):
        derived['truncation_happened'] = True
        derived['truncation_rate'] = (stats['messages_truncated'] / stats['messages_total']) * 100
    else:
        derived['truncation_happened'] = False
        derived['truncation_rate'] = 0.0
    
    if stats.get('messages_kept') and stats.get('final_token_count'):
        derived['tokens_per_message'] = stats['final_token_count'] / stats['messages_kept']
    
    return derived


# Database configuration
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 5434,  # Jakes-agent DB (docker port 5434->5432 on 10.79.85.35)
    'dbname': 'dishchat',
    'user': 'dev_user',
    'password': 'dev123'
}

# Model pricing (per 1M tokens)
MODEL_PRICING = {
    'claude-sonnet-4-5': {'input': 3.00, 'output': 15.00, 'cache_read': 0.30, 'cache_create': 3.75},
    'claude-sonnet-3-5': {'input': 3.00, 'output': 15.00, 'cache_read': 0.30, 'cache_create': 3.75},
    'claude-haiku': {'input': 0.25, 'output': 1.25, 'cache_read': 0.03, 'cache_create': 0.30},
    'claude-opus': {'input': 15.00, 'output': 75.00, 'cache_read': 1.50, 'cache_create': 18.75},
    'default': {'input': 3.00, 'output': 15.00, 'cache_read': 0.30, 'cache_create': 3.75}
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(ttl=300)
def get_db_connection():
    """Get database connection (cached for 5 min)"""
    try:
        conn = psycopg.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            dbname=DB_CONFIG['dbname'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            row_factory=dict_row,
            connect_timeout=5
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def identify_model(model_str: str) -> str:
    """Identify model type from model string"""
    model_lower = model_str.lower()
    if 'sonnet-4' in model_lower or 'sonnet-4-5' in model_lower:
        return 'claude-sonnet-4-5'
    elif 'sonnet' in model_lower:
        return 'claude-sonnet-3-5'
    elif 'haiku' in model_lower:
        return 'claude-haiku'
    elif 'opus' in model_lower:
        return 'claude-opus'
    else:
        return 'default'

def calculate_cost(row: dict) -> float:
    """Calculate cost for a single record"""
    model_type = identify_model(row.get('model', ''))
    pricing = MODEL_PRICING.get(model_type, MODEL_PRICING['default'])
    
    cost = 0.0
    cost += (row.get('input_tokens', 0) / 1_000_000) * pricing['input']
    cost += (row.get('output_tokens', 0) / 1_000_000) * pricing['output']
    cost += (row.get('input_cache_read', 0) / 1_000_000) * pricing['cache_read']
    cost += (row.get('input_cache_create', 0) / 1_000_000) * pricing['cache_create']
    
    return cost

@st.cache_data(ttl=10)
def query_usage_data(hours_back: int = 24):
    """Query usage data from database"""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
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
        LIMIT 5000
        """
        
        with conn.cursor() as cursor:
            cursor.execute(query, (hours_back,))
            rows = cursor.fetchall()
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Calculate cost for each row
        df['cost'] = df.apply(calculate_cost, axis=1)
        
        # Clean up model names for display
        df['model_short'] = df['model'].apply(lambda x: x.split('/')[-1][:30] if x else 'unknown')
        
        return df
        
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

# Header
st.markdown('<div class="main-header">📊 Jakes-agent Token Usage Tracker</div>', unsafe_allow_html=True)
st.caption("Real-time monitoring — Jakes-agent @ 10.79.85.35:8002 — DB port 5434")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    hours_back = st.slider(
        "Time Range (hours)",
        min_value=1,
        max_value=720,
        value=720,
        help="How far back to query historical data (720h = 30 days)"
    )
    
    auto_refresh = st.checkbox(
        "Auto-refresh (10s)",
        value=True,
        help="Automatically refresh data every 10 seconds"
    )
    
    if auto_refresh:
        st.caption("⏱️ Auto-refreshing every 10 seconds...")
    
    st.divider()
    
    st.markdown("""
    ### 📊 About
    This enhanced tracker provides:
    - 💰 Cost analysis
    - 📈 Advanced charts
    - 🧩 Model breakdowns
    - 💾 Cache efficiency
    - 📋 Multi-tab interface
    
    **Agent:** Jakes-agent (port 8002)  
    **Database:** PostgreSQL (port 5434)  
    **Update Freq:** 10 seconds
    """)

# Fetch data
df = query_usage_data(hours_back)

# Auto-refresh
# ─────────────────────────────────────────────────────────────────────────────
# Metrics Row
# ─────────────────────────────────────────────────────────────────────────────

if not df.empty:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_input = int(df['input_tokens'].sum())
    total_output = int(df['output_tokens'].sum())
    total_cache_read = int(df['input_cache_read'].sum())
    total_cache_create = int(df['input_cache_create'].sum())
    total_cost = df['cost'].sum()
    num_calls = len(df)
    
    with col1:
        st.metric("📥 Input Tokens", f"{total_input:,}")
    with col2:
        st.metric("📤 Output Tokens", f"{total_output:,}")
    with col3:
        st.metric("💾 Cache Read", f"{total_cache_read:,}")
    with col4:
        st.metric("💰 Total Cost", f"${total_cost:.4f}")
    with col5:
        st.metric("🔢 API Calls", f"{num_calls:,}")
else:
    st.warning("⚠️ No data available from database. The database has no records in the selected time range.")
    
    # ── Log-file fallback diagnostic ──
    import os
    LOG_FILE = os.path.expanduser("~/Token_Tracker/agent_usage.log")  # jakebot home
    if os.path.exists(LOG_FILE):
        mtime = os.path.getmtime(LOG_FILE)
        import datetime
        mod_time = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        size = os.path.getsize(LOG_FILE)
        st.info(f"📄 **agent_usage.log** found: `{LOG_FILE}`\n\n"
                f"Last modified: `{mod_time}` | Size: `{size:,}` bytes\n\n"
                f"⚠️ The log file exists but the database is not being updated. "
                f"Run `extract_real_data.py` to populate the database, "
                f"or increase the **Time Range** slider above {hours_back}h.")
    else:
        st.error(f"❌ Log file not found at: {LOG_FILE}")
        st.info("💡 To generate data, run: `python3 extract_real_data.py` from ~/Token_Tracker/")
    
    st.info(f"📊 **Database status**: Last record was on `2026-05-05`. "
            f"Increase the time range slider past **{hours_back}h** or re-populate the database.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Token Timeline", 
    "🧩 Breakdown Analysis", 
    "💾 Cache Efficiency",
    "📋 Raw Data",
    "🔄 Current Conversation"
])

# ─── Tab 1: Token Timeline ───
with tab1:
    st.subheader(f"Token Usage Over Last {hours_back} Hours")
    
    if not df.empty:
        # Resample to 5-minute buckets
        df_chart = df.copy()
        df_chart['timestamp'] = pd.to_datetime(df_chart['timestamp'])
        df_chart = df_chart.set_index('timestamp').resample('5min').agg({
            'input_tokens': 'sum',
            'output_tokens': 'sum',
            'input_cache_read': 'sum',
            'input_cache_create': 'sum',
            'cost': 'sum'
        }).reset_index()
        
        # Filter out empty buckets
        df_chart = df_chart[df_chart['input_tokens'] + df_chart['output_tokens'] > 0]
        
        if not df_chart.empty:
            # Stacked area chart
            df_melted = df_chart.melt(
                id_vars=['timestamp'],
                value_vars=['input_tokens', 'output_tokens', 'input_cache_read'],
                var_name='Token Type',
                value_name='Count'
            )
            
            chart = alt.Chart(df_melted).mark_area(
                opacity=0.7,
                interpolate='monotone'
            ).encode(
                x=alt.X('timestamp:T', title='Time', axis=alt.Axis(format='%H:%M')),
                y=alt.Y('Count:Q', title='Tokens', stack=True),
                color=alt.Color(
                    'Token Type:N',
                    scale=alt.Scale(
                        domain=['input_tokens', 'output_tokens', 'input_cache_read'],
                        range=['#4fc3f7', '#ff7043', '#66bb6a']
                    ),
                    legend=alt.Legend(title=None, orient='top')
                ),
                tooltip=[
                    alt.Tooltip('timestamp:T', title='Time', format='%Y-%m-%d %H:%M:%S'),
                    alt.Tooltip('Token Type:N'),
                    alt.Tooltip('Count:Q', format=',', title='Tokens')
                ]
            ).properties(
                height=400
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
            # Cost timeline
            st.subheader("💰 Cost per 5-Minute Window")
            
            cost_chart = alt.Chart(df_chart).mark_line(
                color='#ffd54f',
                strokeWidth=3,
                point=True
            ).encode(
                x=alt.X('timestamp:T', title='Time', axis=alt.Axis(format='%H:%M')),
                y=alt.Y('cost:Q', title='Cost ($)'),
                tooltip=[
                    alt.Tooltip('timestamp:T', format='%Y-%m-%d %H:%M:%S'),
                    alt.Tooltip('cost:Q', title='Cost', format='$.6f')
                ]
            ).properties(
                height=200
            ).interactive()
            
            st.altair_chart(cost_chart, use_container_width=True)
        else:
            st.info("No data points in the selected time range after aggregation.")

# ─── Tab 2: Breakdown Analysis ───
with tab2:
    if not df.empty:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("📊 Tokens by Model")
            
            model_df = df.groupby('model_short').agg({
                'input_tokens': 'sum',
                'output_tokens': 'sum',
                'total_tokens': 'sum',
                'cost': 'sum'
            }).reset_index().sort_values('total_tokens', ascending=False)
            
            model_chart = alt.Chart(model_df).mark_bar(
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4
            ).encode(
                x=alt.X('total_tokens:Q', title='Total Tokens'),
                y=alt.Y('model_short:N', title='Model', sort='-x'),
                color=alt.Color('model_short:N', legend=None, scale=alt.Scale(scheme='category10')),
                tooltip=[
                    alt.Tooltip('model_short:N', title='Model'),
                    alt.Tooltip('total_tokens:Q', format=',', title='Total'),
                    alt.Tooltip('input_tokens:Q', format=',', title='Input'),
                    alt.Tooltip('output_tokens:Q', format=',', title='Output'),
                    alt.Tooltip('cost:Q', format='$.4f', title='Cost')
                ]
            ).properties(height=350)
            
            st.altair_chart(model_chart, use_container_width=True)
        
        with col_b:
            st.subheader("🎯 Distribution by Task")
            
            task_df = df.groupby('task').agg({
                'total_tokens': 'sum',
                'cost': 'sum'
            }).reset_index()
            
            task_chart = alt.Chart(task_df).mark_arc(
                innerRadius=60,
                outerRadius=130
            ).encode(
                theta=alt.Theta('total_tokens:Q'),
                color=alt.Color('task:N', scale=alt.Scale(scheme='category10')),
                tooltip=[
                    alt.Tooltip('task:N'),
                    alt.Tooltip('total_tokens:Q', format=','),
                    alt.Tooltip('cost:Q', format='$.4f')
                ]
            ).properties(height=350)
            
            st.altair_chart(task_chart, use_container_width=True)
        
        # Cost breakdown
        st.subheader("💰 Cost Breakdown by Model")
        
        cost_df = df.groupby('model_short')['cost'].sum().reset_index().sort_values('cost', ascending=False)
        
        cost_bar = alt.Chart(cost_df).mark_bar(
            color='#ffd54f',
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4
        ).encode(
            x=alt.X('cost:Q', title='Total Cost ($)'),
            y=alt.Y('model_short:N', title='Model', sort='-x'),
            tooltip=[
                alt.Tooltip('model_short:N', title='Model'),
                alt.Tooltip('cost:Q', format='$.4f', title='Cost')
            ]
        ).properties(height=250)
        
        st.altair_chart(cost_bar, use_container_width=True)

# ─── Tab 3: Cache Efficiency ───
with tab3:
    st.subheader("💾 Cache Performance Analysis")
    
    if not df.empty:
        total_input_all = df['input_tokens'].sum()
        total_cache_read = df['input_cache_read'].sum()
        total_cache_create = df['input_cache_create'].sum()
        
        # Cache hit rate
        total_cacheable = total_input_all + total_cache_read + total_cache_create
        if total_cacheable > 0:
            cache_hit_rate = (total_cache_read / total_cacheable) * 100
        else:
            cache_hit_rate = 0
        
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        
        with col_c1:
            st.metric("🎯 Cache Hit Rate", f"{cache_hit_rate:.1f}%")
        with col_c2:
            st.metric("📖 Cache Reads", f"{int(total_cache_read):,}")
        with col_c3:
            st.metric("✍️ Cache Creates", f"{int(total_cache_create):,}")
        with col_c4:
            # Cost saved by cache
            cache_pricing = MODEL_PRICING['default']
            cache_saved = (total_cache_read / 1_000_000) * (cache_pricing['input'] - cache_pricing['cache_read'])
            st.metric("💵 Cache Savings", f"${cache_saved:.4f}")
        
        st.divider()
        
        # Cache timeline
        st.subheader("📈 Cache Usage Over Time")
        
        df_cache = df.copy()
        df_cache['timestamp'] = pd.to_datetime(df_cache['timestamp'])
        df_cache = df_cache.set_index('timestamp').resample('10min').agg({
            'input_cache_read': 'sum',
            'input_cache_create': 'sum'
        }).reset_index()
        
        df_cache_melted = df_cache.melt(
            id_vars=['timestamp'],
            value_vars=['input_cache_read', 'input_cache_create'],
            var_name='Cache Type',
            value_name='Tokens'
        )
        
        cache_chart = alt.Chart(df_cache_melted).mark_area(
            opacity=0.7
        ).encode(
            x=alt.X('timestamp:T', title='Time', axis=alt.Axis(format='%H:%M')),
            y=alt.Y('Tokens:Q', title='Cache Tokens'),
            color=alt.Color(
                'Cache Type:N',
                scale=alt.Scale(
                    domain=['input_cache_read', 'input_cache_create'],
                    range=['#66bb6a', '#ffa726']
                ),
                legend=alt.Legend(title=None, orient='top')
            ),
            tooltip=[
                alt.Tooltip('timestamp:T', format='%Y-%m-%d %H:%M:%S'),
                alt.Tooltip('Cache Type:N'),
                alt.Tooltip('Tokens:Q', format=',')
            ]
        ).properties(height=350).interactive()
        
        st.altair_chart(cache_chart, use_container_width=True)
        
        # Efficiency explanation
        st.info(f"""
        💡 **Cache Efficiency Insights:**
        - Your cache hit rate is **{cache_hit_rate:.1f}%**, which means {cache_hit_rate:.1f}% of potential input tokens were served from cache.
        - You've saved **${cache_saved:.4f}** by using cached tokens instead of full input processing.
        - Cache reads are ~10x cheaper than regular input tokens!
        """)

# ─── Tab 4: Raw Data ───
with tab4:
    st.subheader("📋 Raw Usage Data")
    
    if not df.empty:
        st.caption(f"Showing {len(df)} records from the last {hours_back} hours")
        
        # Display options
        show_cols = st.multiselect(
            "Select columns to display:",
            options=df.columns.tolist(),
            default=['timestamp', 'model_short', 'task', 'input_tokens', 'output_tokens', 'cost']
        )
        
        if show_cols:
            st.dataframe(
                df[show_cols].head(100),
                use_container_width=True,
                height=500
            )
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"token_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )



# ============================================================================
# TAB 5: CURRENT CONVERSATION STATS
# ============================================================================

with tab5:
    st.header("🔄 Current Conversation Statistics")
    

    
    # Parse logs
    stats = parse_streaming_logs()
    # Show data freshness at the top
    if stats and stats.get('timestamp'):
        raw_ts = stats.get('timestamp')
        emoji, freshness, color = get_data_freshness_indicator(raw_ts)
        time_ago = get_time_ago(raw_ts)
        if time_ago:
            st.caption(f"{emoji} Data from {time_ago} - Status: {freshness}")
    
    if stats and stats.get('final_token_count'):
        derived = calculate_conversation_derived_stats(stats)
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Tokens",
                f"{(stats.get('final_token_count') or 0):,f}",
                f"{stats['percentage_used']:.1f}% of limit"
            )
        
        with col2:
            pct_safe = derived.get('percentage_of_safe_limit', 0)
            color = "🟢" if pct_safe < 70 else "🟡" if pct_safe < 85 else "🔴"
            st.metric(
                "Safe Limit Usage",
                f"{pct_safe:.1f}%",
                f"{color} {derived.get('tokens_until_safe_limit', 0):,} remaining"
            )
        
        with col3:
            st.metric(
                "Messages Kept",
                f"{stats.get('messages_kept') or 0:,}",
                f"{(stats.get('messages_truncated') or 0):,f} truncated" if stats.get('messages_truncated') else "No truncation"
            )
        
        with col4:
            st.metric(
                "Model",
                f"Claude {stats.get('model', 'Sonnet')} 4.5",
                f"{stats.get('complexity', 'Simple')} complexity"
            )
        
        st.divider()
        
        # Two columns for charts
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Token Distribution")
            
            if stats.get('system_prompt_tokens') and derived.get('message_tokens'):
                dist_df = pd.DataFrame({
                    'Category': ['System Prompt', 'Message History'],
                    'Tokens': [stats['system_prompt_tokens'], derived['message_tokens']],
                    'Percentage': [derived['system_percentage'], derived['message_percentage']]
                })
                
                dist_chart = alt.Chart(dist_df).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta('Tokens:Q'),
                    color=alt.Color('Category:N', scale=alt.Scale(scheme='category10')),
                    tooltip=['Category', 'Tokens', alt.Tooltip('Percentage:Q', format='.1f')]
                ).properties(height=300)
                
                st.altair_chart(dist_chart, use_container_width=True)
                
                st.caption(f"System: {(stats.get('system_prompt_tokens') or 0):,f} tokens ({derived['system_percentage']:.1f}%)")
                st.caption(f"Messages: {derived['message_tokens']:,} tokens ({derived['message_percentage']:.1f}%)")
            else:
                st.info("Token distribution data not available")
        
        with col_right:
            st.subheader("Capacity Status")
            
            # Hard limit progress
            st.write("**Hard Limit (100,000 tokens)**")
            st.progress(stats.get('percentage_used', 0) / 100)
            st.caption(f"{(stats.get('final_token_count') or 0):,f} / {(stats.get('token_limit') or 100000):,f} ({stats['percentage_used']:.1f}%)")
            
            # Safe limit progress
            st.write("**Safe Limit (85,000 tokens)**")
            pct_safe = derived.get('percentage_of_safe_limit', 0)
            st.progress(min(pct_safe / 100, 1.0))
            st.caption(f"{(stats.get('final_token_count') or 0):,f} / {(stats.get('safe_limit') or 85000):,f} ({pct_safe:.1f}%)")
            
            # Remaining capacity
            st.write("**Remaining Capacity**")
            remaining_safe = derived.get('tokens_until_safe_limit', 0)
            remaining_hard = derived.get('tokens_until_hard_limit', 0)
            st.info(f"Until safe limit: {remaining_safe:,} tokens")
            st.info(f"Until hard limit: {remaining_hard:,} tokens")
        
        st.divider()
        
        # Message truncation details
        if stats.get('messages_total'):
            st.subheader("Message Truncation Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Messages", f"{stats['messages_total']:,}")
            with col2:
                st.metric("Messages Kept", f"{stats['messages_kept']:,}")
            with col3:
                trunc = stats.get('messages_truncated', 0)
                st.metric("Messages Dropped", f"{trunc:,}", 
                         f"{derived.get('truncation_rate', 0):.1f}%")
            
            if stats.get('tokens_before_truncation'):
                st.caption(f"Pre-truncation: {stats['tokens_before_truncation']:,} tokens")
                st.caption(f"Post-truncation: {derived.get('message_tokens', 0):,} tokens")
        
        st.divider()
        
        # Health assessment
        st.subheader("Health Assessment")
        
        pct_safe = derived.get('percentage_of_safe_limit', 0)
        
        if pct_safe > 90:
            st.error("🔴 Critical: Approaching token limit. Consider starting a new conversation.")
        elif pct_safe > 75:
            st.warning("🟡 Caution: High token usage. Monitor closely.")
        elif pct_safe > 50:
            st.info("🟡 Normal: Moderate token usage.")
        else:
            st.success("🟢 Healthy: Plenty of capacity remaining.")
        
        # Additional details
        with st.expander("Technical Details"):
            # Format timestamp with timezone and time ago
            raw_ts = stats.get('timestamp', 'N/A')
            formatted_ts = format_timestamp_with_timezone(raw_ts)
            time_ago = get_time_ago(raw_ts)
            emoji, freshness, color = get_data_freshness_indicator(raw_ts)
            
            if time_ago:
                st.write(f"**Data Timestamp:** {formatted_ts}")
                st.write(f"**Data Age:** {time_ago} {emoji} ({freshness})")
            else:
                st.write(f"**Data Timestamp:** {formatted_ts}")
            st.write(f"**Model:** Claude {stats.get('model', 'Sonnet')} 4.5")
            st.write(f"**Complexity:** {stats.get('complexity', 'Simple')}")
            st.write(f"**Tools Available:** {stats.get('tool_count', 'N/A')}")
            st.write(f"**Tools Metadata:** ~{(stats.get('tools_tokens') or 0):,f} tokens")
            if derived.get('tokens_per_message'):
                st.write(f"**Avg Tokens/Message:** {derived['tokens_per_message']:.0f}")
            
            log_file = find_log_file()
            if log_file:
                st.write(f"**Log Source:** {log_file}")
    else:
        st.info("📊 No conversation data available yet")
        st.write("Conversation statistics will appear here once the backend processes requests.")
        st.write("This data comes from live streaming logs and updates in real-time.")
        
        with st.expander("What will be shown here?"):
            st.write("- **Token Usage**: Current token count vs limits")
            st.write("- **Token Distribution**: System prompt vs message history")
            st.write("- **Capacity Status**: Visual progress bars")
            st.write("- **Message Truncation**: How many messages were dropped")
            st.write("- **Health Assessment**: System health indicators")


# Footer
st.divider()
_now_mt  = datetime.now(tz=MT)
now_str  = _now_mt.strftime('%B %-d, %Y %-I:%M:%S %p')
tz_abbr  = _now_mt.strftime('%Z')   # 'MST' or 'MDT'
st.caption(f"🔄 Dashboard refreshed: {now_str} {tz_abbr} | Database: {DB_CONFIG['host']}:{DB_CONFIG['port']} | Auto-refresh: {'ON' if auto_refresh else 'OFF'}")

# Auto-refresh logic at the end
if auto_refresh:
    import time
    time.sleep(10)
    st.rerun()
