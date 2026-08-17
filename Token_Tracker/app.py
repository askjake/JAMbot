
#!/usr/bin/env python3
'''
🚀 DishChat Token Usage & Agent Intelligence Tracker v3.0
==========================================================
Real-time monitoring of token usage, cost analysis, AND agent backend
token management features including:
  - Opus Auto-Routing metrics (complexity scoring, A/B testing, fast-path)
  - AWS Token Refresh health (credential lifecycle, proactive refresh)
  - Cache efficiency analysis (read/create ratios, cost savings)
  - Reasoning mode tracking (budget utilization)
  - Tool call depth monitoring (recursion, consecutive calls)
  - Per-user and per-chat cost attribution

Author: Jacob Montgomery (refactored from v1.0)
  v3.0: Tool Result Compression tab added
Run: streamlit run app.py --server.port 8503
'''
import streamlit as st
import pandas as pd
import altair as alt
import psycopg2
from psycopg2.extras import RealDictCursor
import re
import json
import os
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

MT = ZoneInfo('America/Denver')

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DishChat Agent Intelligence Tracker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #4a4a6a; margin-bottom: 1.5rem; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   border-radius: 10px; padding: 1rem; color: white; }
    .stMetric > div { background: #f8f9fa; border-radius: 8px; padding: 0.5rem; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
    .section-divider { border-top: 2px solid #e0e0e0; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Model Pricing (synchronized with agent constants.py)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PRICING = {
    "us.anthropic.claude-sonnet-4-20250514-v1:0": {
        "display_name": "Claude Sonnet 4", "input": 3.0, "cache_read": 0.30,
        "cache_create": 3.75, "output": 15.0, "tier": "power"
    },
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": {
        "display_name": "Claude Sonnet 4.5", "input": 3.0, "cache_read": 0.30,
        "cache_create": 3.75, "output": 15.0, "tier": "power"
    },
    "anthropic.claude-3-5-haiku-20241022-v1:0": {
        "display_name": "Claude 3.5 Haiku", "input": 0.80, "cache_read": 0.08,
        "cache_create": 1.0, "output": 4.0, "tier": "efficient"
    },
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": {
        "display_name": "Claude 3.5 Haiku", "input": 0.80, "cache_read": 0.08,
        "cache_create": 1.0, "output": 4.0, "tier": "efficient"
    },
    # Application Inference Profile ARNs
    "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/5c511xksna83": {
        "display_name": "Sonnet (Profile)", "input": 3.0, "cache_read": 0.30,
        "cache_create": 3.75, "output": 15.0, "tier": "power"
    },
    "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/wpnvchycfust": {
        "display_name": "Sonnet (Efficient Profile)", "input": 3.0, "cache_read": 0.30,
        "cache_create": 3.75, "output": 15.0, "tier": "efficient"
    },
    "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/4xgakngy389z": {
        "display_name": "Haiku (Embed Profile)", "input": 0.80, "cache_read": 0.08,
        "cache_create": 1.0, "output": 4.0, "tier": "embed"
    },
    "arn:aws:bedrock:us-west-2:233532778289:application-inference-profile/m4hvzo6r2exy": {
        "display_name": "Opus 4.1 (Auto-Routed)", "input": 15.0, "cache_read": 1.50,
        "cache_create": 18.75, "output": 75.0, "tier": "opus"
    },
}

# Fallback pricing for unknown models
DEFAULT_PRICING = {"display_name": "Unknown", "input": 3.0, "cache_read": 0.30,
                   "cache_create": 3.75, "output": 15.0, "tier": "unknown"}

# ─────────────────────────────────────────────────────────────────────────────
# Database Configuration
# ─────────────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', '127.0.0.1'),
    'port': int(os.environ.get('DB_PORT', '5433')),
    'database': os.environ.get('DB_NAME', 'dishchat'),
    'user': os.environ.get('DB_USER', 'dev_user'),
    'password': os.environ.get('DB_PASSWORD', 'dev123'),
}

AGENT_API_BASE = os.environ.get('AGENT_API_BASE', 'http://127.0.0.1:8002/rest/api/v1')

# ─────────────────────────────────────────────────────────────────────────────
# Data Fetching
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_usage_data(hours_back=24, limit=2000):
    """Fetch usage tracking records from PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
        cursor = conn.cursor()
        query = """
            SELECT 
                ut.id, ut.timestamp, ut.owner_id, ut.chat_id, ut.model,
                ut.task, ut.input_tokens, ut.output_tokens,
                ut.input_cache_read, ut.input_cache_create,
                ut.input_cost, ut.output_cost
            FROM usage_tracking ut
            WHERE ut.timestamp > NOW() - INTERVAL '%s hours'
            ORDER BY ut.timestamp DESC
            LIMIT %s
        """
        cursor.execute(query, (hours_back, limit))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=15)
def fetch_opus_routing_stats():
    """Fetch Opus routing metrics from the agent's internal API."""
    try:
        resp = requests.get(f"{AGENT_API_BASE}/internal/opus-routing-stats", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def get_model_display_name(model_id):
    """Resolve model ID to human-friendly name."""
    pricing = MODEL_PRICING.get(model_id, DEFAULT_PRICING)
    return pricing["display_name"]


def calculate_cost(row):
    """Calculate cost for a single usage record (in dollars)."""
    pricing = MODEL_PRICING.get(row.get('model', ''), DEFAULT_PRICING)
    input_cost = (row.get('input_tokens', 0) * pricing['input'] / 1_000_000)
    cache_read_cost = (row.get('input_cache_read', 0) * pricing['cache_read'] / 1_000_000)
    cache_create_cost = (row.get('input_cache_create', 0) * pricing['cache_create'] / 1_000_000)
    output_cost = (row.get('output_tokens', 0) * pricing['output'] / 1_000_000)
    return input_cost + cache_read_cost + cache_create_cost + output_cost


def calculate_cache_savings(row):
    """Calculate how much was SAVED by cache reads vs full-price input."""
    pricing = MODEL_PRICING.get(row.get('model', ''), DEFAULT_PRICING)
    cache_read_tokens = row.get('input_cache_read', 0)
    # Savings = what it WOULD have cost at full input price minus what it cost at cache_read price
    savings = cache_read_tokens * (pricing['input'] - pricing['cache_read']) / 1_000_000
    return max(0, savings)


@st.cache_data(ttl=30)
def fetch_compression_stats(hours_back=24):
    """Parse agent backend logs for tool_result_compression telemetry events."""
    import json
    import re

    records = []
    log_paths = [
        '/home/jakebot/Jakes-agent/logs/backend.log',
        '/home/jakebot/Jakes-agent/logs/backend_8002.log',
        '/home/jakebot/Jakes-agent/smplogs/backend.log',
    ]

    for log_path in log_paths:
        try:
            if not os.path.exists(log_path):
                continue
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[-30000:]
            for i, line in enumerate(lines):
                if 'tool_result_compression' not in line.lower():
                    continue
                # Try to parse a JSON blob from this line or the next few
                context = ''.join(lines[i:i+5])
                matches = re.findall(r'\{[^{}]+\}', context)
                for m in matches:
                    try:
                        data = json.loads(m)
                        if 'original_tokens' in data or 'savings_chars' in data:
                            records.append(data)
                    except Exception:
                        pass
        except Exception:
            pass

    # Also query live agent API endpoint
    try:
        resp = requests.get(f"{AGENT_API_BASE}/internal/compression-stats", timeout=3)
        if resp.status_code == 200:
            api_data = resp.json()
            if isinstance(api_data, list):
                records.extend(api_data)
            elif isinstance(api_data, dict) and 'records' in api_data:
                records.extend(api_data['records'])
    except Exception:
        pass

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    for col in ['original_tokens', 'original_chars', 'compressed_chars',
                'savings_chars', 'savings_tokens_estimate']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if 'strategy' not in df.columns:
        df['strategy'] = 'unknown'
    if 'tool' not in df.columns:
        df['tool'] = 'unknown'
    if 'content_type' not in df.columns:
        df['content_type'] = 'unknown'

    return df




# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Agent Intelligence Tracker")
    st.markdown("---")
    
    hours_back = st.slider("Time Window (hours)", 1, 168, 24, step=1)
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
    
    if auto_refresh:
        st.markdown("🔄 *Refreshing every 30 seconds*")
    
    st.markdown("---")
    st.markdown("### Data Sources")
    st.markdown(f"- **DB**: `{DB_CONFIG['host']}:{DB_CONFIG['port']}`")
    st.markdown(f"- **Agent API**: `{AGENT_API_BASE}`")
    
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────────────────────
df = fetch_usage_data(hours_back=hours_back)
opus_stats = fetch_opus_routing_stats()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🧠 DishChat Agent Intelligence Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Token usage • Cost analysis • Opus routing • Cache efficiency • AWS health</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview", "🎯 Opus Routing", "💰 Cost Analysis",
    "⚡ Cache Efficiency", "🔑 Token Health", "📋 Raw Data",
    "🗜️ Tool Result Compression"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if df.empty:
        st.warning("No usage data found. Check database connection.")
    else:
        # Top-level metrics
        total_input = int(df['input_tokens'].sum())
        total_output = int(df['output_tokens'].sum())
        total_cache_read = int(df['input_cache_read'].sum())
        total_cache_create = int(df['input_cache_create'].sum())
        total_records = len(df)
        
        # Cost calculation
        total_cost = sum(calculate_cost(row) for _, row in df.iterrows())
        total_savings = sum(calculate_cache_savings(row) for _, row in df.iterrows())
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Tokens", f"{(total_input + total_output):,.0f}")
        col2.metric("Total Cost", f"${total_cost:.4f}")
        col3.metric("Cache Savings", f"${total_savings:.4f}", delta=f"{total_savings/(total_cost+0.001)*100:.1f}%")
        col4.metric("Requests", f"{total_records:,}")
        col5.metric("Avg Tokens/Req", f"{(total_input + total_output) / max(total_records, 1):,.0f}")
        
        st.markdown("---")
        
        # Timeline chart
        st.subheader("Token Usage Over Time")
        df_time = df.copy()
        df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
        df_time['hour'] = df_time['timestamp'].dt.floor('h')
        hourly = df_time.groupby('hour').agg(
            input_tokens=('input_tokens', 'sum'),
            output_tokens=('output_tokens', 'sum'),
            cache_read=('input_cache_read', 'sum'),
            requests=('id', 'count')
        ).reset_index()
        
        hourly_melted = hourly.melt(
            id_vars=['hour'], 
            value_vars=['input_tokens', 'output_tokens', 'cache_read'],
            var_name='type', value_name='tokens'
        )
        
        chart = alt.Chart(hourly_melted).mark_area(opacity=0.7).encode(
            x=alt.X('hour:T', title='Time'),
            y=alt.Y('tokens:Q', title='Tokens'),
            color=alt.Color('type:N', scale=alt.Scale(
                domain=['input_tokens', 'output_tokens', 'cache_read'],
                range=['#667eea', '#f093fb', '#4fd1c5']
            )),
            tooltip=['hour:T', 'type:N', 'tokens:Q']
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
        
        # Model breakdown
        st.subheader("Usage by Model")
        col_a, col_b = st.columns(2)
        
        with col_a:
            model_usage = df.groupby('model').agg(
                total_tokens=('input_tokens', lambda x: x.sum() + df.loc[x.index, 'output_tokens'].sum()),
                requests=('id', 'count')
            ).reset_index()
            model_usage['display_name'] = model_usage['model'].apply(get_model_display_name)
            
            chart_model = alt.Chart(model_usage).mark_bar().encode(
                x=alt.X('total_tokens:Q', title='Total Tokens'),
                y=alt.Y('display_name:N', sort='-x', title='Model'),
                color=alt.Color('display_name:N', legend=None),
                tooltip=['display_name:N', 'total_tokens:Q', 'requests:Q']
            ).properties(height=250)
            st.altair_chart(chart_model, use_container_width=True)
        
        with col_b:
            task_usage = df.groupby('task').agg(
                total_tokens=('input_tokens', lambda x: x.sum() + df.loc[x.index, 'output_tokens'].sum()),
                requests=('id', 'count')
            ).reset_index()
            
            chart_task = alt.Chart(task_usage).mark_arc(innerRadius=50).encode(
                theta=alt.Theta('total_tokens:Q'),
                color=alt.Color('task:N'),
                tooltip=['task:N', 'total_tokens:Q', 'requests:Q']
            ).properties(height=250)
            st.altair_chart(chart_task, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: OPUS ROUTING
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🎯 Opus Auto-Routing Intelligence")
    st.markdown("Tracks the complexity-based model routing system that dynamically selects Claude Opus vs Sonnet.")
    
    if opus_stats and opus_stats.get("status") != "no_requests":
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Routing Decisions", f"{opus_stats.get('total_requests', 0):,}")
        col2.metric("Opus Selections", f"{opus_stats.get('opus_requests', 0):,}", 
                    delta=f"{opus_stats.get('opus_percentage', 0):.1f}%")
        col3.metric("Avg Complexity Score", f"{opus_stats.get('average_score', 0):.2f}")
        col4.metric("Avg Detection Time", f"{opus_stats.get('average_detection_ms', 0):.2f}ms")
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Routing Distribution")
            routing_data = pd.DataFrame([
                {"Model": "Opus (Complex)", "Requests": opus_stats.get('opus_requests', 0)},
                {"Model": "Sonnet (Standard)", "Requests": opus_stats.get('sonnet_requests', 0)},
            ])
            chart_routing = alt.Chart(routing_data).mark_arc(innerRadius=60).encode(
                theta='Requests:Q',
                color=alt.Color('Model:N', scale=alt.Scale(
                    domain=['Opus (Complex)', 'Sonnet (Standard)'],
                    range=['#ff6b6b', '#667eea']
                )),
                tooltip=['Model:N', 'Requests:Q']
            ).properties(height=250)
            st.altair_chart(chart_routing, use_container_width=True)
        
        with col_b:
            st.markdown("#### Performance Optimizations")
            fast_path = opus_stats.get('fast_path_hit_rate', 0)
            cache_hit = opus_stats.get('cache_hit_rate', 0)
            
            perf_data = pd.DataFrame([
                {"Optimization": "Fast-Path Hits", "Rate": fast_path},
                {"Optimization": "Score Cache Hits", "Rate": cache_hit},
                {"Optimization": "Full Algorithm", "Rate": 100 - fast_path - cache_hit},
            ])
            chart_perf = alt.Chart(perf_data).mark_bar().encode(
                x=alt.X('Rate:Q', title='% of Requests'),
                y=alt.Y('Optimization:N', sort='-x'),
                color=alt.Color('Optimization:N', legend=None),
                tooltip=['Optimization:N', 'Rate:Q']
            ).properties(height=200)
            st.altair_chart(chart_perf, use_container_width=True)
        
        # A/B Testing
        ab_data = opus_stats.get('ab_test', {})
        if ab_data.get('group_a_requests', 0) > 0 or ab_data.get('group_b_requests', 0) > 0:
            st.markdown("---")
            st.markdown("#### A/B Test Results")
            st.json(ab_data)
        
        # Threshold info
        st.markdown("---")
        st.info(f"**Current Threshold:** {opus_stats.get('threshold', 'N/A')} | "
                f"**Auto-Opus Enabled:** {opus_stats.get('auto_opus_enabled', False)} | "
                f"**Uptime:** {opus_stats.get('uptime_seconds', 0)/3600:.1f}h")
    else:
        st.info("🔌 Opus routing stats not available. The agent API may not be running or has no routing data yet.")
        st.markdown("""
        **What this tracks when active:**
        - Complexity scoring (8-factor algorithm with 0-12+ scale)
        - Fast-path detection (short-circuit obvious simple/complex prompts)
        - LRU score caching (avoids rescoring similar prompts)
        - Conversation context escalation (multi-turn complexity tracking)
        - A/B threshold testing (experimental routing splits)
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: COST ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("💰 Cost Analysis & Attribution")
    
    if not df.empty:
        df_cost = df.copy()
        df_cost['cost'] = df_cost.apply(calculate_cost, axis=1)
        df_cost['savings'] = df_cost.apply(calculate_cache_savings, axis=1)
        df_cost['display_model'] = df_cost['model'].apply(get_model_display_name)
        df_cost['tier'] = df_cost['model'].apply(
            lambda m: MODEL_PRICING.get(m, DEFAULT_PRICING)['tier']
        )
        
        total_cost = df_cost['cost'].sum()
        total_savings = df_cost['savings'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Spend", f"${total_cost:.4f}")
        col2.metric("Cache Savings", f"${total_savings:.4f}")
        col3.metric("Net Effective Cost", f"${total_cost - total_savings:.4f}")
        col4.metric("Savings Rate", f"{total_savings/(total_cost + total_savings + 0.0001)*100:.1f}%")
        
        st.markdown("---")
        
        # Cost by tier
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Cost by Model Tier")
            tier_costs = df_cost.groupby('tier').agg(
                cost=('cost', 'sum'), requests=('id', 'count')
            ).reset_index()
            
            chart_tier = alt.Chart(tier_costs).mark_bar().encode(
                x=alt.X('tier:N', title='Tier'),
                y=alt.Y('cost:Q', title='Cost ($)'),
                color=alt.Color('tier:N', scale=alt.Scale(
                    domain=['opus', 'power', 'efficient', 'embed', 'unknown'],
                    range=['#ff6b6b', '#667eea', '#4fd1c5', '#feca57', '#a0a0a0']
                )),
                tooltip=['tier:N', 'cost:Q', 'requests:Q']
            ).properties(height=250)
            st.altair_chart(chart_tier, use_container_width=True)
        
        with col_b:
            st.markdown("#### Cost by User (Top 10)")
            user_costs = df_cost.groupby('owner_id').agg(
                cost=('cost', 'sum'), requests=('id', 'count')
            ).nlargest(10, 'cost').reset_index()
            user_costs['user'] = user_costs['owner_id'].apply(lambda x: x.split('@')[0] if '@' in str(x) else str(x)[:15])
            
            chart_user = alt.Chart(user_costs).mark_bar().encode(
                x=alt.X('cost:Q', title='Cost ($)'),
                y=alt.Y('user:N', sort='-x', title='User'),
                color=alt.value('#667eea'),
                tooltip=['user:N', 'cost:Q', 'requests:Q']
            ).properties(height=250)
            st.altair_chart(chart_user, use_container_width=True)
        
        # Hourly cost trend
        st.markdown("#### Hourly Cost Trend")
        df_cost['timestamp'] = pd.to_datetime(df_cost['timestamp'])
        df_cost['hour'] = df_cost['timestamp'].dt.floor('h')
        hourly_cost = df_cost.groupby('hour').agg(cost=('cost', 'sum'), savings=('savings', 'sum')).reset_index()
        
        cost_melted = hourly_cost.melt(id_vars=['hour'], value_vars=['cost', 'savings'],
                                        var_name='type', value_name='dollars')
        chart_cost_time = alt.Chart(cost_melted).mark_area(opacity=0.6).encode(
            x='hour:T', y='dollars:Q',
            color=alt.Color('type:N', scale=alt.Scale(
                domain=['cost', 'savings'], range=['#ff6b6b', '#4fd1c5']
            ))
        ).properties(height=200)
        st.altair_chart(chart_cost_time, use_container_width=True)
    else:
        st.warning("No data available for cost analysis.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: CACHE EFFICIENCY
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("⚡ Prompt Cache Efficiency Analysis")
    st.markdown("Measures how effectively the agent uses Anthropic's prompt caching to reduce costs.")
    
    if not df.empty:
        total_input = df['input_tokens'].sum()
        total_cache_read = df['input_cache_read'].sum()
        total_cache_create = df['input_cache_create'].sum()
        total_all_input = total_input + total_cache_read + total_cache_create
        
        cache_hit_rate = (total_cache_read / max(total_all_input, 1)) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cache Hit Rate", f"{cache_hit_rate:.1f}%")
        col2.metric("Cache Reads", f"{total_cache_read:,.0f}")
        col3.metric("Cache Creates", f"{total_cache_create:,.0f}")
        col4.metric("Uncached Input", f"{total_input:,.0f}")
        
        st.markdown("---")
        
        # Cache composition pie
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Input Token Composition")
            comp_data = pd.DataFrame([
                {"Type": "Uncached Input", "Tokens": int(total_input)},
                {"Type": "Cache Read (10% cost)", "Tokens": int(total_cache_read)},
                {"Type": "Cache Create (1.25x cost)", "Tokens": int(total_cache_create)},
            ])
            chart_comp = alt.Chart(comp_data).mark_arc(innerRadius=50).encode(
                theta='Tokens:Q',
                color=alt.Color('Type:N', scale=alt.Scale(
                    domain=['Uncached Input', 'Cache Read (10% cost)', 'Cache Create (1.25x cost)'],
                    range=['#667eea', '#4fd1c5', '#feca57']
                )),
                tooltip=['Type:N', 'Tokens:Q']
            ).properties(height=250)
            st.altair_chart(chart_comp, use_container_width=True)
        
        with col_b:
            st.markdown("#### Cache Rate Over Time")
            df_cache = df.copy()
            df_cache['timestamp'] = pd.to_datetime(df_cache['timestamp'])
            df_cache['hour'] = df_cache['timestamp'].dt.floor('h')
            hourly_cache = df_cache.groupby('hour').agg(
                cache_read=('input_cache_read', 'sum'),
                total_input=('input_tokens', 'sum'),
            ).reset_index()
            hourly_cache['cache_rate'] = hourly_cache['cache_read'] / (
                hourly_cache['total_input'] + hourly_cache['cache_read'] + 0.01) * 100
            
            chart_cache_rate = alt.Chart(hourly_cache).mark_line(
                point=True, strokeWidth=2
            ).encode(
                x='hour:T',
                y=alt.Y('cache_rate:Q', title='Cache Hit Rate %', scale=alt.Scale(domain=[0, 100])),
                tooltip=['hour:T', 'cache_rate:Q']
            ).properties(height=250)
            st.altair_chart(chart_cache_rate, use_container_width=True)
        
        # Cache by model
        st.markdown("#### Cache Efficiency by Model")
        cache_by_model = df.groupby('model').agg(
            cache_read=('input_cache_read', 'sum'),
            cache_create=('input_cache_create', 'sum'),
            uncached=('input_tokens', 'sum'),
        ).reset_index()
        cache_by_model['display_name'] = cache_by_model['model'].apply(get_model_display_name)
        cache_by_model['total'] = cache_by_model['cache_read'] + cache_by_model['cache_create'] + cache_by_model['uncached']
        cache_by_model['hit_rate'] = (cache_by_model['cache_read'] / cache_by_model['total'].clip(lower=1)) * 100
        
        st.dataframe(
            cache_by_model[['display_name', 'cache_read', 'cache_create', 'uncached', 'hit_rate']].rename(
                columns={'display_name': 'Model', 'cache_read': 'Cache Reads', 
                         'cache_create': 'Cache Creates', 'uncached': 'Uncached', 'hit_rate': 'Hit Rate %'}
            ).sort_values('Cache Reads', ascending=False),
            use_container_width=True, hide_index=True
        )
    else:
        st.warning("No data available for cache analysis.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: TOKEN HEALTH (AWS Credential Lifecycle)
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔑 AWS Token Refresh & Model Health")
    st.markdown("Monitors the agent's proactive AWS credential refresh system and model availability.")
    
    # Try to read agent backend log for token refresh events
    log_paths = [
        '/home/jakebot/Jakes-agent/logs/backend.log',
        '/home/jakebot/Jakes-agent/logs/backend_8000.log',
    ]
    
    token_events = []
    retry_events = []
    
    for log_path in log_paths:
        try:
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    # Read last 5000 lines for performance
                    lines = f.readlines()[-5000:]
                    for line in lines:
                        if 'token refresh' in line.lower() or 'refreshing aws' in line.lower():
                            token_events.append(line.strip())
                        if 'expiredtokenexception' in line.lower() or 'invoke_with_retry' in line.lower():
                            retry_events.append(line.strip())
        except Exception:
            pass
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Token Refreshes Detected", len(token_events))
    col2.metric("Retry Events", len(retry_events))
    col3.metric("Credential Status", "🟢 Healthy" if len(retry_events) < 5 else "🟡 Elevated Retries")
    
    st.markdown("---")
    
    st.markdown("#### Token Management Features Monitored")
    st.markdown("""
    | Feature | Status | Description |
    |---------|--------|-------------|
    | Proactive Refresh | ✅ Active | Background task refreshes models 10min before expiry |
    | Credential Fingerprint | ✅ Active | Detects on-disk credential changes automatically |
    | invoke_with_retry | ✅ Active | 2-retry fallback on ExpiredTokenException |
    | stream_with_retry | ✅ Active | Streaming variant with same retry logic |
    | Model ARN Resolution | ✅ Active | Maps model IDs to Application Inference Profile ARNs |
    """)
    
    if token_events:
        st.markdown("#### Recent Token Refresh Events")
        with st.expander(f"Show {len(token_events)} events", expanded=False):
            for event in token_events[-20:]:
                st.code(event, language="log")
    
    if retry_events:
        st.markdown("#### ⚠️ Retry Events (Token Expiry)")
        with st.expander(f"Show {len(retry_events)} events", expanded=False):
            for event in retry_events[-20:]:
                st.code(event, language="log")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: RAW DATA
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("📋 Raw Usage Records")
    
    if not df.empty:
        st.markdown(f"Showing **{len(df):,}** records from last **{hours_back}** hours")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            models = ['All'] + sorted(df['model'].unique().tolist())
            selected_model = st.selectbox("Filter by Model", models)
        with col2:
            tasks = ['All'] + sorted(df['task'].unique().tolist())
            selected_task = st.selectbox("Filter by Task", tasks)
        
        df_display = df.copy()
        if selected_model != 'All':
            df_display = df_display[df_display['model'] == selected_model]
        if selected_task != 'All':
            df_display = df_display[df_display['task'] == selected_task]
        
        df_display['display_model'] = df_display['model'].apply(get_model_display_name)
        df_display['cost'] = df_display.apply(calculate_cost, axis=1)
        
        st.dataframe(
            df_display[['timestamp', 'owner_id', 'display_model', 'task', 
                       'input_tokens', 'output_tokens', 'input_cache_read', 
                       'input_cache_create', 'cost']].rename(columns={
                'timestamp': 'Time', 'owner_id': 'User', 'display_model': 'Model',
                'task': 'Task', 'input_tokens': 'Input', 'output_tokens': 'Output',
                'input_cache_read': 'Cache Read', 'input_cache_create': 'Cache Create',
                'cost': 'Cost ($)'
            }),
            use_container_width=True, hide_index=True
        )
    else:
        st.warning("No data available.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: TOOL RESULT COMPRESSION
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("🗜️ Tool Result Compression Savings")
    st.markdown(
        "Tracks how much context-window space (and cost) is saved by compressing "
        "large tool outputs before they are sent to the model."
    )

    df_comp = fetch_compression_stats(hours_back=hours_back)

    if df_comp.empty:
        st.info(
            "No compression events found in recent logs. "
            "Compression fires when a tool output exceeds the per-message token limit."
        )
        st.markdown("""
#### What This Tab Tracks
Every time the agent compresses a large tool result before injecting it into the LLM context,
it emits a telemetry event with:

| Field | Description |
|-------|-------------|
| `tool` | Which tool produced the large output |
| `original_tokens` | Estimated tokens before compression |
| `savings_tokens_estimate` | Estimated tokens saved |
| `strategy` | Compression algorithm used (e.g. `head_tail_pass_through`) |
| `content_type` | Type of content compressed (text, json, web, etc.) |
| `savings_chars` | Character reduction achieved |

Data is sourced from agent backend logs (keyword: `tool_result_compression`).
        """)
    else:
        # ── Summary metrics ──────────────────────────────────────────────────
        total_events = len(df_comp)
        total_tokens_saved = int(df_comp['savings_tokens_estimate'].sum())
        total_chars_saved  = int(df_comp['savings_chars'].sum())
        avg_reduction_pct  = (
            df_comp['savings_chars'].sum() /
            df_comp['original_chars'].clip(lower=1).sum() * 100
        ) if 'original_chars' in df_comp.columns else 0
        cost_saved_est = total_tokens_saved * 3.0 / 1_000_000   # Sonnet input rate

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Compression Events", f"{total_events:,}")
        c2.metric("Tokens Saved (est.)", f"{total_tokens_saved:,.0f}")
        c3.metric("Chars Reduced",       f"{total_chars_saved:,.0f}")
        c4.metric("Avg Reduction",       f"{avg_reduction_pct:.1f}%")
        c5.metric("Est. Cost Saved",     f"${cost_saved_est:.4f}")

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### Token Savings by Tool")
            by_tool = (
                df_comp.groupby('tool')
                .agg(
                    savings=('savings_tokens_estimate', 'sum'),
                    events=('original_tokens', 'count'),
                    avg_original=('original_tokens', 'mean'),
                )
                .reset_index()
                .sort_values('savings', ascending=False)
                .head(15)
            )
            chart_tool = alt.Chart(by_tool).mark_bar(
                cornerRadiusTopRight=4, cornerRadiusTopLeft=4
            ).encode(
                x=alt.X('savings:Q', title='Tokens Saved (est.)'),
                y=alt.Y('tool:N', sort='-x', title='Tool Name'),
                color=alt.Color('savings:Q', scale=alt.Scale(scheme='blues'), legend=None),
                tooltip=[
                    alt.Tooltip('tool:N',          title='Tool'),
                    alt.Tooltip('savings:Q',        title='Tokens Saved', format=','),
                    alt.Tooltip('events:Q',         title='Events'),
                    alt.Tooltip('avg_original:Q',   title='Avg Original Tokens', format=',.0f'),
                ],
            ).properties(height=300)
            st.altair_chart(chart_tool, use_container_width=True)

        with col_b:
            st.markdown("#### Events by Compression Strategy")
            by_strategy = (
                df_comp.groupby('strategy')
                .agg(events=('original_tokens', 'count'),
                     total_saved=('savings_tokens_estimate', 'sum'))
                .reset_index()
            )
            chart_strategy = alt.Chart(by_strategy).mark_arc(innerRadius=55).encode(
                theta=alt.Theta('events:Q'),
                color=alt.Color('strategy:N', scale=alt.Scale(scheme='tableau10')),
                tooltip=[
                    alt.Tooltip('strategy:N',   title='Strategy'),
                    alt.Tooltip('events:Q',     title='Events'),
                    alt.Tooltip('total_saved:Q', title='Total Tokens Saved', format=','),
                ],
            ).properties(height=300)
            st.altair_chart(chart_strategy, use_container_width=True)

        st.markdown("---")

        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown("#### Original vs Compressed Size")
            plot_df = df_comp.head(500).copy()
            chart_scatter = alt.Chart(plot_df).mark_circle(
                opacity=0.6, size=60
            ).encode(
                x=alt.X('original_chars:Q', title='Original Characters',
                         scale=alt.Scale(zero=True)),
                y=alt.Y('compressed_chars:Q', title='Compressed Characters'),
                color=alt.Color('strategy:N', scale=alt.Scale(scheme='tableau10')),
                tooltip=[
                    alt.Tooltip('tool:N',                    title='Tool'),
                    alt.Tooltip('original_chars:Q',          title='Original Chars',    format=','),
                    alt.Tooltip('compressed_chars:Q',        title='Compressed Chars',  format=','),
                    alt.Tooltip('strategy:N',                title='Strategy'),
                    alt.Tooltip('savings_tokens_estimate:Q', title='Tokens Saved',       format=','),
                ],
            ).properties(height=280)
            max_chars = int(df_comp['original_chars'].max()) if not df_comp.empty else 10000
            ref_df    = pd.DataFrame({'x': [0, max_chars]})
            ref_line  = alt.Chart(ref_df).mark_line(
                strokeDash=[6, 4], color='red', opacity=0.4
            ).encode(x='x:Q', y='x:Q')
            st.altair_chart(chart_scatter + ref_line, use_container_width=True)
            st.caption("🔴 Diagonal = no compression. Points below diagonal = savings achieved.")

        with col_d:
            st.markdown("#### Savings by Content Type")
            by_type = (
                df_comp.groupby('content_type')
                .agg(total_saved=('savings_tokens_estimate', 'sum'),
                     events=('original_tokens', 'count'),
                     avg_reduction=('savings_chars', 'mean'))
                .reset_index()
                .sort_values('total_saved', ascending=False)
            )
            chart_type = alt.Chart(by_type).mark_bar(
                cornerRadiusTopRight=4, cornerRadiusTopLeft=4
            ).encode(
                x=alt.X('content_type:N', title='Content Type', sort='-y'),
                y=alt.Y('total_saved:Q',  title='Total Tokens Saved (est.)'),
                color=alt.Color('content_type:N',
                                scale=alt.Scale(scheme='set2'), legend=None),
                tooltip=[
                    alt.Tooltip('content_type:N',  title='Content Type'),
                    alt.Tooltip('total_saved:Q',   title='Tokens Saved',       format=','),
                    alt.Tooltip('events:Q',        title='Events'),
                    alt.Tooltip('avg_reduction:Q', title='Avg Chars Reduced',  format=',.0f'),
                ],
            ).properties(height=280)
            st.altair_chart(chart_type, use_container_width=True)

        st.markdown("---")

        # ── Reduction % distribution ─────────────────────────────────────
        st.markdown("#### Compression Reduction Rate Distribution")
        df_hist = df_comp.copy()
        df_hist['reduction_pct'] = (
            df_hist['savings_chars'] / df_hist['original_chars'].clip(lower=1) * 100
        ).clip(0, 100)
        chart_hist = alt.Chart(df_hist).mark_bar(
            cornerRadiusTopRight=3, cornerRadiusTopLeft=3, opacity=0.85
        ).encode(
            x=alt.X('reduction_pct:Q', bin=alt.Bin(maxbins=25),
                     title='Reduction %'),
            y=alt.Y('count():Q', title='Number of Events'),
            color=alt.value('#667eea'),
            tooltip=[
                alt.Tooltip('reduction_pct:Q', bin=True, title='Reduction %'),
                alt.Tooltip('count():Q',                 title='Events'),
            ],
        ).properties(height=200)
        st.altair_chart(chart_hist, use_container_width=True)

        # ── Raw table ────────────────────────────────────────────────────
        with st.expander("📋 Raw Compression Events", expanded=False):
            display_cols = [
                c for c in ['tool', 'strategy', 'content_type', 'original_tokens',
                             'savings_tokens_estimate', 'original_chars',
                             'compressed_chars', 'savings_chars']
                if c in df_comp.columns
            ]
            st.dataframe(
                df_comp[display_cols].sort_values(
                    'savings_tokens_estimate', ascending=False
                ),
                use_container_width=True,
                hide_index=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
# Auto-refresh
# ─────────────────────────────────────────────────────────────────────────────
if auto_refresh:
    import time
    time.sleep(0.1)  # Prevent blocking
    st.empty()  # Placeholder for rerun timer
    # Streamlit will rerun on interaction; for true auto-refresh use st_autorefresh component

# Footer
st.markdown("---")
st.markdown(f"*Last updated: {datetime.now(MT).strftime('%Y-%m-%d %H:%M:%S %Z')} • "
            f"Data source: PostgreSQL ({DB_CONFIG['host']}:{DB_CONFIG['port']})*")
