# 🤖 DishChat Token Usage Tracker

Real-time monitoring dashboard for AI agent token usage, tracking actual conversations from the DishChat PostgreSQL database.

## 📊 What This Tracks

### Real Agent Data
- **Chat IDs**: Unique identifiers for each conversation
- **Agent Types**: Which agent handled the conversation (agentic_rag, title_gen, etc.)
- **Users**: Who initiated the conversation (anonymized)
- **Models**: Which AI model was used (Claude Sonnet 4.5, Opus 4, etc.)
- **Token Usage**: Input, output, and cache token statistics
- **Tasks**: Type of operation (chat, tool_call, embedding, etc.)

### Data Source
✅ **REAL DATA** from production DishChat database  
✅ PostgreSQL `usage_tracking` table  
✅ Joined with `chat` table for agent identifiers  
✅ Live tracking of all AI agent conversations  

## 🚀 Quick Start

### One-Command Launch
```bash
cd ~/Token_Tracker
bash start_tracker.sh
```

This will:
1. Install dependencies (streamlit, pandas, plotly, psycopg2)
2. Extract real usage data from database (last 500 records)
3. Launch the dashboard at http://localhost:8501

### Manual Steps

1. **Install Dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Extract Real Data**
   ```bash
   python3 extract_real_data.py --limit 500
   ```

3. **Launch Dashboard**
   ```bash
   streamlit run app.py
   ```

## 📈 Dashboard Features

### Real-Time Metrics
- Total Input Tokens across all conversations
- Total Output Tokens generated
- Cache Hit Tokens (reducing costs)
- Average Tokens per Message

### Interactive Charts
1. **Token Usage Timeline** - Track usage trends over time
2. **Token Distribution** - Compare input vs output vs cache
3. **Cumulative Usage** - Total token accumulation
4. **Message Type Distribution** - Agent vs Tool vs User messages

### Agent Identification
Each log entry includes:
```
2026-05-08 08:00:00 INFO [agentic_rag] chat_id: abc-123-def, user: john, 
message_type: agent, model: claude-sonnet-4.5-v2:0, input_tokens: 1500, ...
```

## 🔧 Configuration

### Database Connection
Edit `extract_real_data.py` to change database settings:
```python
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 5432,
    'database': 'dishchat',
    'user': 'dev_user',
    'password': 'dev123'
}
```

### Dashboard Settings
Adjust in the sidebar:
- **Log File Path**: Point to different log files
- **Refresh Interval**: How often to update (1-30 seconds)
- **Max Data Points**: Buffer size (20-200 messages)

## 🔄 Continuous Monitoring

### Option 1: Auto-Refresh Dashboard
The dashboard auto-refreshes every 5 seconds by default. Just keep it open.

### Option 2: Continuous Data Export
```bash
python3 extract_real_data.py --continuous --interval 30
```
This updates the log file every 30 seconds with new database records.

### Option 3: Both Together
Terminal 1:
```bash
python3 extract_real_data.py --continuous --interval 30
```

Terminal 2:
```bash
streamlit run app.py
```

## 📍 File Structure

```
~/Token_Tracker/
├── app.py                    # Streamlit dashboard
├── extract_real_data.py      # Database extraction script
├── start_tracker.sh           # One-command launcher
├── requirements.txt           # Python dependencies
├── agent_usage.log           # Generated token usage log
├── README.md                 # This file
└── QUICKSTART.md             # Quick reference guide
```

## 🎯 Use Cases

### Cost Monitoring
- Track total tokens per day/week
- Identify expensive operations
- Calculate API costs (tokens × price per token)

### Performance Optimization
- Find high-token conversations
- Identify inefficient prompts
- Optimize agent strategies

### Agent Analysis
- Compare performance across agents
- Track which agents use most tokens
- Analyze cache hit rates

### User Insights
- See which users generate most traffic
- Identify power users
- Understand usage patterns

## 🔍 Extracting Specific Data

### Last N Records
```bash
python3 extract_real_data.py --limit 1000
```

### Continuous Export
```bash
python3 extract_real_data.py --continuous --interval 60
```

### Custom Output Location
```bash
python3 extract_real_data.py --output /path/to/custom.log
```

## 📊 Example Log Format

```
2026-05-08 12:34:56 INFO [agentic_rag] chat_id: 550e8400-e29b-41d4-a716-446655440000, 
user: john, message_type: agent, model: claude-sonnet-4.5-v2:0, task: chat, 
input_tokens: 1847, output_tokens: 623, total_tokens: 2470, 
cache_read_tokens: 342, cache_create_tokens: 89
```

## 🛠️ Troubleshooting

### Database Connection Failed
**Issue**: `psql: connection refused`

**Solutions**:
1. Check PostgreSQL is running: `systemctl status postgresql`
2. Verify port: default 5432
3. Test connection: 
   ```bash
   psql -h 127.0.0.1 -p 5432 -U dev_user -d dishchat
   ```

### No Data Appearing
**Issue**: Dashboard shows "Waiting for data"

**Solutions**:
1. Verify log file exists: `ls -lh ~/Token_Tracker/agent_usage.log`
2. Check file has entries: `wc -l ~/Token_Tracker/agent_usage.log`
3. Re-extract data: `python3 extract_real_data.py --limit 500`

### Port Already in Use
**Issue**: `Port 8501 is already in use`

**Solutions**:
1. Kill existing streamlit: `pkill -f streamlit`
2. Or use different port: `streamlit run app.py --server.port 8502`

## 💡 Pro Tips

1. **Set up log rotation** to prevent log files from growing too large
2. **Run continuous export** to always have fresh data
3. **Bookmark the dashboard** for quick access
4. **Use date filters** in queries for specific time ranges
5. **Export charts** using Plotly's built-in download feature

## 📞 Support

- Check logs: `tail -f ~/Token_Tracker/agent_usage.log`
- Verify database: `python3 extract_real_data.py --limit 10`
- Test dashboard: `streamlit run app.py`

## ✨ What Makes This Better

### vs Simulated Data
✅ **Real conversations** from production  
✅ **Actual token usage** not estimates  
✅ **Real agent identifiers** for traceability  

### vs Basic Logging
✅ **Visual analytics** with interactive charts  
✅ **Real-time updates** every 5 seconds  
✅ **Agent identification** to track which agent did what  
✅ **Cost analysis** with token breakdowns  

---

**Created for tracking REAL DishChat AI agent token usage**  
**Last updated: 2026-05-08**
