"""
Thought Visualization Router - REFACTORED
Serves the AI thought visualization UI with REAL DATA from database
Fixes: Shows demo data instead of live data
"""
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

from app.dependencies import DBSessionDep, UserEmailDep
from app.usage_tracking.models import UsageTracking
from app.message.models import MessageMD
from app.chat.models import Chat
# Using UserEmailDep from app.dependencies instead

logger = logging.getLogger(__name__)

router = APIRouter(tags=["visualization"])

# HTML template for the visualization (same as before)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Thought Visualization - LIVE</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a0a2e 100%);
            color: #00ff00;
            overflow: hidden;
        }
        
        .header {
            background: rgba(255, 0, 255, 0.1);
            border-bottom: 2px solid #ff00ff;
            padding: 15px;
            text-align: center;
            box-shadow: 0 0 20px rgba(255, 0, 255, 0.5);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 20px rgba(255, 0, 255, 0.5); }
            50% { box-shadow: 0 0 40px rgba(255, 0, 255, 0.8); }
        }
        
        h1 {
            color: #ff00ff;
            text-shadow: 0 0 10px #ff00ff;
            font-size: 2em;
            letter-spacing: 3px;
        }
        
        .status {
            color: #00ff00;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .container {
            display: grid;
            grid-template-columns: 2fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 10px;
            padding: 10px;
            height: calc(100vh - 100px);
        }
        
        .panel {
            background: rgba(0, 0, 0, 0.7);
            border: 1px solid #00ff00;
            border-radius: 5px;
            padding: 15px;
            overflow: auto;
            box-shadow: 0 0 15px rgba(0, 255, 0, 0.3);
        }
        
        .panel h2 {
            color: #00ffff;
            border-bottom: 1px solid #00ffff;
            padding-bottom: 5px;
            margin-bottom: 10px;
            font-size: 1.2em;
        }
        
        #graph { grid-row: 1 / 3; }
        
        #graph svg {
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 5px;
        }
        
        .node circle {
            stroke: #fff;
            stroke-width: 2px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .node:hover circle {
            r: 12;
            filter: drop-shadow(0 0 10px currentColor);
        }
        
        .node text {
            font-size: 10px;
            fill: #fff;
            text-anchor: middle;
            pointer-events: none;
        }
        
        .link {
            stroke: #00ff00;
            stroke-opacity: 0.6;
            stroke-width: 2px;
            fill: none;
        }
        
        .event-item {
            padding: 5px;
            margin: 3px 0;
            background: rgba(0, 255, 0, 0.1);
            border-left: 3px solid #00ff00;
            font-size: 0.85em;
            animation: slideIn 0.3s;
        }
        
        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .event-item.tool { border-left-color: #ff0000; background: rgba(255, 0, 0, 0.1); }
        .event-item.decision { border-left-color: #ffff00; background: rgba(255, 255, 0, 0.1); }
        
        .timestamp {
            color: #666;
            font-size: 0.8em;
        }
        
        .context-item {
            padding: 5px;
            margin: 5px 0;
            background: rgba(0, 255, 255, 0.1);
            border-radius: 3px;
        }
        
        .context-key {
            color: #00ffff;
            font-weight: bold;
        }
        
        .metric {
            display: inline-block;
            margin: 10px;
            padding: 10px 20px;
            background: rgba(255, 0, 255, 0.2);
            border: 1px solid #ff00ff;
            border-radius: 5px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2em;
            color: #ff00ff;
            font-weight: bold;
        }
        
        .metric-label {
            font-size: 0.8em;
            color: #aaa;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 AI THOUGHT VISUALIZATION - LIVE MODE 🧠</h1>
        <div class="status">● ACTIVE - Showing real conversation data</div>
    </div>
    
    <div class="container">
        <div class="panel" id="graph">
            <h2>Reasoning Graph (Live)</h2>
            <svg id="graph-svg"></svg>
        </div>
        
        <div class="panel">
            <h2>Context Memory</h2>
            <div id="context"></div>
            <h2 style="margin-top: 20px;">Metrics</h2>
            <div id="metrics">
                <div class="metric">
                    <div class="metric-value" id="metric-tokens">0</div>
                    <div class="metric-label">Tokens</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="metric-tools">0</div>
                    <div class="metric-label">Tools</div>
                </div>
                <div class="metric">
                    <div class="metric-value" id="metric-time">0s</div>
                    <div class="metric-label">Time</div>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <h2>Event Log</h2>
            <div id="events"></div>
        </div>
    </div>
    
    <script>
        // D3 graph setup
        const svg = d3.select("#graph-svg");
        const width = svg.node().parentElement.clientWidth;
        const height = svg.node().parentElement.clientHeight;
        
        svg.attr("width", width).attr("height", height);
        
        const simulation = d3.forceSimulation()
            .force("link", d3.forceLink().id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(30));
        
        let linkGroup = svg.append("g").attr("class", "links");
        let nodeGroup = svg.append("g").attr("class", "nodes");
        
        const colorMap = {
            thinking: "#00ff00",
            tool: "#ff0000",
            decision: "#ffff00",
            result: "#00ffff",
            user: "#0088ff",
            assistant: "#00ff00"
        };
        
        // Poll for updates every 2 seconds
        async function pollUpdates() {
            try {
                const response = await fetch('/rest/api/v1/viz/state');
                const data = await response.json();
                
                // Update graph
                updateGraph(data.graph);
                
                // Update context
                updateContext(data.context);
                
                // Update metrics
                updateMetrics(data.metrics);
                
                // Update events
                updateEvents(data.events);
            } catch (err) {
                console.error('Failed to poll updates:', err);
            }
            
            setTimeout(pollUpdates, 2000);
        }
        
        function updateGraph(graph) {
            if (!graph || !graph.nodes) return;
            
            // Update links
            const links = linkGroup.selectAll("path")
                .data(graph.links, d => `${d.source.id || d.source}-${d.target.id || d.target}`);
            
            links.enter()
                .append("path")
                .attr("class", "link")
                .merge(links);
            
            links.exit().remove();
            
            // Update nodes
            const nodes = nodeGroup.selectAll("g")
                .data(graph.nodes, d => d.id);
            
            const nodeEnter = nodes.enter()
                .append("g")
                .attr("class", "node")
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));
            
            nodeEnter.append("circle")
                .attr("r", 8)
                .attr("fill", d => colorMap[d.category] || "#00ff00");
            
            nodeEnter.append("text")
                .attr("dy", 20)
                .text(d => d.label);
            
            nodeEnter.append("title")
                .text(d => d.full_text);
            
            // Update simulation
            simulation.nodes(graph.nodes);
            simulation.force("link").links(graph.links);
            simulation.alpha(0.3).restart();
        }
        
        simulation.on("tick", () => {
            linkGroup.selectAll("path")
                .attr("d", d => {
                    const dx = d.target.x - d.source.x;
                    const dy = d.target.y - d.source.y;
                    return `M${d.source.x},${d.source.y} L${d.target.x},${d.target.y}`;
                });
            
            nodeGroup.selectAll("g")
                .attr("transform", d => `translate(${d.x},${d.y})`);
        });
        
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        
        function updateContext(context) {
            if (!context) return;
            const contextDiv = document.getElementById('context');
            contextDiv.innerHTML = Object.entries(context).map(([key, value]) => `
                <div class="context-item">
                    <span class="context-key">${key}:</span> ${JSON.stringify(value)}
                </div>
            `).join('');
        }
        
        function updateMetrics(metrics) {
            if (!metrics) return;
            document.getElementById('metric-tokens').textContent = metrics.tokens || 0;
            document.getElementById('metric-tools').textContent = metrics.tools || 0;
            document.getElementById('metric-time').textContent = (metrics.time || 0).toFixed(1) + 's';
        }
        
        function updateEvents(events) {
            if (!events || events.length === 0) return;
            const eventsDiv = document.getElementById('events');
            events.slice(-10).forEach(event => {
                const item = document.createElement('div');
                item.className = `event-item ${event.type}`;
                item.innerHTML = `
                    <span class="timestamp">${new Date(event.timestamp).toLocaleTimeString()}</span>
                    <div>${event.message}</div>
                `;
                eventsDiv.insertBefore(item, eventsDiv.firstChild);
            });
            
            // Keep only last 50 events
            while (eventsDiv.children.length > 50) {
                eventsDiv.removeChild(eventsDiv.lastChild);
            }
        }
        
        // Start polling
        pollUpdates();
    </script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def get_viz_ui():
    """Serve the visualization UI"""
    return HTMLResponse(content=HTML_TEMPLATE)


@router.get("/state")
async def get_viz_state(
    db: DBSessionDep,
    current_user_email: UserEmailDep,
    chat_id: str = None
):
    """
    Get current visualization state from REAL DATABASE DATA
    
    This replaces the old demo/hardcoded data with actual conversation data.
    """
    try:
        # If no chat_id provided, get the most recent active chat for this user
        if not chat_id:
            query = select(Chat).where(
                Chat.owner_id == current_user_email
            ).order_by(desc(Chat.last_message_at)).limit(1)
            result = await db.execute(query)
            recent_chat = result.scalar_one_or_none()
            if recent_chat:
                chat_id = str(recent_chat.chat_id)
        
        if not chat_id:
            return JSONResponse({
                "graph": {"nodes": [], "links": []},
                "context": {"status": "No active chats found"},
                "metrics": {"tokens": 0, "tools": 0, "time": 0},
                "events": []
            })
        
        # Build thought graph from messages
        graph_data = await _build_thought_graph_from_messages(db, chat_id)
        
        # Get usage metrics from database
        metrics = await _get_real_usage_metrics(db, chat_id, current_user_email)
        
        # Get context from recent chat activity
        context = await _get_chat_context(db, chat_id)
        
        # Get event timeline from messages and usage data
        events = await _get_event_timeline(db, chat_id)
        
        return JSONResponse({
            "graph": graph_data,
            "context": context,
            "metrics": metrics,
            "events": events
        })
    
    except Exception as e:
        logger.error(f"Error getting viz state: {e}", exc_info=True)
        return JSONResponse({
            "graph": {"nodes": [], "links": []},
            "context": {"error": str(e)},
            "metrics": {"tokens": 0, "tools": 0, "time": 0},
            "events": []
        }, status_code=500)


async def _build_thought_graph_from_messages(db: AsyncSession, chat_id: str) -> Dict[str, Any]:
    """
    Build a thought graph from actual message data in the database.
    
    This replaces the hardcoded demo nodes with real conversation flow.
    """
    try:
        # Get messages for this chat
        query = select(MessageMD).where(
            MessageMD.chat_id == chat_id
        ).order_by(MessageMD.checkpoint_id)
        result = await db.execute(query)
        messages = result.scalars().all()
        
        if not messages:
            return {"nodes": [], "links": []}
        
        # Build nodes and links from message flow
        nodes = []
        links = []
        node_map = {}  # Map checkpoint_id to node index
        
        for idx, msg in enumerate(messages):
            # Determine category based on role
            category = "user" if msg.role.value == "user" else "assistant"
            if msg.role.value == "tool":
                category = "tool"
            
            # Create node
            checkpoint_text = str(msg.checkpoint_id)[:20]
            label = f"{msg.role.value}: {checkpoint_text}..."
            
            node = {
                "id": idx,
                "label": label,
                "category": category,
                "timestamp": str(msg.checkpoint_id),
                "full_text": f"Checkpoint: {msg.checkpoint_id}, Role: {msg.role.value}"
            }
            nodes.append(node)
            node_map[msg.checkpoint_id] = idx
            
            # Create link to parent if exists
            if msg.parent_checkpoint_id and msg.parent_checkpoint_id in node_map:
                links.append({
                    "source": node_map[msg.parent_checkpoint_id],
                    "target": idx
                })
        
        return {"nodes": nodes, "links": links}
    
    except Exception as e:
        logger.error(f"Error building thought graph: {e}", exc_info=True)
        return {"nodes": [], "links": []}


async def _get_real_usage_metrics(db: AsyncSession, chat_id: str, owner_id: str) -> Dict[str, Any]:
    """
    Get REAL usage metrics from the usage_tracking table.
    
    This replaces the hardcoded demo metrics with actual token usage.
    """
    try:
        # Get usage records for this chat
        query = select(
            func.sum(UsageTracking.input_tokens + UsageTracking.output_tokens).label("total_tokens"),
            func.count(UsageTracking.id).label("tool_count"),
            func.max(UsageTracking.timestamp).label("last_activity")
        ).where(
            UsageTracking.chat_id == chat_id,
            UsageTracking.owner_id == owner_id
        )
        result = await db.execute(query)
        row = result.one_or_none()
        
        if not row or row.total_tokens is None:
            return {"tokens": 0, "tools": 0, "time": 0}
        
        # Calculate time elapsed
        time_elapsed = 0
        if row.last_activity:
            # Get the earliest activity for this chat
            earliest_query = select(func.min(UsageTracking.timestamp)).where(
                UsageTracking.chat_id == chat_id
            )
            earliest_result = await db.execute(earliest_query)
            earliest_time = earliest_result.scalar_one_or_none()
            
            if earliest_time:
                time_delta = row.last_activity - earliest_time
                time_elapsed = time_delta.total_seconds()
        
        return {
            "tokens": int(row.total_tokens or 0),
            "tools": int(row.tool_count or 0),
            "time": round(time_elapsed, 1)
        }
    
    except Exception as e:
        logger.error(f"Error getting usage metrics: {e}", exc_info=True)
        return {"tokens": 0, "tools": 0, "time": 0}


async def _get_chat_context(db: AsyncSession, chat_id: str) -> Dict[str, Any]:
    """
    Get context information about the chat.
    """
    try:
        query = select(Chat).where(Chat.chat_id == chat_id)
        result = await db.execute(query)
        chat = result.scalar_one_or_none()
        
        if not chat:
            return {"status": "Chat not found"}
        
        return {
            "title": chat.title,
            "owner": chat.owner_id,
            "last_message": str(chat.last_message_at),
            "vault_mode": chat.vault_mode,
            "namespace": chat.namespace
        }
    
    except Exception as e:
        logger.error(f"Error getting chat context: {e}", exc_info=True)
        return {"error": str(e)}


async def _get_event_timeline(db: AsyncSession, chat_id: str) -> List[Dict[str, Any]]:
    """
    Get event timeline from usage tracking and messages.
    """
    try:
        # Get recent usage events
        query = select(UsageTracking).where(
            UsageTracking.chat_id == chat_id
        ).order_by(desc(UsageTracking.timestamp)).limit(20)
        result = await db.execute(query)
        usage_records = result.scalars().all()
        
        events = []
        for record in usage_records:
            event_type = "tool" if record.task == "tool" else "thinking"
            message = f"{record.model} - {record.task}: {record.input_tokens}→{record.output_tokens} tokens"
            
            events.append({
                "type": event_type,
                "timestamp": record.timestamp.isoformat(),
                "message": message
            })
        
        # Sort by timestamp (most recent first)
        events.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return events
    
    except Exception as e:
        logger.error(f"Error getting event timeline: {e}", exc_info=True)
        return []


@router.post("/clear")
async def clear_viz_state():
    """
    Clear all visualization state.
    
    Note: This endpoint is now a no-op since we fetch from database.
    To clear data, you would need to delete from the database tables.
    """
    return JSONResponse({
        "status": "cleared",
        "note": "Visualization now uses live database data. To clear, delete from database tables."
    })


@router.post("/event")
async def receive_viz_event(request: Request):
    """
    Receive thought/tool/metric events from ThoughtInterceptor.

    Resolves the 404 that occurred when ThoughtInterceptor posted to:
      http://localhost:8000/rest/api/v1/viz/event

    The viz_router is now mounted at /rest/api/v1/viz (see main.py),
    so this handler will be reachable at /rest/api/v1/viz/event.

    Events are logged with a structured prefix so they can be grepped
    from backend.log and surfaced in the Token Savings tab.
    """
    try:
        event = await request.json()
        event_type = event.get("type", "unknown")

        if event_type == "metric":
            metric_name = event.get("metric", "unknown")
            value = event.get("value", {})

            if metric_name == "tool_result_compression":
                logger.info(
                    "COMPRESSION_EVENT tool=%s original_chars=%s compressed_chars=%s "
                    "savings_chars=%s strategy=%s",
                    value.get("tool", "unknown"),
                    value.get("original_chars", 0),
                    value.get("compressed_chars", 0),
                    value.get("savings_chars", 0),
                    value.get("strategy", "unknown"),
                )
            else:
                logger.debug("Viz metric: %s = %s", metric_name, value)

        else:
            logger.debug("Viz event type=%s", event_type)

        return JSONResponse({"status": "ok", "event_type": event_type})

    except Exception as exc:
        logger.error("Error processing viz event: %s", exc)
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=400)
