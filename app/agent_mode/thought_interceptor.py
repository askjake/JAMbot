"""
Thought interceptor for live agent visualization.
Captures agent reasoning and broadcasts to visualization server via HTTP.
"""

import os
import time
from datetime import datetime
from typing import Any, Dict, Optional
import requests
from threading import Thread
import queue


class ThoughtInterceptor:
    """
    Captures agent thoughts and sends them to the visualization server.
    Thread-safe singleton that can be used across the application.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Check if enabled via environment variable
        self.enabled = os.getenv("AGENT_THOUGHT_CAPTURE_ENABLED", "true").lower() == "true"
        self.viz_url = os.getenv("AGENT_VIZ_SERVER_URL", "http://localhost:8000/rest/api/v1/viz/event")
        
        self.start_time = time.time()
        self.event_queue = queue.Queue(maxsize=1000)
        self._initialized = True
        
        if self.enabled:
            # Start background thread to send events
            self._sender_thread = Thread(
                target=self._send_events,
                daemon=True,
                name="ThoughtInterceptor"
            )
            self._sender_thread.start()
            print(f"✓ ThoughtInterceptor enabled, sending to {self.viz_url}")
    
    def _send_events(self):
        """Background thread that sends events to visualization server."""
        while True:
            try:
                event = self.event_queue.get(timeout=1.0)
                
                # Try to send to viz server (non-blocking)
                try:
                    response = requests.post(
                        self.viz_url,
                        json=event,
                        timeout=0.5
                    )
                    if response.status_code != 200:
                        print(f"Warning: Viz server returned {response.status_code}")
                except requests.exceptions.RequestException as e:
                    # Silently fail if viz server not available
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"ThoughtInterceptor error: {e}")
    
    def thought(self, text: str, category: str = "thinking"):
        """Capture a reasoning step."""
        if not self.enabled:
            return
        
        event = {
            "type": "thought",
            "category": category,
            "text": str(text)[:500],
            "timestamp": datetime.now().isoformat(),
            "elapsed": time.time() - self.start_time
        }
        
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass
    
    def tool_call(self, tool_name: str, params: Optional[Dict] = None, 
                  result: Optional[Any] = None):
        """Capture a tool execution."""
        if not self.enabled:
            return
        
        safe_params = {}
        if params:
            for k, v in params.items():
                safe_params[k] = str(v)[:100]
        
        safe_result = None
        if result is not None:
            safe_result = str(result)[:200]
        
        event = {
            "type": "tool",
            "tool": tool_name,
            "params": safe_params,
            "result": safe_result,
            "timestamp": datetime.now().isoformat(),
            "elapsed": time.time() - self.start_time
        }
        
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass
    
    def decision(self, text: str, options: Optional[list] = None):
        """Capture a decision point."""
        if not self.enabled:
            return
        
        event = {
            "type": "decision",
            "text": str(text)[:500],
            "options": [str(o)[:100] for o in (options or [])],
            "timestamp": datetime.now().isoformat(),
            "elapsed": time.time() - self.start_time
        }
        
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass
    
    def context_update(self, key: str, value: Any):
        """Update context memory."""
        if not self.enabled:
            return
        
        event = {
            "type": "context",
            "key": str(key),
            "value": str(value)[:200],
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass
    
    def metric_update(self, metric_type: str, value: Any):
        """Update performance metrics."""
        if not self.enabled:
            return
        
        event = {
            "type": "metric",
            "metric": str(metric_type),
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass
    
    def reset(self):
        """Reset the start time (useful for new conversations)."""
        self.start_time = time.time()


# Global singleton instance
interceptor = ThoughtInterceptor()
