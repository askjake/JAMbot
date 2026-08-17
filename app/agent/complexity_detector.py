"""
Provider-Neutral Model Routing Complexity Detector v2.1
=======================================================
Intelligent prompt complexity scoring for automatic model-role selection.

Features:
- 8-factor scoring algorithm
- Fast-path detection for obvious simple/complex prompts
- Conversation context tracking (multi-turn escalation)
- LRU cache for duplicate prompt fingerprints
- Structured telemetry logging
- A/B testing support

Author: Jacob Montgomery (jacob.montgomery@dish.com)
Created: 2026-05-07
Version: 2.1
"""

import logging
import hashlib
import json
import time
import re
import random
from typing import Optional, Dict, Any
from collections import defaultdict
from functools import lru_cache
from datetime import datetime, timezone

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# ============================================================================
# TELEMETRY & METRICS
# ============================================================================

class RoutingMetrics:
    """Collects and exposes metrics for provider-neutral model routing."""
    
    def __init__(self):
        self.total_requests: int = 0
        self.complex_requests: int = 0
        self.primary_requests: int = 0
        # Legacy aliases retained for old dashboards.
        self.fast_path_hits: int = 0
        self.cache_hits: int = 0
        self.total_detection_time_ms: float = 0.0
        self.scores: list[int] = []
        self.threshold_a_requests: int = 0  # A/B test group A
        self.threshold_b_requests: int = 0  # A/B test group B
        self._start_time: datetime = datetime.now(timezone.utc)
    
    def record_request(self, model_selected: str, score: int, detection_time_ms: float,
                       fast_path_used: bool, cache_hit: bool, ab_group: Optional[str] = None):
        """Record a single routing decision."""
        self.total_requests += 1
        self.scores.append(score)
        self.total_detection_time_ms += detection_time_ms
        
        if model_selected == "complex":
            self.complex_requests += 1
        else:
            self.primary_requests += 1
        
        if fast_path_used:
            self.fast_path_hits += 1
        if cache_hit:
            self.cache_hits += 1
        if ab_group == "A":
            self.threshold_a_requests += 1
        elif ab_group == "B":
            self.threshold_b_requests += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Return current metrics as a dictionary."""
        if self.total_requests == 0:
            return {
                "status": "no_requests",
                "message": "No requests processed yet",
                "uptime_seconds": (datetime.now(timezone.utc) - self._start_time).total_seconds()
            }
        
        avg_score = sum(self.scores) / len(self.scores) if self.scores else 0
        avg_detection_ms = self.total_detection_time_ms / self.total_requests
        complex_pct = (self.complex_requests / self.total_requests) * 100
        
        return {
            "status": "active",
            "total_requests": self.total_requests,
            "complex_requests": self.complex_requests,
            "primary_requests": self.primary_requests,
            "complex_percentage": round(complex_pct, 1),
            "average_score": round(avg_score, 2),
            "average_detection_ms": round(avg_detection_ms, 2),
            "fast_path_hit_rate": round((self.fast_path_hits / self.total_requests) * 100, 1),
            "cache_hit_rate": round((self.cache_hits / self.total_requests) * 100, 1),
            "ab_test": {
                "group_a_requests": self.threshold_a_requests,
                "group_b_requests": self.threshold_b_requests,
            },
            "uptime_seconds": (datetime.now(timezone.utc) - self._start_time).total_seconds(),
            "threshold": getattr(settings, "MODEL_ROUTING_COMPLEXITY_THRESHOLD", 3),
            "auto_model_routing_enabled": getattr(settings, "AUTO_MODEL_ROUTING_ENABLED", True),
        }
    
    def reset(self):
        """Clear all metrics (for testing)."""
        self.__init__()


# Global metrics instance
routing_metrics = RoutingMetrics()


# ============================================================================
# CONVERSATION CONTEXT TRACKER
# ============================================================================

class ConversationComplexityTracker:
    """
    Tracks conversation state across turns to inform routing decisions.
    
    Multi-turn conversations that escalate in complexity should be routed
    to the configured complex role even if individual messages score below threshold.
    """
    
    def __init__(self, max_sessions: int = 1000, decay_factor: float = 0.7):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._max_sessions = max_sessions
        self._decay_factor = decay_factor
    
    def get_context_bonus(self, session_id: str, current_score: int) -> int:
        """
        Calculate context bonus based on conversation history.
        
        Rules:
        - If previous turns scored high, maintain elevated role routing
        - Decays over turns so a single complex query does not lock in the complex role forever
        - Returns bonus points (0-2) to add to current score
        """
        if not session_id or session_id not in self._sessions:
            return 0
        
        session = self._sessions[session_id]
        history = session.get("score_history", [])
        
        if not history:
            return 0
        
        # Calculate weighted average of recent scores (exponential decay)
        weighted_sum = 0.0
        weight_total = 0.0
        for i, past_score in enumerate(reversed(history[-5:])):  # Last 5 turns
            weight = self._decay_factor ** i
            weighted_sum += past_score * weight
            weight_total += weight
        
        avg_recent = weighted_sum / weight_total if weight_total > 0 else 0
        
        # Award bonus if conversation has been complex
        if avg_recent >= getattr(settings, "MODEL_ROUTING_COMPLEXITY_THRESHOLD", 3) + 1:
            return 2  # Strong history → +2 bonus
        elif avg_recent >= getattr(settings, "MODEL_ROUTING_COMPLEXITY_THRESHOLD", 3):
            return 1  # Moderate history → +1 bonus
        
        return 0
    
    def record_turn(self, session_id: str, score: int, model_used: str):
        """Record a completed turn for context tracking."""
        if not session_id:
            return
        
        # Evict oldest sessions if at capacity
        if len(self._sessions) >= self._max_sessions and session_id not in self._sessions:
            oldest_key = next(iter(self._sessions))
            del self._sessions[oldest_key]
        
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "score_history": [],
                "models_used": [],
                "turn_count": 0,
                "created_at": time.time()
            }
        
        session = self._sessions[session_id]
        session["score_history"].append(score)
        session["models_used"].append(model_used)
        session["turn_count"] += 1
        
        # Keep only last 20 turns
        if len(session["score_history"]) > 20:
            session["score_history"] = session["score_history"][-20:]
            session["models_used"] = session["models_used"][-20:]
    
    def get_turn_number(self, session_id: str) -> int:
        """Get the current turn number for a session."""
        if not session_id or session_id not in self._sessions:
            return 1
        return self._sessions[session_id]["turn_count"] + 1


# Global context tracker
context_tracker = ConversationComplexityTracker()


# ============================================================================
# FAST-PATH DETECTION
# ============================================================================

def _fast_path_check(text: str) -> Optional[bool]:
    """
    Quick determination for obviously simple or complex prompts.
    
    Returns:
        True → select the complex role (skip full scoring)
        False → select the primary role (skip full scoring)
        None → undetermined, run full scoring algorithm
    """
    if not text:
        return False  # Empty → primary role
    
    text_lower = text.lower().strip()
    text_len = len(text)
    
    # FAST PRIMARY: Very short, simple messages (< 80 chars, no code, no question complexity)
    if text_len < 80:
        # Check it's not a power-user command that's short but complex
        power_patterns = ["follow protocol", "investigate", "debug", "refactor",
                         "analyze", "review and", "step by step"]
        if not any(p in text_lower for p in power_patterns):
            if "```" not in text and text.count("?") <= 1:
                return False  # Obviously simple → primary role
    
    # FAST COMPLEX: Messages with explicit multi-system investigation requests
    complex_role_triggers = [
        ("follow protocol" in text_lower and "investigate" in text_lower),
        ("debug" in text_lower and "refactor" in text_lower and text_len > 200),
        (text.count("```") >= 4 and text_len > 500),  # Multiple code blocks
        ("architecture" in text_lower and "optimization" in text_lower and text_len > 300),
    ]
    if any(complex_role_triggers):
        return True  # Obviously complex → complex role
    
    return None  # Undetermined → run full algorithm


# ============================================================================
# PROMPT FINGERPRINTING (LRU CACHE)
# ============================================================================

def _get_prompt_fingerprint(text: str) -> str:
    """Generate a cache-friendly fingerprint for a prompt."""
    # Normalize whitespace and lowercase for fingerprinting
    normalized = " ".join(text.lower().split())
    # Use first 500 chars for fingerprint (captures essence without bloating)
    truncated = normalized[:500]
    return hashlib.md5(truncated.encode()).hexdigest()


# Cache: fingerprint → (score, timestamp)
_score_cache: Dict[str, tuple[int, float]] = {}
_CACHE_MAX_SIZE = 500
_CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_cached_score(fingerprint: str) -> Optional[int]:
    """Check cache for a previously computed score."""
    if fingerprint in _score_cache:
        score, timestamp = _score_cache[fingerprint]
        if time.time() - timestamp < _CACHE_TTL_SECONDS:
            return score
        else:
            del _score_cache[fingerprint]  # Expired
    return None


def _set_cached_score(fingerprint: str, score: int):
    """Store a score in the cache."""
    # Evict oldest entries if at capacity
    if len(_score_cache) >= _CACHE_MAX_SIZE:
        oldest_key = min(_score_cache, key=lambda k: _score_cache[k][1])
        del _score_cache[oldest_key]
    _score_cache[fingerprint] = (score, time.time())


# ============================================================================
# CONTENT EXTRACTION HELPER
# ============================================================================

def _extract_text_from_content(content) -> str:
    """
    Normalize message content to a plain text string.
    
    Handles:
    - str: returned as-is
    - list: extracts text from provider-structured content blocks
    - dict: extracts 'text' or 'content' fields
    """
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    text_parts.append(item["text"])
                elif "content" in item:
                    text_parts.append(str(item["content"]))
        return " ".join(text_parts)
    
    if isinstance(content, dict):
        if "text" in content:
            return content["text"]
        elif "content" in content:
            return str(content["content"])
        return str(content)
    
    return str(content) if content else ""


# ============================================================================
# CORE SCORING ALGORITHM (8 FACTORS)
# ============================================================================

def detect_prompt_complexity(content) -> int:
    """
    Score prompt complexity on a scale of 0-12+.
    
    8 scoring factors:
    1. Message length (+1 to +2)
    2. Code blocks present (+2)
    3. Complexity keywords (+1 each, max 3)
    4. Agent/protocol keywords (+2)
    5. Multiple questions (+1 to +2)
    6. Technical patterns (+1)
    7. Multi-step instructions (+1)
    8. Analytical depth markers (+1)
    
    Returns:
        Integer complexity score (0 = trivial, 10+ = highly complex)
    """
    text = _extract_text_from_content(content)
    
    if not text or not text.strip():
        return 0
    
    # Check fast-path first
    fast_result = _fast_path_check(text)
    if fast_result is True:
        return 10  # Guaranteed complex role
    elif fast_result is False:
        return 0  # Guaranteed primary role
    
    # Check cache
    fingerprint = _get_prompt_fingerprint(text)
    cached = _get_cached_score(fingerprint)
    if cached is not None:
        return cached
    
    score = 0
    text_lower = text.lower()
    text_len = len(text)
    
    # Factor 1: Message length
    if text_len > 500:
        score += 2
    elif text_len > 300:
        score += 1
    
    # Factor 2: Code blocks
    code_block_count = text.count("```")
    if code_block_count >= 2:  # At least one complete code block
        score += 2
    
    # Factor 3: Complexity keywords (max +3)
    complexity_keywords = [
        "debug", "refactor", "optimize", "architect", "implement",
        "integrate", "migrate", "redesign", "troubleshoot", "benchmark",
        "performance", "scalability", "security", "concurrent", "distributed"
    ]
    keyword_hits = sum(1 for kw in complexity_keywords if kw in text_lower)
    score += min(keyword_hits, 3)
    
    # Factor 4: Agent/protocol keywords (+2)
    agent_keywords = [
        "follow protocol", "investigate", "step by step",
        "thorough analysis", "comprehensive review", "deep dive",
        "root cause", "end to end"
    ]
    if any(kw in text_lower for kw in agent_keywords):
        score += 2
    
    # Factor 5: Multiple questions
    question_count = text.count("?")
    if question_count >= 3:
        score += 2
    elif question_count >= 2:
        score += 1
    
    # Factor 6: Technical patterns (APIs, versions, URLs, file paths)
    technical_patterns = [
        r'https?://\S+',          # URLs
        r'v\d+\.\d+',             # Version numbers
        r'/[a-z_]+/[a-z_]+',     # File paths
        r'\b\w+_\w+_\w+\b',      # Snake_case identifiers (3+ parts)
        r'\b[A-Z]{2,}_[A-Z]{2,}\b',  # CONSTANT_NAMES
    ]
    tech_hits = sum(1 for pattern in technical_patterns if re.search(pattern, text))
    if tech_hits >= 2:
        score += 1
    
    # Factor 7: Multi-step instructions
    step_indicators = [
        r'\b(first|second|third|then|next|finally|after that)\b',
        r'\b(step \d|phase \d|part \d)\b',
        r'\d+[.)]',  # Numbered lists
    ]
    step_hits = sum(1 for pattern in step_indicators if re.search(pattern, text_lower))
    if step_hits >= 2:
        score += 1
    
    # Factor 8: Analytical depth markers
    depth_markers = [
        "trade-off", "tradeoff", "pros and cons", "compare",
        "implications", "consequences", "evaluate", "assess",
        "recommend", "strategy", "approach"
    ]
    depth_hits = sum(1 for marker in depth_markers if marker in text_lower)
    if depth_hits >= 2:
        score += 1
    
    # Cache the result
    _set_cached_score(fingerprint, score)
    
    return score


# ============================================================================
# A/B TESTING
# ============================================================================

def get_complexity_threshold() -> int:
    """
    Get complexity threshold with optional A/B testing.
    
    When A/B testing is enabled, randomly assigns threshold A (default)
    or threshold B (test variant) at 50/50 split.
    
    Returns:
        (threshold, ab_group) tuple
    """
    if not getattr(settings, 'MODEL_ROUTING_AB_TEST_ENABLED', False):
        return getattr(settings, "MODEL_ROUTING_COMPLEXITY_THRESHOLD", 3), None
    
    # 50/50 random split
    if random.random() < 0.5:
        threshold = getattr(settings, "MODEL_ROUTING_COMPLEXITY_THRESHOLD", 3)
        group = "A"
    else:
        threshold = getattr(settings, "MODEL_ROUTING_THRESHOLD_B", 4)
        group = "B"
    
    logger.debug(f"\U0001f9ea A/B Test: Using threshold={threshold} (Group {group})")
    return threshold, group


# ============================================================================
# MAIN ROUTING FUNCTION
# ============================================================================

def choose_model_role_for_context(content, session_id: str = None) -> str:
    """Return provider-neutral model role: ``complex`` for complex work, else ``primary``."""
    start_time = time.time()
    if not getattr(settings, 'AUTO_MODEL_ROUTING_ENABLED', True):
        return "primary"

    text = _extract_text_from_content(content)
    raw_score = detect_prompt_complexity(text)
    context_bonus = context_tracker.get_context_bonus(session_id, raw_score) if session_id else 0
    effective_score = raw_score + context_bonus
    threshold, ab_group = get_complexity_threshold()
    selected_role = "complex" if effective_score >= threshold else "primary"
    detection_time_ms = (time.time() - start_time) * 1000
    fast_path_used = _fast_path_check(text) is not None
    fingerprint = _get_prompt_fingerprint(text)
    cache_hit = fingerprint in _score_cache and _score_cache[fingerprint][0] == raw_score
    turn_number = context_tracker.get_turn_number(session_id) if session_id else 1

    context_tracker.record_turn(session_id, raw_score, selected_role)
    routing_metrics.record_request(
        model_selected=selected_role,
        score=effective_score,
        detection_time_ms=detection_time_ms,
        fast_path_used=fast_path_used,
        cache_hit=cache_hit,
        ab_group=ab_group,
    )
    _emit_routing_telemetry(
        session_id=session_id,
        raw_score=raw_score,
        context_bonus=context_bonus,
        effective_score=effective_score,
        threshold=threshold,
        model_selected=selected_role,
        turn_number=turn_number,
        fast_path_used=fast_path_used,
        cache_hit=cache_hit,
        detection_time_ms=detection_time_ms,
        ab_group=ab_group,
        text_preview=text[:100] if text else "",
    )
    logger.info(
        "Model routing decision: role=%s score=%s threshold=%s%s",
        selected_role,
        effective_score,
        threshold,
        f" context_bonus={context_bonus}" if context_bonus else "",
    )
    if detection_time_ms > 50:
        logger.warning("Slow complexity detection: %.1fms (target: <50ms)", detection_time_ms)
    return selected_role


def should_use_complex_role_for_context(content, session_id: str = None) -> bool:
    """Return True when the provider-neutral routing decision selects the complex role."""
    return choose_model_role_for_context(content, session_id=session_id) == "complex"


def should_use_opus_for_context(content, session_id: str = None) -> bool:
    """Deprecated compatibility wrapper; active code should call provider-neutral role helpers."""
    return should_use_complex_role_for_context(content, session_id=session_id)


# ============================================================================
# TELEMETRY EMISSION
# ============================================================================

def _emit_routing_telemetry(
    session_id: str,
    raw_score: int,
    context_bonus: int,
    effective_score: int,
    threshold: int,
    model_selected: str,
    turn_number: int,
    fast_path_used: bool,
    cache_hit: bool,
    detection_time_ms: float,
    ab_group: Optional[str],
    text_preview: str
):
    """Emit structured telemetry for routing decisions."""
    telemetry = {
        "event": "model_routing_decision",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id or "unknown",
        "raw_score": raw_score,
        "context_bonus": context_bonus,
        "effective_score": effective_score,
        "threshold": threshold,
        "model_selected": model_selected,
        "turn_number": turn_number,
        "fast_path_used": fast_path_used,
        "cache_hit": cache_hit,
        "detection_time_ms": round(detection_time_ms, 2),
        "ab_group": ab_group,
        "text_preview": text_preview[:80]  # Truncate for privacy
    }
    
    # Log as structured JSON for easy parsing
    logger.info(f"ROUTING_TELEMETRY: {json.dumps(telemetry)}")
