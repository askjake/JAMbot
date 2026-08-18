"""
Calibrated token counter with Bedrock CountTokens API validation.

Architecture:
- Hot path (per-fragment): tiktoken cl100k_base with adaptive calibration factor
- Validation path (per-invocation): Bedrock CountTokens API for exact pre-flight count
- Calibration: Rolling average of (actual/estimated) ratio auto-tunes the factor

The calibration factor starts at 1.0 and converges toward the true ratio over
the first ~20 model calls. After convergence, the hot-path estimate typically
drifts <3% from Bedrock's exact count.

Usage:
    from app.message.token_counter import count_tokens, validate_token_count_async

    # Hot path - instant, calibrated offline estimate
    tokens = count_tokens("some text")

    # Validation path - exact Bedrock count (async, ~100ms latency)
    exact = await validate_token_count_async(messages, model_id, system_prompt)
"""
import logging
import time
from collections import deque
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger(__name__)
calibration_logger = logging.getLogger("app.message.token_counter.calibration")

# ─── Offline Tokenizer (hot path) ─────────────────────────────────────────────

_tokenizer = None
_tokenizer_name = "none"

try:
    import tiktoken
    _tokenizer = tiktoken.get_encoding("cl100k_base")
    _tokenizer_name = "tiktoken-cl100k_base"
except Exception as e:
    logger.warning(f"tiktoken unavailable, using char//4 fallback: {e}")


def _raw_token_count(text: str) -> int:
    """Raw token count using best available offline tokenizer (no calibration)."""
    if _tokenizer is not None:
        try:
            return len(_tokenizer.encode(text))
        except Exception:
            pass
    # Fallback: 1 token ≈ 4 characters
    return max(1, len(text) // 4)


# ─── Adaptive Calibration ─────────────────────────────────────────────────────

# D3B-FIX(2026-08-18): plausible band for (actual / estimated). tiktoken
# cl100k_base vs the served model's real tokenizer diverges by tens of percent,
# not multiples. Values outside this band are treated as scope mismatch or
# mispaired samples, not as tokenizer drift, and are rejected.
_PLAUSIBLE_RATIO_MIN = 0.5
_PLAUSIBLE_RATIO_MAX = 2.0

class _CalibrationState:
    """
    Maintains a rolling calibration factor between offline estimates and
    Bedrock actual counts.
    
    The factor = actual / estimated. When actual > estimated, factor > 1.0
    meaning we were under-counting and should multiply estimates up.
    """
    __slots__ = ("factor", "_history", "_max_history", "_rejected")

    def __init__(self, initial_factor: float = 1.0, max_history: int = 50):
        self.factor = initial_factor
        self._history: deque = deque(maxlen=max_history)
        self._max_history = max_history
        self._rejected = 0

    def update(self, estimated: int, actual: int) -> None:
        """Record a new calibration data point and update the factor.

        D3B-FIX(2026-08-18): ratios outside ``_PLAUSIBLE_RATIO_BAND`` are
        rejected rather than averaged in. Two distinct problems produced
        implausible ratios in production:

        1. Cross-request contamination from the old module-global estimate
           stash (fixed via ContextVar below). This produced ratios from 0.04x
           to ~197x in a single window.
        2. A structural scope mismatch that still exists: the estimate counts
           only the message list, while Bedrock's ``input_tokens`` also includes
           the system prompt and the bound tool schemas. So ``actual`` is
           legitimately larger than ``estimated`` by a roughly *fixed* amount,
           which a *multiplicative* factor models incorrectly.

        This factor scales ``count_tokens()``, which drives compression and
        truncation decisions. ``effective_context_budget()`` already reserves
        space for the system prompt and tool schemas separately, so folding that
        same overhead into the factor double-counts it and makes compression
        fire far earlier than necessary. A tokenizer-vs-tokenizer disagreement
        cannot be 6x; anything that large is a scope/pairing problem, and
        silently scaling every budget decision by it destroys history for no
        reason. Rejecting those samples keeps the factor near the honest
        tokenizer-drift value.
        """
        if estimated <= 0 or actual <= 0:
            return
        ratio = actual / estimated
        if not (_PLAUSIBLE_RATIO_MIN <= ratio <= _PLAUSIBLE_RATIO_MAX):
            self._rejected += 1
            calibration_logger.warning(
                "CALIBRATION_REJECT_OUTLIER estimated=%d actual=%d ratio=%.4f "
                "band=[%.2f, %.2f] rejected_total=%d factor_unchanged=%.4f "
                "(implausible for tokenizer drift; indicates estimate/actual "
                "scope mismatch or mispaired estimate)",
                estimated, actual, ratio,
                _PLAUSIBLE_RATIO_MIN, _PLAUSIBLE_RATIO_MAX,
                self._rejected, self.factor,
            )
            return
        self._history.append(ratio)
        # Exponential moving average weighted toward recent observations
        if len(self._history) >= 3:
            # Use trimmed mean (drop highest/lowest) for robustness
            sorted_history = sorted(self._history)
            trim = max(1, len(sorted_history) // 10)
            trimmed = sorted_history[trim:-trim] if trim < len(sorted_history) // 2 else sorted_history
            self.factor = sum(trimmed) / len(trimmed)
        else:
            self.factor = sum(self._history) / len(self._history)

        # Final guard: never let the factor leave the plausible band, even if
        # the accepted-sample mean somehow drifts there.
        self.factor = min(_PLAUSIBLE_RATIO_MAX, max(_PLAUSIBLE_RATIO_MIN, self.factor))

        calibration_logger.debug(
            "CALIBRATION_UPDATE estimated=%d actual=%d ratio=%.4f new_factor=%.4f history_size=%d",
            estimated, actual, ratio, self.factor, len(self._history)
        )

    @property
    def sample_count(self) -> int:
        return len(self._history)

    @property
    def is_converged(self) -> bool:
        """True when we have enough samples for a stable factor."""
        return len(self._history) >= 10

    def get_stats(self) -> dict:
        """Return calibration statistics for observability."""
        if not self._history:
            return {
                "factor": self.factor,
                "samples": 0,
                "converged": False,
                "rejected_outliers": self._rejected,
            }
        sorted_h = sorted(self._history)
        return {
            "factor": round(self.factor, 4),
            "samples": len(self._history),
            "converged": self.is_converged,
            "rejected_outliers": self._rejected,
            "min_ratio": round(sorted_h[0], 4),
            "max_ratio": round(sorted_h[-1], 4),
            "p50_ratio": round(sorted_h[len(sorted_h) // 2], 4),
        }


# Global calibration state (persists across the process lifetime)
_calibration = _CalibrationState(initial_factor=1.0)


# ─── Public API: Calibrated Count ─────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """
    Count tokens with adaptive calibration applied.
    
    This is the hot-path function called hundreds of times per compression pass.
    It uses the offline tokenizer multiplied by the calibration factor learned
    from Bedrock CountTokens API responses.
    
    Returns:
        Calibrated token count (integer, always >= 1)
    """
    raw = _raw_token_count(text)
    calibrated = int(raw * _calibration.factor + 0.5)  # Round to nearest
    return max(1, calibrated)


def count_tokens_raw(text: str) -> int:
    """
    Uncalibrated raw token count (for drift comparison).
    Exposed for the drift logger to compare raw vs. calibrated vs. actual.
    """
    return _raw_token_count(text)


def get_calibration_stats() -> dict:
    """Return current calibration statistics for dashboards/health checks."""
    return {
        "tokenizer": _tokenizer_name,
        **_calibration.get_stats(),
    }


def record_calibration_point(estimated_tokens: int, actual_tokens: int) -> None:
    """
    Record a calibration data point from a Bedrock API response.
    
    Called by the usage tracking layer after each model invocation,
    feeding the actual token count back to calibrate future estimates.
    
    Args:
        estimated_tokens: What count_tokens() estimated before invocation
        actual_tokens: What Bedrock reported as actual input tokens
    """
    _calibration.update(estimated_tokens, actual_tokens)




# ─── Pre-Invocation Estimate Stash ────────────────────────────────────────────
# Before each model invocation, call_model() calls stash_pre_invocation_estimate()
# with the message token count it computed. The usage tracker retrieves this after
# the invocation completes to feed calibration.
#
# D3B-FIX(2026-08-18): this was a plain module-level global, justified by the
# comment "each async request runs in a single coroutine chain, so a simple
# module variable suffices". That reasoning is wrong. One coroutine chain per
# request does not isolate requests: concurrent requests interleave at every
# ``await`` on the same event loop, so request B's stash overwrote request A's
# before A's _feed_calibration() read it back. The calibration loop was
# therefore pairing one request's estimate with another request's actual count.
#
# Production evidence: within a single 50-sample window, per-sample ratios
# ranged from 0.04x (estimated=131279 actual=5610) to ~197x (estimated=94
# actual=18519), and the "converged" factor swung between 1.19 and 28.5 across
# windows. That is cross-request contamination, not tokenizer drift.
#
# A ContextVar isolates the value per async task while still propagating down a
# single request's coroutine chain. This is the same primitive the consumer
# module (app/usage_tracking/service.py) already uses for
# ``usage_metadata_callback_var``.

_pre_invocation_estimate_var: ContextVar[int] = ContextVar(
    "pre_invocation_estimate", default=0
)


def stash_pre_invocation_estimate(estimated_tokens: int) -> None:
    """
    Store the estimated input token count just before model invocation.

    Called by the compression pipeline after truncate_messages() completes.
    The usage tracking layer retrieves this via get_last_estimate() to
    feed the calibration system.

    The value is scoped to the current async context, so concurrent requests
    cannot overwrite each other's estimate.
    """
    _pre_invocation_estimate_var.set(int(estimated_tokens))
    calibration_logger.debug(
        "PRE_INVOCATION_ESTIMATE stashed=%d calibration_factor=%.4f",
        estimated_tokens, _calibration.factor
    )


def get_last_estimate() -> int:
    """
    Retrieve the pre-invocation estimate for the *current* async context.

    Called by the usage tracking service after Bedrock responds, to feed
    the calibration loop. Returns 0 if no estimate was stashed in this context.
    """
    return _pre_invocation_estimate_var.get()


def reset_pre_invocation_estimate() -> None:
    """Clear the current context's estimate so it cannot be reused.

    A stale estimate paired with a later invocation's actual count is exactly
    the corruption this module is guarding against.
    """
    _pre_invocation_estimate_var.set(0)

# ─── Bedrock CountTokens API Validation ───────────────────────────────────────

async def validate_token_count_async(
    messages: list,
    model_id: str,
    system_prompt: Optional[str] = None,
    region: Optional[str] = None,
) -> Optional[int]:
    """
    Call Bedrock CountTokens API for exact pre-flight token count.
    
    This is the validation path — called ONCE per model invocation after
    compression is complete. It provides the exact count that Bedrock will
    bill for, and feeds the calibration system.
    
    Args:
        messages: LangChain messages (will be converted to Bedrock Converse format)
        model_id: Bedrock model identifier (e.g., "anthropic.claude-sonnet-4-20250514-v1:0")
        system_prompt: Optional system prompt text
        region: AWS region (defaults to settings)
        
    Returns:
        Exact input token count from Bedrock, or None if the call fails.
        Failures are non-fatal (logged as warning, compression proceeds with estimate).
    """
    try:
        import boto3
        from app.config import get_settings
        settings = get_settings()
        
        region = region or getattr(settings, "AWS_REGION", "us-west-2")
        
        # Build the Converse-format messages for CountTokens
        converse_messages = _langchain_to_converse_messages(messages)
        if not converse_messages:
            logger.warning("validate_token_count_async: no messages to count")
            return None
        
        # Build the request
        request_params = {
            "modelId": model_id,
            "input": {
                "converse": {
                    "messages": converse_messages,
                }
            }
        }
        
        # Add system prompt if provided
        if system_prompt:
            request_params["input"]["converse"]["system"] = [
                {"text": system_prompt}
            ]
        
        # Make the API call (async via thread pool to avoid blocking)
        import asyncio
        loop = asyncio.get_event_loop()
        
        def _sync_count():
            client = boto3.client("bedrock-runtime", region_name=region)
            start = time.perf_counter()
            response = client.count_tokens(**request_params)
            elapsed_ms = (time.perf_counter() - start) * 1000
            exact_count = response.get("inputTokens", 0)
            calibration_logger.info(
                "BEDROCK_COUNT_TOKENS model=%s exact_tokens=%d latency_ms=%.1f",
                model_id, exact_count, elapsed_ms
            )
            return exact_count
        
        exact_count = await loop.run_in_executor(None, _sync_count)
        return exact_count
        
    except Exception as e:
        logger.warning(
            "Bedrock CountTokens validation failed (non-fatal, using estimate): %s", e
        )
        return None


def _langchain_to_converse_messages(messages: list) -> list:
    """
    Convert LangChain messages to minimal Bedrock Converse format for CountTokens.
    
    This is intentionally minimal — we only need enough structure for token
    counting, not full-fidelity conversion (that happens in the actual Converse call).
    """
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
    
    converse_msgs = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            # System messages are passed separately, skip here
            continue
        
        role = "user" if isinstance(msg, (HumanMessage, ToolMessage)) else "assistant"
        content = msg.content
        
        # Normalize content to Bedrock format
        if isinstance(content, str):
            blocks = [{"text": content}] if content else [{"text": " "}]
        elif isinstance(content, list):
            blocks = []
            for item in content:
                if isinstance(item, str):
                    blocks.append({"text": item})
                elif isinstance(item, dict):
                    if "text" in item:
                        blocks.append({"text": item["text"]})
                    elif "type" in item and item["type"] == "text":
                        blocks.append({"text": item.get("text", " ")})
                    else:
                        blocks.append({"text": str(item)})
            if not blocks:
                blocks = [{"text": " "}]
        else:
            blocks = [{"text": str(content) if content else " "}]
        
        # Bedrock requires alternating user/assistant - merge consecutive same-role
        if converse_msgs and converse_msgs[-1]["role"] == role:
            converse_msgs[-1]["content"].extend(blocks)
        else:
            converse_msgs.append({"role": role, "content": blocks})
    
    # Ensure starts with user
    if converse_msgs and converse_msgs[0]["role"] != "user":
        converse_msgs.insert(0, {"role": "user", "content": [{"text": " "}]})
    
    return converse_msgs
