"""
Token counting utility for the Token Efficiency Layer.

Uses tiktoken (cl100k_base) as an approximation for Claude/GPT-4 tokenization.
For production use with Claude specifically, consider using Anthropic's 
token counting API endpoint.
"""

import re
from typing import Union

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False


def count_tokens(text: Union[str, list[dict]]) -> int:
    """
    Count tokens in text or a messages list.
    
    Args:
        text: Either a string or a list of message dicts 
              [{"role": "user", "content": "..."}]
    
    Returns:
        Approximate token count
    """
    if isinstance(text, list):
        # Messages list - count each message with overhead
        total = 0
        for msg in text:
            total += 4  # message overhead (role, content markers)
            total += count_tokens(msg.get("content", ""))
        total += 2  # conversation overhead
        return total
    
    if _HAS_TIKTOKEN:
        return len(_ENCODER.encode(text))
    else:
        # Fallback: rough approximation (1 token ≈ 4 chars for English)
        return len(text) // 4


def count_tokens_in_segments(text: str, segments: list[tuple[int, int]]) -> list[int]:
    """
    Count tokens for specific character ranges within text.
    Useful for measuring compression of specific sections.
    """
    return [count_tokens(text[start:end]) for start, end in segments]


def estimate_cost(input_tokens: int, output_tokens: int, 
                  input_rate: float = 0.003, output_rate: float = 0.015) -> float:
    """
    Estimate API cost in USD.
    Default rates are for Claude Sonnet (as of 2024).
    """
    return (input_tokens / 1000 * input_rate) + (output_tokens / 1000 * output_rate)


def fits_in_budget(messages: list[dict], budget: int) -> bool:
    """Check if messages fit within token budget."""
    return count_tokens(messages) <= budget


def tokens_remaining(messages: list[dict], context_window: int, 
                     response_reserve: int = 4000) -> int:
    """Calculate remaining tokens available for response."""
    used = count_tokens(messages)
    return context_window - used - response_reserve
