"""
Message history compression and token management utilities.
"""
import logging
from typing import List, Optional
try:
    from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
except ModuleNotFoundError:
    class BaseMessage:  # type: ignore
        content = ""
    class SystemMessage(BaseMessage):  # type: ignore
        pass
    class HumanMessage(BaseMessage):  # type: ignore
        pass
    class AIMessage(BaseMessage):  # type: ignore
        pass

from app.config import get_settings
from app.core.llm.model_roles import resolve_model_role

settings = get_settings()
logger = logging.getLogger(__name__)

# Token counting is delegated to the calibrated token_counter module.
# Import lazily to avoid circular imports during early module loading.
_token_counter_module = None

def _get_token_counter():
    """Lazy import of the calibrated token counter module."""
    global _token_counter_module
    if _token_counter_module is None:
        try:
            from app.message import token_counter as _tc
            _token_counter_module = _tc
        except ImportError:
            _token_counter_module = None
    return _token_counter_module



def effective_context_budget(role: str = "primary") -> dict:
    role_config = resolve_model_role(role, settings=settings)
    configured_context = int(role_config.context_length or settings.MAX_CONTEXT)
    reserved_output = int(role_config.max_output_tokens or getattr(settings, "OLLAMA_MAX_OUTPUT_TOKENS", 4096) or 4096)
    reserved_system = min(8192, max(2048, configured_context // 10))
    reserved_tools = min(8192, max(2048, configured_context // 8))
    history_budget = max(1024, configured_context - reserved_output - reserved_system - reserved_tools)
    tool_result_budget = max(500, min(8000, history_budget // 16))
    return {
        "model_role": role_config.role,
        "provider": role_config.provider,
        "model_name": role_config.model_name,
        "configured_context": configured_context,
        "reserved_system_tool_output_budget": reserved_output + reserved_system + reserved_tools,
        "reserved_output_budget": reserved_output,
        "reserved_system_budget": reserved_system,
        "reserved_tool_schema_budget": reserved_tools,
        "history_budget": history_budget,
        "tool_result_budget": tool_result_budget,
    }

def count_tokens(text: str) -> int:
    """
    Count approximate tokens in text using the calibrated token counter.
    
    Delegates to app.message.token_counter which applies an adaptive
    calibration factor learned from Bedrock CountTokens API responses.
    Falls back to tiktoken cl100k_base directly if the module isn't available.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Calibrated approximate token count
    """
    tc = _get_token_counter()
    if tc is not None:
        return tc.count_tokens(text)
    
    # Direct fallback if token_counter module fails to load
    try:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
        return len(_enc.encode(text))
    except Exception:
        pass
    
    # Final fallback: rough approximation (1 token ≈ 4 characters)
    return max(1, len(text) // 4)

# Pressure-management estimate, not a billing counter.  Do not tokenize
# serialized image/base64 bytes as text.  5k/image is intentionally
# conservative for local history-pressure decisions while remaining bounded.
MULTIMODAL_IMAGE_TOKEN_ESTIMATE = 5000


def _is_image_content_block(item) -> bool:
    if not isinstance(item, dict):
        return False

    block_type = str(item.get("type") or "").strip().lower()
    if block_type in {"image", "image_url"}:
        return True

    # Bedrock Converse tagged-union representation.
    if isinstance(item.get("image"), dict):
        return True

    # Anthropic-style source block if a caller passes the source itself.
    source = item.get("source")
    if isinstance(source, dict):
        media_type = str(source.get("media_type") or "").lower()
        source_type = str(source.get("type") or "").lower()
        if media_type.startswith("image/") and source_type in {
            "base64",
            "bytes",
            "url",
            "file",
        }:
            return True

    return False


def count_tokens_for_message(msg: BaseMessage) -> int:
    '''Count approximate model-context tokens for one LangChain message.

    Text is counted with the existing tokenizer.  Image blocks receive a
    bounded visual-token estimate; their serialized base64/byte payload is
    deliberately not passed through the text tokenizer.

    Unknown non-image structured blocks retain the prior ``str(item)``
    behavior so this focused fix does not silently change other semantics.
    '''
    total = 0
    content = getattr(msg, "content", "")

    if isinstance(content, str):
        total += count_tokens(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                total += count_tokens(item)
            elif _is_image_content_block(item):
                total += MULTIMODAL_IMAGE_TOKEN_ESTIMATE
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                total += count_tokens(str(item["text"]))
            elif isinstance(item, dict):
                total += count_tokens(str(item))
            elif item is not None:
                total += count_tokens(str(item))
    elif isinstance(content, dict):
        if _is_image_content_block(content):
            total += MULTIMODAL_IMAGE_TOKEN_ESTIMATE
        elif isinstance(content.get("text"), str):
            total += count_tokens(str(content["text"]))
        else:
            total += count_tokens(str(content))
    elif content is not None:
        total += count_tokens(str(content))

    return total + 10



def count_message_tokens(messages: List[BaseMessage]) -> int:
    """
    Count total tokens in a list of messages.
    
    Args:
        messages: List of LangChain messages
        
    Returns:
        Total token count
    """
    return sum(count_tokens_for_message(msg) for msg in messages)

def trim_message_history(
    messages: List[BaseMessage],
    max_tokens: Optional[int] = None,
    strategy: str = "last",
    keep_system: bool = True,
    token_counter = None
) -> List[BaseMessage]:
    """
    Trim message history to fit within token limit.
    
    Args:
        messages: List of messages to trim
        max_tokens: Maximum tokens to keep (default: 80% of model context)
        strategy: Trimming strategy - "last" keeps most recent messages
        keep_system: Whether to always keep system messages
        token_counter: Optional custom token counter function
        
    Returns:
        Trimmed list of messages
    """
    if not messages:
        return messages
    
    # Use the active model-role context window by default.
    if max_tokens is None:
        max_tokens = int(effective_context_budget("primary")["history_budget"])
    
    # Use custom counter or default
    counter = token_counter or count_message_tokens
    
    # Check if we're already under limit
    current_tokens = counter(messages)
    if current_tokens <= max_tokens:
        logger.debug(f"Message history within limit: {current_tokens}/{max_tokens} tokens")
        return messages
    
    logger.info(f"Trimming message history: {current_tokens} -> {max_tokens} tokens")
    
    # Separate system messages if we're keeping them
    system_messages = []
    other_messages = []
    
    for msg in messages:
        if isinstance(msg, SystemMessage) and keep_system:
            system_messages.append(msg)
        else:
            other_messages.append(msg)
    
    # Calculate tokens used by system messages
    system_tokens = counter(system_messages) if system_messages else 0
    available_tokens = max_tokens - system_tokens
    
    if available_tokens <= 0:
        logger.error("System messages exceed token limit!")
        return messages[:1]  # Return just first message
    
    # Try using langchain's trim_messages (if available)
    try:
        from langchain_core.messages import trim_messages as lc_trim
        
        # Trim other messages to fit
        trimmed_others = lc_trim(
            other_messages,
            max_tokens=available_tokens,
            strategy=strategy,
            token_counter=lambda msgs: counter(msgs),
            include_system=False,
            allow_partial=False,
            start_on="human"  # Always start with a human message
        )
        
        result = system_messages + trimmed_others
        final_tokens = counter(result)
        logger.info(f"Trimmed to {len(result)} messages ({final_tokens} tokens)")
        return result
        
    except ImportError:
        # Fallback: simple last-N strategy
        logger.warning("langchain trim_messages not available, using simple fallback")
        return _simple_trim(messages, max_tokens, keep_system, counter)

def _simple_trim(
    messages: List[BaseMessage],
    max_tokens: int,
    keep_system: bool,
    counter
) -> List[BaseMessage]:
    """
    Simple fallback trimming: keep system message + most recent messages.
    """
    system_msg = None
    other_msgs = []
    
    for msg in messages:
        if isinstance(msg, SystemMessage) and keep_system and system_msg is None:
            system_msg = msg
        else:
            other_msgs.append(msg)
    
    # Start with system message if present
    result = [system_msg] if system_msg else []
    current_tokens = counter(result)
    
    # Add messages from the end (most recent first)
    for msg in reversed(other_msgs):
        msg_tokens = counter([msg])
        if current_tokens + msg_tokens <= max_tokens:
            result.insert(1 if system_msg else 0, msg)
            current_tokens += msg_tokens
        else:
            break
    
    logger.info(f"Simple trim kept {len(result)} messages ({current_tokens} tokens)")
    return result

def should_compress_history(messages: List[BaseMessage]) -> bool:
    """
    Check if message history should be compressed.
    
    Args:
        messages: List of messages to check
        
    Returns:
        True if compression is recommended
    """
    token_count = count_message_tokens(messages)
    # Compress if we're using more than 70% of the active role history budget.
    threshold = int(effective_context_budget("primary")["history_budget"] * 0.7)
    
    if token_count > threshold:
        logger.info(f"Message history compression recommended: {token_count}/{threshold} tokens")
        return True
    
    return False

def get_compression_stats(messages: List[BaseMessage]) -> dict:
    """
    Get statistics about message history for monitoring.
    
    Args:
        messages: List of messages
        
    Returns:
        Dictionary with token counts and message stats
    """
    total_tokens = count_message_tokens(messages)
    
    message_types = {}
    for msg in messages:
        msg_type = type(msg).__name__
        message_types[msg_type] = message_types.get(msg_type, 0) + 1
    
    return {
        "total_messages": len(messages),
        "total_tokens": total_tokens,
        "context_budget": effective_context_budget("primary"),
        "context_usage_pct": (total_tokens / effective_context_budget("primary")["configured_context"]) * 100,
        "message_types": message_types,
        "avg_tokens_per_message": total_tokens // len(messages) if messages else 0
    }
