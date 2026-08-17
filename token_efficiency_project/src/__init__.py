"""
Token Efficiency Layer for LLM-based chat agents.

Reduces token consumption by 40-80% through:
- Codebook-based entity compression (sigils)
- Rolling conversation summarization (tiered memory)
- Prompt compression (redundancy removal)

Usage:
    from token_efficiency import ContextAssemblyEngine

    engine = ContextAssemblyEngine.from_config("config.yaml")
    engine.process_new_turn("user", message)
    optimized = engine.build_optimized_prompt(message)
    # Send optimized to LLM API
"""

from .codebook_manager import CodebookManager
from .context_assembly import ContextAssemblyEngine
from .prompt_compressor import PromptCompressor
from .rolling_summarizer import RollingSummarizer
from .token_counter import count_tokens, estimate_cost, fits_in_budget, tokens_remaining

__version__ = "0.1.0"

__all__ = [
    "CodebookManager",
    "ContextAssemblyEngine",
    "PromptCompressor",
    "RollingSummarizer",
    "count_tokens",
    "estimate_cost",
    "fits_in_budget",
    "tokens_remaining",
]
