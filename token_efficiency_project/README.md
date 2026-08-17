# Token Efficiency Layer

## Problem
LLM API calls scale linearly with conversation length. A 50-turn conversation
may consume 80K+ tokens per turn, most of which is redundant history.

## Solution
A middleware layer that compresses conversation context through three
complementary mechanisms:

1. **Codebook Manager** — Reusable shorthand symbols for repeated entities
2. **Rolling Summarizer** — Tiered memory (verbatim → detailed → abstract)
3. **Prompt Compressor** — Remove informationally redundant tokens

## Quick Start

```bash
pip install tiktoken sentence-transformers pyyaml
```

```python
from src.context_assembly import ContextAssemblyEngine

engine = ContextAssemblyEngine.from_config("config.yaml")

# Each turn:
engine.process_new_turn("user", user_message)
optimized_messages = engine.build_optimized_prompt(user_message)
response = await llm_api.chat(messages=optimized_messages)
engine.process_new_turn("assistant", response)
```

## Architecture

See `docs/AGENT_IMPLEMENTATION_BRIEF.md` for the complete specification.

## Expected Results

| Conversation Length | Token Savings | Quality Impact |
|--------------------|--------------:|:--------------:|
| 10 turns           | 20-30%        | Negligible     |
| 20 turns           | 40-60%        | < 5% degradation |
| 50 turns           | 60-80%        | < 10% degradation |

## File Structure

```
├── config.yaml                         # All tunable parameters
├── docs/
│   └── AGENT_IMPLEMENTATION_BRIEF.md   # Complete spec for implementation agent
├── src/
│   ├── __init__.py
│   ├── token_counter.py                # Token counting utilities (implemented)
│   ├── codebook_manager.py             # CodebookManager implementation
│   ├── rolling_summarizer.py           # RollingSummarizer implementation
│   ├── prompt_compressor.py            # PromptCompressor implementation
│   └── context_assembly.py             # ContextAssemblyEngine orchestrator
└── tests/                              # Integration and component tests
```

## Implementation Status

- [x] Architecture design & specification
- [x] Configuration schema
- [x] Token counting utility
- [x] Codebook Manager
- [x] Rolling Summarizer
- [x] Prompt Compressor
- [x] Context Assembly Engine
- [x] Test suite
- [ ] Benchmarks

## Key Design Decisions

1. **Never compress the current user message** — faithfulness to intent
2. **Never compress system instructions** — model behavior depends on exact wording
3. **Graceful degradation** — if compression fails, fall back to uncompressed
4. **Observable** — metrics on every call showing savings and fidelity
5. **Composable** — each component works independently and can be toggled
