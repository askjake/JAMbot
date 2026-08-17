# TOKEN EFFICIENCY SYSTEM — Agent Implementation Brief

## IDENTITY & MISSION

You are an implementation agent tasked with building a **Token Efficiency Layer** 
for an LLM-based chat agent system. This layer sits between the user-facing 
application and the LLM API, reducing token consumption while preserving semantic 
fidelity of conversations.

---

## THE PROBLEM

LLM API calls are billed per-token. Long conversations accumulate history that 
consumes the context window. Current approaches send full verbatim chat history 
on every turn, which:

1. **Wastes tokens** — Repeating information the model already "processed" in prior turns
2. **Hits context limits** — Long investigations exhaust the window (128K-200K tokens)
3. **Degrades quality** — Overloaded contexts cause the model to lose focus on recent/relevant info
4. **Increases latency** — More input tokens = slower time-to-first-token
5. **Increases cost** — Linear cost scaling with conversation length

### What Does NOT Work (Proven Dead Ends)

| Approach | Why It Fails |
|----------|-------------|
| Base64 encoding of messages | Tokenizers still split b64 into multiple tokens; 33% size INCREASE |
| Forcing LLM to "think in base64" | Model generates b64 char-by-char (more tokens), accuracy collapses |
| Arbitrary compression (gzip, etc.) | Binary output is not in vocabulary; model cannot interpret compressed bytes |
| Single-token encoding of sentences | Vocabulary is fixed at training time; you cannot inject new token IDs |

### What DOES Work (Implement These)

The following approaches are proven viable and should be implemented as a 
composable pipeline:

---

## ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────┐
│                     USER APPLICATION                          │
└─────────────────────┬───────────────────────────────────────┘
                      │ (raw messages)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              TOKEN EFFICIENCY LAYER (you build this)          │
│                                                               │
│  ┌───────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │ Codebook  │  │  Summarizer  │  │  Prompt Compressor    │ │
│  │ Manager   │  │  (rolling)   │  │  (LLMLingua-style)    │ │
│  └─────┬─────┘  └──────┬───────┘  └───────────┬───────────┘ │
│        │               │                       │             │
│        ▼               ▼                       ▼             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Context Assembly Engine                      │ │
│  │  (combines compressed history + codebook + RAG refs)      │ │
│  └─────────────────────────┬───────────────────────────────┘ │
│                            │                                  │
└────────────────────────────┼──────────────────────────────────┘
                             │ (optimized prompt)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      LLM API (Claude, GPT, etc.)             │
└─────────────────────────────────────────────────────────────┘
```

---

## COMPONENT 1: CODEBOOK MANAGER

### Purpose
Define reusable shorthand symbols ("sigils") for frequently-referenced entities.
When the same concept appears 5+ times in a conversation, replace verbose 
references with a compact symbol defined once in the system prompt.

### Specification

```python
class CodebookManager:
    """
    Manages a dynamic codebook of sigils mapped to verbose definitions.
    
    A sigil is a short symbol (e.g., Σ1, §A, ⟨receiver⟩) that maps to a 
    longer definition. The codebook is injected into the system prompt once,
    and sigils replace verbose text throughout the conversation history.
    
    Token economics:
      - Codebook definition cost: ~N tokens per entry (paid once)
      - Per-occurrence savings: (original_tokens - sigil_tokens) per use
      - Break-even: when (uses × savings_per_use) > definition_cost
      - Typically profitable after 3-5 occurrences of the same entity
    """
    
    def __init__(self, max_entries: int = 20, sigil_prefix: str = "Σ"):
        self.entries = {}       # sigil -> definition
        self.usage_count = {}   # sigil -> times referenced
        self.candidate_pool = {} # phrase -> occurrence count (pre-promotion)
    
    def observe(self, text: str) -> None:
        """
        Scan text for repeated phrases that are candidates for codebook entry.
        Uses n-gram frequency analysis (n=3 to n=15 words) to detect repetition.
        Phrases appearing 3+ times with length > 10 tokens become candidates.
        """
        pass
    
    def promote_candidates(self) -> list[str]:
        """
        Promote candidates to codebook entries when break-even threshold is met.
        Returns list of newly created sigils.
        """
        pass
    
    def encode(self, text: str) -> str:
        """
        Replace known codebook phrases in text with their sigils.
        Must be exact-match or semantic-match (configurable).
        """
        pass
    
    def decode(self, text: str) -> str:
        """
        Expand sigils back to full definitions (for human-readable output).
        """
        pass
    
    def render_codebook_prompt(self) -> str:
        """
        Generate the system prompt section that defines all active sigils.
        Format:
          [CODEBOOK]
          Σ1 = "User's Hopper receiver R1955245852 running software U520"
          Σ2 = "Guide error 1031 popup appearing after channel change"
          [/CODEBOOK]
        """
        pass
    
    def estimate_savings(self) -> dict:
        """
        Return current token savings statistics.
        {
            "codebook_overhead_tokens": int,
            "total_tokens_saved": int,
            "net_savings": int,
            "entries_active": int,
            "entries_profitable": int
        }
        """
        pass
```

### Rules
- Maximum 20 codebook entries (beyond this, definition overhead dominates)
- Sigils must be visually distinct from natural language (use Unicode symbols)
- Codebook must be injected at the TOP of the system prompt so it's always in scope
- Entries expire after 10 turns of non-use (garbage collection)
- Never codebook-encode the user's current message (only history)

---

## COMPONENT 2: ROLLING SUMMARIZER

### Purpose
Replace old conversation turns with progressive summaries. Recent turns stay 
verbatim; older turns are compressed into summaries that preserve key decisions, 
facts, and context.

### Specification

```python
class RollingSummarizer:
    """
    Implements a tiered memory system:
    
    TIER 1 (Verbatim): Last N turns (configurable, default 4-6 turns)
        - Full text, no compression
        - This is where active reasoning happens
    
    TIER 2 (Detailed Summary): Turns N+1 through M
        - Key facts, decisions, code snippets preserved
        - Conversational filler removed
        - ~3:1 compression ratio
    
    TIER 3 (Abstract Summary): Turns older than M
        - Only topic, decisions, and critical facts
        - ~10:1 compression ratio
    
    Token budget allocation (configurable):
        - System prompt + codebook: ~2000 tokens
        - Tier 3 summary: ~500 tokens
        - Tier 2 summary: ~1500 tokens  
        - Tier 1 verbatim: ~4000 tokens
        - Current turn + response budget: remainder of context window
    """
    
    def __init__(self, 
                 tier1_turns: int = 5,
                 tier2_turns: int = 10,
                 tier2_budget_tokens: int = 1500,
                 tier3_budget_tokens: int = 500):
        self.tiers = {1: [], 2: "", 3: ""}
        self.full_history = []  # kept for reference, never sent to LLM
    
    def add_turn(self, role: str, content: str) -> None:
        """
        Add a new turn. Triggers tier promotion if tier 1 is full.
        When a turn is promoted from tier 1 → tier 2:
          - Extract: key facts, decisions, named entities, code/commands
          - Discard: pleasantries, acknowledgments, repeated context
          - Preserve: any information that might be referenced later
        """
        pass
    
    def summarize_for_tier2(self, turns: list[dict]) -> str:
        """
        Compress turns into a detailed summary.
        Uses the LLM itself (cheap model like Haiku) for summarization.
        
        Prompt template:
          "Summarize the following conversation turns into a concise but 
           complete record. Preserve: all decisions made, technical facts, 
           file paths, error codes, receiver IDs, action items. 
           Remove: greetings, acknowledgments, restated context.
           Budget: {tier2_budget_tokens} tokens maximum."
        """
        pass
    
    def summarize_for_tier3(self, current_tier2: str, demoted_tier2: str) -> str:
        """
        Merge existing tier3 summary with demoted tier2 content.
        Produces a high-level abstract preserving only critical context.
        """
        pass
    
    def render_context(self) -> list[dict]:
        """
        Assemble the full context for the LLM call.
        Returns messages list:
          [
            {"role": "system", "content": system_prompt + codebook + tier3 + tier2},
            ...tier1 verbatim turns...,
            {"role": "user", "content": current_message}
          ]
        """
        pass
    
    def estimate_tokens(self) -> dict:
        """
        Return current token usage breakdown by tier.
        """
        pass
```

### Rules
- Summarization itself costs tokens — use a cheap/fast model (Haiku, GPT-4o-mini)
- Never summarize the current turn or the immediately preceding assistant response
- Preserve ALL named entities (IDs, paths, error codes) even in tier 3
- If the user references something from tier 2/3, promote it back to tier 1 context
- Track summarization quality: if the LLM asks "what did we discuss about X?" and X 
  is in a summary, that's a quality signal to preserve more detail

---

## COMPONENT 3: PROMPT COMPRESSOR (LLMLingua-style)

### Purpose
Remove informationally redundant tokens from the assembled prompt while 
preserving tokens that carry high information content. This is a final 
optimization pass before sending to the LLM.

### Specification

```python
class PromptCompressor:
    """
    Implements token-level compression inspired by LLMLingua (Microsoft, 2023).
    
    Core idea: Use a small language model to compute per-token perplexity.
    High-perplexity tokens are "surprising" and carry information.
    Low-perplexity tokens are predictable and can be removed.
    
    Example:
      Input:  "The user previously asked about deploying a Kubernetes service 
               with three replicas using Helm charts on the production cluster"
      Output: "user asked deploying Kubernetes service three replicas Helm production"
      
      The LLM can still understand the compressed version because the 
      high-information tokens (nouns, numbers, technical terms) are preserved.
    
    Compression targets:
      - Articles (the, a, an) — almost always removable
      - Filler phrases ("I think that", "it seems like", "as we discussed")
      - Redundant prepositions when meaning is clear from word order
      - Repeated information within the same context block
    
    NEVER compress:
      - Code blocks (syntax matters)
      - Error messages (exact text matters for debugging)
      - IDs, paths, URLs (every character matters)
      - The current user message (must be verbatim)
      - The system prompt (model instructions must be precise)
    """
    
    def __init__(self, 
                 target_ratio: float = 0.6,  # Keep 60% of tokens
                 small_model: str = "gpt2",  # Perplexity scorer
                 protected_patterns: list[str] = None):
        self.target_ratio = target_ratio
        self.protected_patterns = protected_patterns or [
            r'```.*?```',           # Code blocks
            r'R\d{10}',            # Receiver IDs
            r'/[\w/]+\.\w+',     # File paths
            r'https?://\S+',       # URLs
            r'\b[A-Z]{2,}[-_]\d+', # Error codes
        ]
    
    def compress(self, text: str, target_ratio: float = None) -> str:
        """
        Compress text to target ratio while preserving meaning.
        Protected patterns are never modified.
        Returns compressed text.
        """
        pass
    
    def compute_token_importance(self, text: str) -> list[tuple[str, float]]:
        """
        Score each token by information content (inverse perplexity).
        Returns list of (token, importance_score) pairs.
        """
        pass
    
    def evaluate_fidelity(self, original: str, compressed: str) -> float:
        """
        Estimate semantic preservation (0.0 = total loss, 1.0 = perfect).
        Uses embedding similarity between original and compressed.
        """
        pass
```

### Implementation Notes
- For a production system, you can use a simpler heuristic approach instead of 
  running a perplexity model:
  - Remove stop words from historical context
  - Remove conversational filler ("I see", "Got it", "Sure")
  - Collapse whitespace and formatting
  - De-duplicate repeated phrases
- The full LLMLingua approach (perplexity-based) gives better results but 
  requires running a small model on every request

---

## COMPONENT 4: CONTEXT ASSEMBLY ENGINE

### Purpose
Orchestrate all components into a single optimized prompt that fits within 
the token budget while maximizing information density.

### Specification

```python
class ContextAssemblyEngine:
    """
    Master orchestrator that combines all compression components.
    
    Token budget strategy:
      Given a context window of W tokens and a desired response length of R:
      Available budget = W - R - safety_margin
      
      Allocation priority (high to low):
        1. System prompt (fixed, non-negotiable)
        2. Current user message (verbatim, non-negotiable)
        3. Codebook definitions (if active)
        4. Tier 1 verbatim history (most recent turns)
        5. Tier 2 detailed summary
        6. Tier 3 abstract summary
        7. RAG-retrieved context (if applicable)
      
      If budget is exceeded, trim from lowest priority upward.
    """
    
    def __init__(self,
                 max_context_tokens: int = 128000,
                 target_response_tokens: int = 4000,
                 safety_margin: int = 500):
        self.codebook = CodebookManager()
        self.summarizer = RollingSummarizer()
        self.compressor = PromptCompressor()
        self.budget = max_context_tokens - target_response_tokens - safety_margin
    
    def process_new_turn(self, role: str, content: str) -> None:
        """
        Ingest a new conversation turn.
        1. Observe for codebook candidates
        2. Add to summarizer history
        3. Trigger tier promotions if needed
        """
        pass
    
    def build_optimized_prompt(self, current_message: str) -> list[dict]:
        """
        Assemble the final optimized prompt for the LLM API call.
        
        Steps:
          1. Render system prompt + codebook definitions
          2. Get tier 3 summary (compressed)
          3. Get tier 2 summary (compressed)
          4. Get tier 1 verbatim turns (codebook-encoded)
          5. Append current message (verbatim, never compressed)
          6. Measure total tokens
          7. If over budget: apply PromptCompressor to tier 2, then tier 1
          8. Final token count verification
        
        Returns: messages list ready for API call
        """
        pass
    
    def get_metrics(self) -> dict:
        """
        Return compression metrics for monitoring.
        {
            "original_tokens": int,      # what we'd send without compression
            "compressed_tokens": int,    # what we actually sent
            "compression_ratio": float,  # compressed/original
            "token_savings": int,        # original - compressed
            "cost_savings_usd": float,   # estimated $ saved at current API rates
            "components": {
                "codebook_savings": int,
                "summarizer_savings": int,
                "compressor_savings": int
            }
        }
        """
        pass
```

---

## IMPLEMENTATION REQUIREMENTS

### Language & Stack
- Python 3.11+
- `tiktoken` for token counting (use cl100k_base for Claude/GPT-4 approximation)
- `sentence-transformers` for semantic similarity (fidelity checks)
- Optional: `transformers` + GPT-2 for perplexity-based compression
- Async-compatible (the LLM API calls for summarization should be async)

### Testing Strategy
1. **Unit tests** for each component in isolation
2. **Integration test**: Feed a 50-turn conversation through the pipeline, verify:
   - Output is valid messages list
   - Token count is within budget
   - No information critical to the last 5 turns is lost
   - Codebook entries are correctly defined and referenced
3. **Quality test**: Run the same multi-turn conversation with and without compression,
   compare LLM response quality (human eval or automated coherence scoring)
4. **Regression test**: Ensure protected patterns (IDs, code, URLs) are never modified

### Configuration
All thresholds should be configurable via a YAML/JSON config:

```yaml
token_efficiency:
  context_window: 128000
  target_response_tokens: 4000
  safety_margin: 500
  
  codebook:
    max_entries: 20
    sigil_prefix: "Σ"
    min_occurrences_to_promote: 3
    expiry_turns: 10
    
  summarizer:
    tier1_verbatim_turns: 5
    tier2_budget_tokens: 1500
    tier3_budget_tokens: 500
    summarization_model: "claude-haiku"  # cheap model for summaries
    
  compressor:
    target_ratio: 0.6
    method: "heuristic"  # or "perplexity" for LLMLingua-style
    protected_patterns:
      - '```.*?```'
      - 'R\d{10}'
      - '/[\w/]+\.\w+'
```

---

## EXPECTED OUTCOMES

When fully implemented, this system should achieve:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Token reduction (20-turn conv) | 40-60% fewer tokens | Compare with/without |
| Token reduction (50-turn conv) | 60-80% fewer tokens | Compare with/without |
| Semantic fidelity | >0.90 cosine similarity | Embedding comparison |
| Response quality degradation | <5% on eval benchmarks | A/B test |
| Latency overhead | <200ms per turn | Pipeline processing time |
| Cost reduction | 40-70% per conversation | API billing comparison |

---

## DELIVERABLES

1. `src/codebook_manager.py` — Complete CodebookManager implementation
2. `src/rolling_summarizer.py` — Complete RollingSummarizer implementation
3. `src/prompt_compressor.py` — Complete PromptCompressor implementation
4. `src/context_assembly.py` — Complete ContextAssemblyEngine implementation
5. `src/token_counter.py` — Utility for accurate token counting
6. `src/config.py` — Configuration loader
7. `tests/` — Comprehensive test suite
8. `config.yaml` — Default configuration
9. `README.md` — Setup, usage, and integration instructions

---

## INTEGRATION PATTERN

The consuming application integrates like this:

```python
from token_efficiency import ContextAssemblyEngine

# Initialize once per conversation session
engine = ContextAssemblyEngine.from_config("config.yaml")

# On each turn:
engine.process_new_turn("user", user_message)

# Build the optimized prompt
optimized_messages = engine.build_optimized_prompt(user_message)

# Send to LLM API
response = await llm_client.chat(messages=optimized_messages)

# Record assistant response
engine.process_new_turn("assistant", response.content)

# Decode any sigils in the response before showing to user
display_text = engine.codebook.decode(response.content)
```

---

## CONSTRAINTS & WARNINGS

1. **NEVER compress the current user message** — it must be verbatim
2. **NEVER compress system instructions** — model behavior depends on exact wording
3. **NEVER lose named entities** — IDs, paths, error codes must survive all compression
4. **Summarization costs tokens** — use the cheapest viable model; cache summaries
5. **Test with YOUR specific use case** — compression that works for casual chat may 
   fail for technical investigations where precision matters
6. **Monitor quality** — if users start saying "I already told you X", your 
   summarization is too aggressive
7. **Graceful degradation** — if any component fails, fall back to uncompressed 
   (correctness > efficiency)

---

## RESEARCH REFERENCES

- LLMLingua: Microsoft, 2023 — "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models"
- Gisting: Mu et al., 2023 — "Learning to Compress Prompts with Gist Tokens"  
- MemGPT: Packer et al., 2023 — "MemGPT: Towards LLMs as Operating Systems" (tiered memory)
- AutoCompressors: Chevalier et al., 2023 — "Adapting Language Models to Compress Contexts"

---

END OF IMPLEMENTATION BRIEF
