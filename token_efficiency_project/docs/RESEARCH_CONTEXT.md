# Research Context & Prior Art

## Why Base64/Binary Encoding Fails for LLMs

### The Tokenizer Bottleneck

LLMs don't see characters — they see token IDs from a fixed vocabulary (typically 
32K-128K entries) learned during training via Byte Pair Encoding (BPE).

When you encode "Hello World" as base64 ("SGVsbG8gV29ybGQ="):
- Original tokenization: ["Hello", " World"] = 2 tokens
- Base64 tokenization: ["SG", "Vs", "bG", "8", "g", "V", "29", "ybG", "Q", "="] = ~10 tokens

**Result: 5x MORE tokens, not fewer.**

### The Comprehension Gap

Even if you could inject base64 as a single token:
- The model's embedding for that token was learned from its training data
- It has NEVER seen base64-encoded semantic content treated as meaningful
- It would be like reading a QR code with your eyes — the format doesn't match the processing

### The Generation Problem

Asking an LLM to "think in base64" means:
- It generates characters one at a time (autoregressive)
- Each base64 character is a separate forward pass
- Error compounds (one wrong char = corrupted decode)
- The model is doing MORE work, not less

## What DOES Work — Theoretical Foundations

### Information Theory Perspective

Shannon's source coding theorem: you can compress data to its entropy rate.
Natural language has ~1.0-1.5 bits per character of entropy (highly redundant).
LLMs implicitly exploit this redundancy — they predict likely next tokens.

The key insight: **compression should happen in the SEMANTIC space, not the CHARACTER space.**

### Gist Tokens (Mu et al., 2023)

Trained special tokens that compress prompt meaning into the model's internal 
representation (embedding space). The model LEARNS to read these compressed 
representations during fine-tuning.

- Requires model fine-tuning (not a prompt engineering solution)
- Achieves 26x compression on some tasks
- Quality degrades gracefully with compression ratio

### LLMLingua (Microsoft, 2023)

Uses a small model (GPT-2) to estimate per-token information content.
Removes low-information tokens while preserving high-information ones.
The target LLM can still understand because it fills in the gaps.

- No fine-tuning required (works with any LLM)
- 2-10x compression
- Small quality loss for most tasks

### MemGPT (Packer et al., 2023)

Treats the context window like an OS memory hierarchy:
- "Registers" = current active context (in-context)
- "Main memory" = recent history (summarized, retrievable)
- "Disk" = full history (external storage, RAG-retrievable)

The LLM itself decides when to page context in/out.

## Practical Token Economics

### Claude Sonnet Pricing (2024-2025)
- Input: $3.00 per million tokens
- Output: $15.00 per million tokens

### Example Savings

A 50-turn technical investigation conversation:
- Uncompressed: ~80,000 input tokens per turn × 50 turns = $12.00
- With 60% compression: ~32,000 input tokens per turn × 50 turns = $4.80
- Savings: $7.20 per conversation

At 1000 conversations/day: $7,200/day = $216,000/month in savings.

### Break-Even Analysis

The compression pipeline costs:
- Summarization calls (Haiku): ~$0.10 per conversation
- Token counting (tiktoken): negligible
- Embedding similarity checks: ~$0.01 per conversation

Net savings per conversation: $7.20 - $0.11 = $7.09 (98.5% of gross savings retained)
