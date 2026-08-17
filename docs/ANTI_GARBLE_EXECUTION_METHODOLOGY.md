---
document_type: methodology
protocol_id: anti-garble-execution
version: "1.0"
status: active
priority: sub-protocol
date: 2026-06-24
owner: montjac
parent_protocol: mcop
triggers:
  - output garbling
  - interleaved text
  - context window bloat
  - many sequential tool calls
  - file-heavy output
  - CSV generation
  - multi-file artifact production
  - long sequential workflow
  - 10+ iterations
  - 30+ messages
---

# ANTI-GARBLE EXECUTION METHODOLOGY

**Protocol ID:** `anti-garble-execution`
**Version:** 1.0
**Date:** 2026-06-24
**Owner:** montjac
**Status:** ACTIVE
**Parent Protocol:** `mcop`

---

## 1. PROBLEM STATEMENT

When a single agent context accumulates many messages (30+) with large tool
results and then attempts to generate multiple files or display large data
structures simultaneously, the LLM's output quality degrades. Symptoms include:
interleaved text from multiple sources, incomplete sentences, mixed data from
different files, and garbled formatting.

This happens because the model must attend to an enormous input context while
generating structured output. The attention mechanism loses coherence between
parallel data streams it is trying to produce.

---

## 2. ROOT CAUSES

Root cause 1: Context window saturation. Each tool call adds its full result to
the conversation history. After 10+ tool calls with large results, the context
may exceed 80k tokens. The model's output quality degrades proportionally.

Root cause 2: Multi-file simultaneous generation. When the prompt asks the model
to "produce these 11 files" at the end of a long workflow, the model attempts to
generate all files in one response. With degraded attention, it interleaves content
between files.

Root cause 3: Data replay. If the model tries to reproduce large data (CSV rows,
JSON bodies, hash strings) that exists in its context history, it may confuse
similar-looking data from different sources.

---

## 3. ANTI-GARBLE RULES

These rules MUST be applied to any prompt that:
- Has 5+ sequential phases with tool calls
- Produces 3+ output files
- Accumulates large tool results (>5k tokens per result)
- Runs for 10+ iterations

### Rule 1: File-First Output

At the end of EACH phase, immediately write that phase's artifact to disk using
agent_run_python. Do NOT accumulate results in response text or defer file
creation to the end.

BAD: "I'll collect all results and produce the files at the end."
GOOD: "Phase 3 complete. Writing RETRIEVAL_COMPARISON.csv now."

### Rule 2: Never Display Large Content Inline

After writing a file, report ONLY: filename, size, classification/status, and
1-2 key scalar values. Do NOT reproduce CSV rows, JSON bodies, full hash
strings, or file contents in the response text.

BAD: "Here are the contents of the evidence bundle: {50KB of JSON}"
GOOD: "Wrote EVIDENCE_BUNDLE.json (4,823 bytes). Classification: CONFIRMED."

### Rule 3: Read From Disk, Not Memory

When a later phase needs a prior phase's output, read it from the file on disk
using agent_run_python. Do NOT rely on conversation history for data values.
This prevents the model from confusing similar data across phases.

BAD: "Using the hash from Phase 2 which was abc123..."
GOOD: "Reading PHASE2_RETRIEVAL.json from disk to get the hash values."

### Rule 4: One Phase Per Turn

Complete one phase, write its artifact, report the status line, then proceed.
Do not try to complete multiple phases in a single LLM response. If the phase
requires multiple tool calls, batch no more than 3 per turn.

### Rule 5: Compact Final Report

The final report/summary MUST be classification-only. It lists pass/fail verdicts
with file references, NOT reproductions of the data. The actual data lives in
the written files.

BAD: "Final report: Comment 361875 has body SHA abc123def456... and was
     retrieved at 2026-06-24T15:47:24Z with author..."
GOOD: "Final report:
       Phase 3: COMMENT_BINDING_CONFIRMED (both comments)
       Phase 4: ALL_EXCERPTS_REBOUND (7/7 verified)
       Files: 11 artifacts in workspace/
       Next action: Ready for reviewer assignment"

### Rule 6: Early Stopping on Failure

If a phase produces a blocking classification (FAILED, BLOCKED, etc.), STOP
immediately. Do not continue to subsequent phases. Write the failure artifact
and produce a compact final report with the blocking status.

---

## 4. PROMPT TEMPLATE ADDITIONS

When structuring prompts for workflows that trigger anti-garble rules, add
this block at the top:

```
## EXECUTION RULES (ANTI-GARBLE)

1. At the end of EACH phase, write that phase's artifact to disk immediately.
   Do NOT accumulate results for later.
2. Never display large file contents inline. Report only: filename, size,
   classification, and 1-2 key values.
3. When a later phase needs prior output, read from disk, not conversation
   history.
4. Complete one phase per response turn. Batch ≤3 tool calls per turn.
5. Final report is classification-only with file references.
6. Stop immediately on any blocking failure.
```

For each phase, use this output template:

```
**Output:** Write `FILENAME.ext` to workspace.
**Report:** [classification] — [1-2 key metrics]. [Stop condition if applicable.]
```

---

## 5. WHEN TO USE MCOP INSTEAD

If the workflow has phases that are INDEPENDENT (no data dependency between them),
use MCOP parallel children instead of sequential anti-garble execution:

- 3+ independent phases → MCOP Wave spawn
- Sequential phases with large outputs → Anti-garble rules
- Mixed (some parallel, some sequential) → MCOP for parallel, anti-garble for
  the sequential segments within each child

The two techniques are complementary:
- MCOP prevents context bloat by distributing work across fresh contexts
- Anti-garble prevents output degradation within a single context

---

## 6. DETECTION AND AUTO-CORRECTION

If you observe garbled output during execution:
1. STOP generating
2. Write what you have so far to disk
3. Start a fresh response turn
4. Read back from disk to verify integrity
5. Continue from the last verified phase

If garbling is detected in a child task's output:
1. Mark the child result as GARBLED
2. Re-run that specific child with reduced scope
3. Do not propagate garbled data to the parent

---

## 7. RELATED DOCUMENTS

- `docs/MCOP_METHODOLOGY.md` — parallel execution for independent phases
- `docs/CODE_TOOLS_AND_REPOSITORY_METHODOLOGY.md` — code search execution
- `docs/STB_FIELD_RCA_AND_COHORT_METHODOLOGY.md` — full RCA workflow

---

## 8. REVISION HISTORY

| Version | Date | Author | Notes |
|---------|------|--------|-------|
| 1.0 | 2026-06-24 | montjac | Initial — anti-garble rules, root cause analysis, prompt templates |
