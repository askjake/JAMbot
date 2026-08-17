---
document_type: methodology
protocol_id: mcop
version: "1.0"
status: active
priority: primary
date: 2026-06-23
owner: montjac
related_protocols:
  - agent-git-workflow
triggers:
  - complex multi-step task
  - 3 or more independent sub-tasks
  - task generates heavy output per sub-step
  - parallel work can be done
  - context window at risk of bloat
  - sub-tasks involve large file reads or code analysis
  - sub-tasks involve running tests or shell commands
  - orchestration with fresh context needed
  - agent_spawn_task requested
  - agent_spawn_parallel requested
  - "do these N things"
  - "in parallel"
  - "simultaneously"
  - "independently"
---

# MCOP — Multi-Conversation Orchestration Protocol

**Protocol ID:** `mcop`
**Version:** 1.0
**Date:** 2026-06-23
**Owner:** montjac
**Status:** ACTIVE

---

## 1. PURPOSE

MCOP is the orchestration protocol for delegating independent sub-tasks to isolated
child conversations. Each child runs in a fresh LangGraph graph with a clean context
window, full tool access, and a shared workspace filesystem. The parent agent spawns
children, waits for their results, and synthesizes the combined output.

The primary motivation is context hygiene. A complex task with 3–5 heavy sub-steps
(large file reads, shell command output, test runs) can easily consume 80–120k tokens
if executed sequentially in a single context window. MCOP distributes that load across
separate contexts, each of which sees only its own task and result, leaving the parent
free for synthesis.

MCOP is not a concurrency mechanism for speed alone. It is a context isolation
mechanism that happens to support parallelism as a by-product.

---

## 2. WHEN TO USE MCOP

### Spawn when ALL of the following are true

- There are **3 or more** independent sub-steps in the current task
- Each sub-step is **self-contained** — it does not need the intermediate reasoning
  of another sub-step, only potentially its output file or artifact
- Each sub-step is likely to generate **heavy output** (large file reads, directory
  listings, shell output, test results, log analysis, code analysis)
- The combined output of all sub-steps in a single context would cause **significant
  context bloat** or risk hitting the iteration limit

### Do NOT spawn when

- The task has fewer than 3 independent sub-steps — just do it inline
- Step B requires Step A's intermediate reasoning, not just a saved artifact
- The sub-task is trivial (a single tool call, a short lookup)
- The user explicitly asked for a sequential, step-by-step walkthrough
- The task is primarily conversational or advisory

### Prefer spawning when

- A single task would plausibly exceed 40k tokens of accumulated tool output
- The agent is mid-task and approaching the iteration limit with work remaining
- Sub-tasks involve independent file analysis, code scanning, or multi-tool pipelines
  that each stand alone

---

## 3. SPAWN MODES

### agent_spawn_task — Single child

Use for one isolated sub-task when you want to delegate a specific unit of work and
read the result before deciding next steps.

```
agent_spawn_task(
    chat_id=<current chat_id>,
    task_id="descriptive_snake_case_id",
    prompt="Fully self-contained prompt. Must not reference 'the conversation above'
            or assume any context. Must include all paths, repo names, and parameters
            the child needs.",
    timeout=300
)
```

The call returns immediately with `{"status": "started", "task_id": "..."}`.
Poll with `agent_check_tasks`, then read with `agent_read_task_result`.

### agent_spawn_parallel — Multiple concurrent children

Use to kick off 2–5 independent children simultaneously. Each child is fully isolated.

```
agent_spawn_parallel(
    chat_id=<current chat_id>,
    tasks=[
        {"task_id": "task_a", "prompt": "...", "timeout": 300},
        {"task_id": "task_b", "prompt": "...", "timeout": 300},
        {"task_id": "task_c", "prompt": "...", "timeout": 300}
    ]
)
```

Returns `{"started": ["task_a", "task_b", "task_c"]}`.

---

## 4. WRITING CHILD PROMPTS

Child prompts must be fully self-contained. A child has **no memory** of the parent
conversation. Every piece of context the child needs must be embedded in the prompt.

### Required elements

- The exact task, stated plainly in the first sentence
- All file paths, repo URLs, branch names, or workspace paths the child needs
- The expected output format (markdown table, JSON file, prose summary, etc.)
- The `chat_id` to pass to `agent_*` tools (children share the parent workspace)
- Any constraints (e.g., "do not modify files", "save output to X.md")

### Prohibited elements

- References to "the above conversation", "as discussed", "the repo we cloned earlier"
- Assumed shared state that was set up by the parent but not restated
- Ambiguous pronouns with no antecedent in the prompt itself

### Example: good child prompt

```
Clone the repo at montjac@10.79.85.35:/home/jakebot/Jakes-agent on branch montjac
into your workspace using chat_id=<id>. Read app/agent_mode/child_conversation.py
in full. Produce a Mermaid flowchart diagram (markdown format) showing the LangGraph
state machine: all nodes, edges, conditional routing, and routing logic. Save the
result to mermaid_diagram.md in the workspace root. Return the diagram in your response.
```

### Example: bad child prompt

```
Read the file we were just looking at and make a diagram of it.
```

---

## 5. POLLING AND RESULT COLLECTION

After spawning, poll until all children are complete before synthesizing.

```
1. Call agent_check_tasks(chat_id=<id>, task_ids=["task_a", "task_b", "task_c"])
2. If any status is "running" or "pending" — wait and re-poll (the tool handles
   the wait internally; call it once and it will block until completion or timeout)
3. Once all statuses are "complete" or "failed":
   - Call agent_read_task_result(chat_id=<id>, task_id="task_a") for each
   - Note which children failed and handle gracefully
4. Synthesize all results into the final response
```

A failed child should be reported clearly in the synthesis: what it attempted, what
error it returned, and whether a retry or manual follow-up is needed.

---

## 6. WORKSPACE SHARING

All children share the parent's workspace filesystem under
`/tmp/dish_chat_agent/<chat_id>/`. This means:

- If Child A clones a repo, Child B and Child C can read from it directly — but
  this only applies if Child A is guaranteed to finish before B and C need the files
- When using `agent_spawn_parallel`, do not assume a shared file is available to
  a sibling unless the file existed before spawning (e.g., a repo already on disk)
- Safe pattern: spawn children that each clone independently, or pre-clone before
  spawning children that only read

Artifacts created by children (via `agent_list_artifacts`) are available to the
parent and to other children via the same path.

---

## 7. DEPTH AND RECURSION LIMITS

Children are configured with `MCOP_CHILD_MAX_ITERS = 5` iterations maximum.

Children **do not have access to spawn tools** — the tools `agent_spawn_task` and
`agent_spawn_parallel` are excluded from the child tool set. This is intentional.
Recursive spawning is not supported. If a child's sub-task is itself complex enough
to warrant spawning, restructure the parent's decomposition instead.

A maximum of `MCOP_MAX_CHILDREN = 5` active children per parent. Attempting to spawn
a 6th child while 5 are active will queue or reject the request.

---

## 8. RESULT QUALITY AND FAILURE HANDLING

### Expected result format

A well-formed child result contains:
- A direct answer or output for the task
- Any artifacts saved to disk (named explicitly)
- A brief self-summary of what was done and what was found

### Handling child failure

If `status == "failed"` or `status == "timeout"`:
- Read the result anyway — it will contain the error message and traceback
- Assess whether the failure is recoverable (wrong path, missing file) or fundamental
- In the synthesis, report the failure with the error and recommend a retry or
  manual follow-up
- Do not silently omit a failed child from the synthesis

---

## 9. SYNTHESIS GUIDELINES

After all children complete, the parent's synthesis response should:

1. Open with a brief statement of what was orchestrated and what completed vs. failed
2. Present each child's result in a logical order (not necessarily spawn order)
3. Integrate findings across children where relevant (e.g., health concern found in
   one child that relates to a finding in another)
4. Flag any anomalies, failures, or items requiring follow-up
5. Not re-print the full raw child output verbatim — summarize and extract the
   key findings, with paths to artifacts for detail

The synthesis should read as a coherent response, not as a dump of three separate
sub-responses stitched together.

---

## 10. AUDIT TRAIL

Each spawned child leaves an audit trail in the workspace:

```
/tmp/dish_chat_agent/<chat_id>/_mcop/
  <task_id>/
    prompt.txt      — the exact prompt sent to the child
    result.json     — status, result text, timing, error if any
```

This trail persists for the lifetime of the workspace and can be inspected to debug
unexpected child behavior or review what a child was asked to do.

---

## 11. CONSTRAINTS AND PROHIBITIONS

- **Do not spawn for fewer than 3 independent sub-tasks.** Below that threshold, the
  overhead of spawning exceeds the context savings.
- **Do not write child prompts that reference the parent conversation.** Children have
  no memory. Ambiguous prompts produce hallucinated context.
- **Do not use spawning to avoid doing work.** If the task requires reasoning that
  only the parent has accumulated, do it in the parent context.
- **Do not spawn recursively.** Children cannot spawn further children. Restructure
  at the parent level if needed.
- **Do not ignore failed children.** Always report failures explicitly in the synthesis.
- **Do not exceed 5 concurrent children.** Prefer 3 when possible to avoid saturating
  the model inference capacity on the server.
- **Do not assume workspace state across parallel children** unless that state was
  established before the spawn call.

---

## 12. RELATED DOCUMENTS

- `app/agent_mode/child_conversation.py` — child graph implementation
- `app/agent_mode/mcop_tools.py` — spawn, parallel, check, and read tool implementations
- `app/agent_mode/adaptive_system_prompt.py` — orchestration hint injected at iteration 0
- `app/config.py` — `MCOP_ENABLED`, `MCOP_MAX_CHILDREN`, `MCOP_PARALLEL_LIMIT` settings
- `docs/guides/CHEAT_SHEET.txt` — quick reference including MCOP trigger summary

---

## 13. REVISION HISTORY

| Version | Date | Author | Notes |
|---------|------|--------|-------|
| 1.0 | 2026-06-23 | montjac | Initial version, derived from MCOP implementation |
