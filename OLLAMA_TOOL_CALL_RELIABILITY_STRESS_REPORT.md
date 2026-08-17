# OLLAMA_TOOL_CALL_RELIABILITY_STRESS_REPORT

## 1. Executive verdict

**TOOL_CALLS_NOT_WORKING in this sandbox verification environment.**

The code path has been patched to force methodology-scoped structured tool binding, retry failed tool-call attempts, and block pseudo-tool/code-stub final answers. However, the minimal real LangChain/Ollama smoke test could not pass here because the sandbox Python environment does not have `langchain_ollama` or `langchain_core` installed. Per protocol, this report does **not** claim live tool calls are fixed until a real deployment smoke test passes.

## 2. Minimal fake-tool test result

Created `scripts/ollama_tool_call_smoke.py` with a local `echo_probe(value: string)` tool returning `{ "echo": value }`. It binds only that fake tool for the 1-tool gate and can also score 5-tool and 20-tool distractor cases.

Command run:

```bash
PYTHONPATH=. python scripts/ollama_tool_call_smoke.py \
  --models llama3.2:latest qwen3 qwen2.5 \
  --tool-counts 1 5 20 \
  --output reports/ollama_tool_call_smoke.json
```

Observed result: **FAIL / NOT LIVE-VERIFIED**. Every model/count case failed before model invocation with `ModuleNotFoundError: No module named 'langchain_ollama'`. The output was saved to `reports/ollama_tool_call_smoke.json` and `reports/ollama_tool_call_smoke.stdout`.

Classification for Phase 0 in this sandbox: **TOOL_CALL_BINDING_UNVERIFIED_DEPENDENCY_MISSING**. In the live backend container, this same script should be the first gate; if it fails with dependencies installed, classify the live issue as `TOOL_CALL_BINDING_BROKEN` or `MODEL_TOOL_CALL_UNRELIABLE` based on whether `AIMessage.tool_calls` is absent or malformed.

## 3. Actual bound-tool instrumentation

Added `app/agent/tool_execution_policy.py` and wired it into `app/agent/agents/agentic_rag.py` and `app/agent_mode/child_conversation.py`.

Instrumentation now logs:

- selected methodology
- selected toolsets
- required toolsets
- bound tool count
- first 20 bound tool names
- whether each of the first 20 tools has an args schema
- sanitized-name status
- duplicate tool names
- model role and model name
- tool-capable flag
- whether `bind_tools()` returned a bound model
- whether the raw response contains `tool_calls`
- the compact tool-choice plan

The active chat graph now calls `binding_audit(...)` before and after model invocation. If a data-investigation prompt reaches a first response with zero tool calls while scoped tools are available, the run is marked `FAILED_TOOL_EXECUTION` and written to `_tool_execution/EVIDENCE_LEDGER.jsonl` under the session workspace.

## 4. Prompt contamination findings

Static grep after the patch found no active prompt-injected full tool catalog patterns outside the detection-policy code itself:

```bash
rg -n "Here are the available tools|available tools:|tool_calls\"\s*:|example schema|json schema|func [A-Za-z0-9_]+\(" app/agent app/agent_mode --glob '!tool_execution_policy.py'
```

Result: no matches, saved to `reports/static_prompt_contamination_grep.txt`.

Patch actions:

- Replaced the verbose Qodo operation catalog in `app/agent/agents/prompts/chat_system_prompt.txt` with a compact instruction to use the selected bound code-intelligence tool family.
- Added the compact rule: “Use bound tools when data is required. Do not describe tool schemas.”
- Updated `app/agent_mode/adaptive_system_prompt.py` so MCOP children are described as receiving methodology-scoped tools, not a broad/full tool catalog.
- The runtime policy prompt exposes only methodology name, selected tool families, first-tool target, and extracted required inputs. It does not include schema bodies or code stubs.

## 5. Methodology-scoped binding result

Implemented exact methodology mapping in `app/agent/methodology.py` and model-binding policy in `app/agent/tool_execution_policy.py`.

Regression prompt A maps to `receiver_reboot_dvr_playback` and initially binds only:

```text
s3_stb_logs, rtr_alerts_mcp
```

Regression prompt B maps to `popup_signal_loss_investigation` and initially binds only:

```text
s3_stb_logs, rtr_alerts_mcp
```

After a prior tool result exists, prompt B follow-up binding expands to:

```text
s3_stb_logs, rtr_alerts_mcp, stbhealth_popups_mcp
```

This preserves the requirement that popup definition lookup happens after log discovery rather than being the first action.

Regression prompt C maps to `qos_ota_switchback_investigation` and initially binds only:

```text
qos_mcp, rtr_alerts_mcp
```

Regression prompt D maps to `viewership_rtr_investigation` and initially binds only:

```text
viewership, rtr_alerts_mcp
```

The graph `ToolNode` still has the broad executor inventory so already-emitted calls can execute, but the model-facing `bind_tools()` call is now dynamically scoped.

## 6. Model/tool-worker comparison result

The comparison script attempted:

- `llama3.2:latest`
- `qwen3`
- `qwen2.5`

Each was tested against 1, 5, and 20 bound tools, but all cases failed before invocation because `langchain_ollama` was unavailable in this sandbox. No model can be recommended from this environment. The live deployment must rerun `scripts/ollama_tool_call_smoke.py`; if `llama3.2:latest` fails more than once and another installed model succeeds, set `MODEL_ROLE_TOOL_WORKER_MODEL` to the best performer and rerun the four regression prompts.

## 7. Regression prompt A result

Static policy result:

```json
{
  "methodology": "receiver_reboot_dvr_playback",
  "candidate_toolsets": ["s3_stb_logs", "rtr_alerts_mcp"],
  "first_tool": "s3 log bundle/search/read tool",
  "required_inputs": {
    "receivers": ["R1911746693"],
    "software_versions": ["U820"],
    "time_windows": ["around 9pm"]
  },
  "forbidden_tools": ["epg_mcp", "qos_mcp", "viewership"]
}
```

Live result: not executed due missing LangChain/Ollama dependencies in this sandbox.

## 8. Regression prompt B result

Static policy result:

```json
{
  "methodology": "popup_signal_loss_investigation",
  "candidate_toolsets": ["s3_stb_logs", "rtr_alerts_mcp"],
  "followup_toolsets_after_log_result": ["s3_stb_logs", "rtr_alerts_mcp", "stbhealth_popups_mcp"],
  "first_tool": "s3 log bundle/search/read tool",
  "required_inputs": {
    "receivers": ["R2200001234", "R1100005678"],
    "time_windows": ["around 2:30pm", "yesterday"]
  },
  "forbidden_tools": ["viewership", "qos_mcp", "dva_mcp", "dva_jam"]
}
```

Live result: not executed due missing LangChain/Ollama dependencies in this sandbox.

## 9. Regression prompt C result

Static policy result:

```json
{
  "methodology": "qos_ota_switchback_investigation",
  "candidate_toolsets": ["qos_mcp", "rtr_alerts_mcp"],
  "first_tool": "qos_get_coverage",
  "required_inputs": {
    "receivers": ["R3300009876"],
    "time_windows": ["around 11:45pm", "last Tuesday"]
  },
  "forbidden_tools": ["dva_mcp", "viewership"]
}
```

Live result: not executed due missing LangChain/Ollama dependencies in this sandbox.

## 10. Regression prompt D result

Static policy result:

```json
{
  "methodology": "viewership_rtr_investigation",
  "candidate_toolsets": ["viewership", "rtr_alerts_mcp"],
  "first_tool": "viewership top-services/watch-hours query",
  "required_inputs": {
    "date_window": "July 1-7, 2026",
    "time_windows": ["July 1-7, 2026"]
  },
  "forbidden_tools": ["qos_mcp", "dva_mcp", "s3_stb_logs"]
}
```

The July 1-7, 2026 date window is preserved explicitly.

Live result: not executed due missing LangChain/Ollama dependencies in this sandbox.

## 11. Verifier failures caught

Implemented runtime guard behavior:

- Data-investigation prompts with relevant bound tools cannot reach a first final answer with zero tool calls.
- Initial zero-tool-call responses write a `FAILED_TOOL_EXECUTION` ledger entry and retry once with only the top 3 relevant tools.
- Retry failure returns: `Tool execution failed; no root cause can be confirmed.`
- Pseudo-code/schema/tool-description answer patterns are detected by `contains_pseudo_tool_response(...)` and covered by tests.

Unit coverage added in `tests/test_tool_execution_policy.py` verifies that code stubs, function definitions, and JSON tool-call suggestions are recognized as blocked patterns.

## 12. Remaining gaps

The main remaining gap is live validation. This sandbox lacks the installed LangChain/Ollama packages required to instantiate `ChatOllama`, bind tools, produce an `AIMessage`, and execute `ToolNode`. Therefore the real smoke criteria remain unproven:

- AIMessage contains a structured tool call.
- ToolNode executes.
- Final answer uses the tool result.
- Regression prompts A-D each perform at least one relevant real tool call.
- Model comparison actually distinguishes `llama3.2:latest`, `qwen3`, `qwen2.5`, or other installed local models.

## 13. Exact next fixes

Run these in the live backend container where dependencies and Ollama are installed:

```bash
PYTHONPATH=. python scripts/ollama_tool_call_smoke.py \
  --base-url "$PLLM_API_BASE" \
  --models llama3.2:latest qwen3 qwen2.5 \
  --tool-counts 1 5 20 \
  --output reports/ollama_tool_call_smoke_live.json
```

If the 1-tool `echo_probe` case fails, stop and fix model/tool binding before running MCP regressions. If the 1-tool case passes but 5/20 fail for `llama3.2:latest`, switch `MODEL_ROLE_TOOL_WORKER_MODEL` to the best-performing local model and rerun.

Then run the four protocol regression prompts through the real chat endpoint and inspect logs for `Tool binding audit` and `Tool response audit`. Acceptance requires no code stubs, no schema descriptions, no JSON tool-call suggestions, preserved date/time windows, and at least one relevant real tool call for each investigation prompt.

## Local verification completed

```bash
pytest -q tests/test_tool_execution_policy.py tests/test_ollama_methodology_packets.py tests/test_ollama_source_contracts.py tests/test_ollama_model_roles.py
# 13 passed

python -m py_compile \
  app/agent/methodology.py \
  app/agent/tool_execution_policy.py \
  app/agent/agents/agentic_rag.py \
  app/agent_mode/child_conversation.py
# pass
```
