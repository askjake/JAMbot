# CURRENT_OLLAMA_RECONCILIATION_GATE

Timestamp: 2026-07-09 sandbox validation pass  
Mode: read-first reconciliation gate; active code inspected before any code patching in this pass.

## Executive verdict

**MATCHES_SUBSTANTIAL_ACTIVE_OLLAMA_CLAIM, WITH HARDENING GAPS.**

This uploaded `jakebot/` artifact is **not** the older sidecar-only bundle. The active backend contains first-class Ollama support in `app/config.py`, `requirements.txt`, `app/core/llm/chat_models.py`, `app/core/llm/model_roles.py`, active Agent Mode role calls, active prompt identity cleanup, active methodology selection, structured orchestration packet helpers, a verifier fallback, context-budget helpers, and `/rest/api/v1/health/ollama-agent-selftest`.

It is **not yet proven live-ready** from this sandbox because `langchain_ollama`, `langchain_core`, and `psycopg_pool` are not installed here, no live Ollama daemon was reachable from the sandbox, and no full FastAPI startup or live Agent Mode smoke test has run in this artifact yet.

## Repository root and active ASGI entrypoint

| Item | Finding | Classification |
|---|---|---|
| Repo root | `/mnt/data/jakebot_reconcile_work/jakebot` | active required |
| Active ASGI app | `app/main.py`, `app = FastAPI(...)`; launch hint `uvicorn.run("main:app", host=settings.FASTAPI_HOST, port=settings.FASTAPI_PORT)` | active required |
| Health router | `app/main.py` includes `app.health.router` under `settings.API_PREFIX`; active selftest route is therefore `/rest/api/v1/health/ollama-agent-selftest` | active required |

## Active-code evidence table

| Requirement / grep | Active evidence | Classification | Result |
|---|---|---|---|
| `PLLM_PROVIDER=ollama` accepted | `app/config.py` declares `Literal["aws-bedrock", "openai", "anthropic", "ollama"]` with default `ollama`; config validation command exited 0 | active required | PASS |
| `requirements.txt` includes `langchain-ollama` | `requirements.txt:5` contains `langchain-ollama>=0.3.0` | active required | PASS |
| `ChatOllama` active factory | `app/core/llm/chat_models.py` imports `ChatOllama` inside `_create_ollama_model()` and returns `ChatOllama(**kwargs)` | active required | PASS |
| Model roles | `app/core/llm/model_roles.py` defines `primary`, `efficient`, `complex`, `tool_worker`, `analyst`, `verifier`, `title`, `summary` | active required | PASS |
| Agent Mode role call | `app/agent_mode/agent.py` calls `get_model(role="complex")`; no active `get_model(model_arn=...)` hit found in Agent Mode | active required | PASS |
| Active chat graph role call | `app/agent/agents/agentic_rag.py` uses `choose_model_role_for_context`, `get_model(role=selected_role)`, and `get_tool_model(role="tool_worker")` | active required | PASS |
| Prompt identity | `app/agent/agents/prompts/chat_system_prompt.txt` and `app/agent_mode/adaptive_system_prompt.py` identify as Dish-Chat backed by configured local LLM provider and include evidence/tool discipline | active required | PASS |
| Methodology selector | `app/agent/methodology.py` defines all required methodology templates and `select_methodology()` | active required | PASS |
| Tool registry inventory | `app/agent/agents/tools/registry.py` defines `get_tool_inventory()`, duplicate detection, prompt drift detection, and provider-name policy | active required | PASS |
| Provider-specific tool-name sanitation | Registry applies Bedrock 64-char truncation only when the active provider requires it; Ollama policy preserves stable names | active required | PASS |
| Structured MCOP packet helpers | `app/agent_mode/orchestration_packets.py` defines `ToolEvidencePacket`, `AnalysisPacket`, `VerifierReport`, `RUN_STATE.json`, `EVIDENCE_LEDGER.jsonl`, `VERIFIER_REPORT.json`, and packet selftest | active required | PASS |
| MCOP child packet path | `app/agent_mode/child_conversation.py` requires a ToolEvidencePacket JSON final response and writes/normalizes packet data | active required | PASS |
| Verifier fallback | `verify_final_answer_against_packets()` and `app/agent_mode/orchestrator.py::Verifier.audit()` exist | active required but currently fallback/static | PASS_WITH_RISKS |
| Context budget | `app/message/compression.py::effective_context_budget()` derives budget from active model role; `app/tools/context_budget.py` exposes selftest | active required | PASS |
| Ollama selftest endpoint | `app/health/router.py` defines `/health/ollama-agent-selftest` | active required | PASS |
| `get_model(model_arn` | No active non-backup call found under active Agent Mode/chat paths; active factory still accepts `model_arn` as Bedrock-only compatibility | active legacy but harmless if not called in Ollama path | PASS |
| `use_opus` / `should_use_opus` | Active factory keeps `use_opus` compatibility shim; active `complexity_detector.py` keeps `should_use_opus_for_context()` wrapper but active chat graph does not call it | active legacy and confusing | NEEDS_CLEANUP |
| Claude/Sonnet/Opus/Anthropic/Bedrock text | Active prompts are clean; active comments/docs/log labels in `complexity_detector.py`, `app/main.py`, `app/opus_metrics_router.py`, and Bedrock compatibility modules still contain legacy brand language | active legacy and confusing | NEEDS_CLEANUP |
| Hardcoded bearer literal in active config | Active `app/config.py` uses `_bearer_headers("QODO_CONTEXT_MCP_BEARER_TOKEN")`; no active hardcoded token value found in active config. Active helper and tools contain `f"Bearer {token}"` env/runtime headers, not literals | active required/security-sensitive | PASS |

## Static checks run before patching

```bash
pwd
rg -n "PLLM_PROVIDER" app/config.py
rg -n "langchain-ollama" requirements.txt
rg -n "ChatOllama" app/core/llm
rg -n "def get_model" app/core/llm
rg -n "get_model\(model_arn" app --glob '!*.bak*' --glob '!*.backup*' --glob '!*.md'
rg -n "use_opus|should_use_opus" app --glob '!*.bak*' --glob '!*.backup*' --glob '!*.md'
rg -n "Claude|Sonnet|Opus|Anthropic" app/agent app/agent_mode app/health app/main.py --glob '!*.bak*' --glob '!*.backup*' --glob '!*.md'
rg -n "authorization header literal" app --glob '!*.bak*' --glob '!*.backup*' --glob '!*.md'
rg -n "ToolEvidencePacket|AnalysisPacket|VerifierReport|RUN_STATE|EVIDENCE_LEDGER" app
rg -n "ollama-agent-selftest" app
PYTHONPATH=. PLLM_PROVIDER=ollama ELLM_PROVIDER=ollama python - <<'PY'
from app.config import Settings
s = Settings()
print('OK', s.PLLM_PROVIDER, s.ELLM_PROVIDER, s.PLLM_MODEL, s.PLLM_API_BASE)
PY
```

Config validation result:

```text
OK ollama ollama deepseek-r1:32b http://10.79.85.35:11434
exit 0
```

Dependency availability in this sandbox:

```text
langchain_ollama: unavailable
langchain_core: unavailable
psycopg_pool: unavailable
boto3: available
```

## Sidecar-only and backup findings

| Path | Finding | Classification |
|---|---|---|
| `app/config.ollama_patched.py` | Sidecar still exists and contains older/default Bedrock-shaped values; active `app/config.py` has absorbed and surpassed the required Ollama provider changes | sidecar only / stale |
| `app/core/llm/chat_models.ollama_patched.py` | Sidecar still exists and is less complete than active role-based factory; active `chat_models.py` has active role support | sidecar only / stale |
| top-level `config.ollama_patched.py` and `chat_models.ollama_patched.py` | Duplicate sidecars outside active package | sidecar only / stale |
| `app/config.py.backup-*`, `app/config.ollama_active.py.bak`, backup prompts | Several backup files contain legacy prompt text or hardcoded bearer-token literals; they are not active imports but should be quarantined | backup/doc only, security-sensitive |

## Active-code failures / risks found at reconciliation gate

1. Legacy routing names are still active in metrics/router/UI names: `app/opus_metrics_router.py` exposes `/internal/opus-routing-stats`, and `app/main.py` labels it as Opus routing metrics.
2. `app/agent/complexity_detector.py` still contains Opus/Sonnet terminology in docstrings, comments, metrics field names (`opus_requests`, `sonnet_requests`, `opus_percentage`), settings fallbacks, and a compatibility wrapper `should_use_opus_for_context()`.
3. `app/agent/agents/agentic_rag.py` still has a user/developer-visible docstring line saying intelligent Opus routing.
4. Verifier gate exists as a deterministic fallback/helper, but the active Agent Mode graph in `app/agent_mode/agent.py` does not obviously route every final answer through the verifier node. Long-workflow verifier use appears available through orchestration helpers, not enforced across every final response.
5. Live readiness is not proven because dependency imports, full backend startup, live Ollama reachability, live MCP inventory, harmless Agent Mode tool smoke, and master → worker → analyst → verifier → final smoke were not run yet in this pass.

## Security findings

- Active `app/config.py` no longer contains hardcoded bearer token values in Qodo MCP config; it builds Authorization from env via `_bearer_headers("QODO_CONTEXT_MCP_BEARER_TOKEN")`.
- Active runtime helpers in `app/tools/internal_tools.py`, `app/tools/grasshopper_tool.py`, and `coverity_assist_gateway.py` build authorization headers from runtime token variables. That is not a hardcoded literal secret, but reports must not print token values.
- Backup files and sidecars contain hardcoded bearer-token-looking strings. Paths were identified without printing values. They should be quarantined or deleted after active equivalence is confirmed, and any real values should be rotated.
- `.env` / `.env.local*` files exist in the bundle. They were not printed. They should be treated as sensitive and excluded from redistributed artifacts unless the deployment intentionally requires sanitized samples.

## Ready for live Ollama testing?

**Not yet as proven by this pass.** Active code is structurally ready enough to continue hardening and targeted verification, but live readiness requires:

1. Install runtime dependencies in the target venv, especially `langchain-ollama`, `langchain-core`, backend DB deps including `psycopg_pool`, and the rest of `requirements.txt`.
2. Run full FastAPI startup on the target deployment host.
3. Confirm `/rest/api/v1/health/ollama-agent-selftest` returns structured JSON with exact pass/fail/unverified fields.
4. Run one harmless Agent Mode read-only tool smoke test.
5. Run one master → worker → analyst → verifier → final smoke test with durable packet files.

## Exact next patch steps

1. Rename active legacy routing metrics from Opus/Sonnet naming to provider-neutral model-role naming while preserving old routes/functions as compatibility aliases where needed.
2. Clean active comments/docstrings in `complexity_detector.py`, `agentic_rag.py`, `app/main.py`, and `opus_metrics_router.py` so Ollama mode does not leak Claude/Sonnet/Opus/Anthropic identity language outside Bedrock-specific compatibility code.
3. Add/update tests proving active routing uses roles, active prompt identity stays provider-neutral, tool inventory/methodology/packet/verifier/context selftests work, and the old `get_model(model_arn=...)` path is not used by Agent Mode.
4. Quarantine stale `.ollama_patched.py` sidecars and old prompt/config backups after confirming active code contains their needed changes.
5. Run targeted tests and static checks in this artifact.
6. Attempt backend import/startup check; record dependency-bound failures exactly.
7. Produce final hardening reports and a patched tarball only after verification.
