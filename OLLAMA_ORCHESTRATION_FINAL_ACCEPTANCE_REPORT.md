# Ollama Orchestration Final Acceptance Report — DishIP Actual Running Agent Bundle

## 1. Executive verdict
PASS_WITH_RISKS.

The actual uploaded DishIP bundle already had Ollama working in the active provider/model factory. This patch preserves that working path and adds the missing protocol control-plane pieces: provider-neutral model roles, role-based routing, prompt identity cleanup, methodology selection, compact evidence packets, MCOP packet semantics, verifier fallback, role-derived context budgeting, and a JSON health/selftest endpoint.

## 2. Files changed

- `app/config.py`
- `app/config.ollama_patched.py`
- `config.ollama_patched.py`
- `app/core/llm/__init__.py`
- `app/core/llm/chat_models.py`
- `app/core/llm/model_roles.py`
- `app/agent/complexity_detector.py`
- `app/agent/agents/agentic_rag.py`
- `app/agent/agents/prompts/chat_system_prompt.txt`
- `app/agent/agents/tools/registry.py`
- `app/agent/methodology.py`
- `app/agent_mode/agent.py`
- `app/agent_mode/adaptive_system_prompt.py`
- `app/agent_mode/child_conversation.py`
- `app/agent_mode/mcop_tools.py`
- `app/agent_mode/orchestration_packets.py`
- `app/agent_mode/orchestrator.py`
- `app/agent_mode/INTEGRATION_GUIDE.md`
- `app/message/compression.py`
- `app/tools/context_budget.py`
- `app/health/router.py`
- `app/health/schemas.py`
- `tests/test_ollama_model_roles.py`
- `tests/test_ollama_prompt_identity.py`
- `tests/test_ollama_methodology_packets.py`
- `tests/test_ollama_source_contracts.py`

## 3. Current provider/model-role map

Roles resolve under provider `ollama`. Model names resolve to the configured synthesis, complex, and tool-capable local model fields already present in the bundle. Base URLs are redacted from reports.

## 4. Prompt identity result

PASS. Active prompts use: `You are Dish-Chat, an internal engineering assistant backed by the configured local LLM provider.` Active prompt identity grep returned 0 legacy identity hits.

## 5. Tool registry result

PASS_WITH_RISKS. Inventory, duplicate detection, drift detection, and provider-specific sanitation are implemented. Full live MCP schema inventory needs running-deployment verification.

## 6. Methodology selector result

PASS. All required methodology templates exist and tests pass.

## 7. MCOP structured packet result

PASS_WITH_RISKS. Child max-iter state, structured packet requirement, task-scoped artifact handling, repeated-call guard, and `agent_read_packet` are patched. Full LangGraph execution remains live-deployment verification.

## 8. Evidence ledger result

PASS. Pure-Python packet/evidence ledger write/read selftest passes.

## 9. Verifier gate result

PASS_WITH_RISKS. Deterministic verifier schema/fallback is implemented and tested; live LLM verifier run remains deployment verification.

## 10. Context budget result

PASS. Role-derived context budget selftest reports 64k configured context and local-tool result budgeting.

## 11. Health/selftest JSON

Added `/rest/api/v1/health/ollama-agent-selftest`. Redacted sample JSON is stored at `reports/ollama_agent_selftest_sample_diship.json`.

## 12. Commands run with exit codes

- Input-gate static grep/config/import checks: completed before edits.
- Config validation with provider=ollama: exit 0.
- Modified module `py_compile`: exit 0.
- Targeted unit tests: exit 0, 9 passed.
- Direct selftest function call: exit 0.
- Backend import startup check: exit 1 in this sandbox because `psycopg_pool` is unavailable.
- `bash -n restart-dishchat.sh`: exit 0 syntax check. I did not execute the restart script because it targets a deployment path outside the uploaded bundle and would be unsafe in this sandbox.

## 13. What remains unverified

- Real `ChatOllama` binding from this sandbox, because `langchain_ollama` is unavailable here despite being in requirements.
- Full FastAPI startup from this sandbox, because `psycopg_pool` is unavailable here.
- Live MCP registry schema inventory and duplicate check.
- One full Agent Mode harmless-tool run and one master → worker → analyst → verifier → final LangGraph workflow in the target deployment.

## 14. Known risks

- Existing backup files and env files in the uploaded bundle may still contain historical secrets or legacy identity text; active config/prompt paths were patched, and rotation notes were added.
- Increasing the default role context target to 64k can increase local model memory pressure. It is configurable by role/env if deployment needs to lower it.

## 15. Recommended next work

Install/confirm runtime requirements in the deployment venv, run the new selftest endpoint against the live Ollama service, then run one harmless Agent Mode tool workflow and one structured orchestration workflow.

## Static grep summary

```text
hardcoded bearer active config count: 0
active prompt legacy identity count: 0
active get_model(model_arn=) count: 0
active agentic use_opus count: 0
```

## Targeted test output

```text
.........                                                                [100%]
9 passed in 0.57s
```

## Backend startup check in sandbox

```text
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/mnt/data/diship_work/jakebot/app/main.py", line 10, in <module>
    from psycopg_pool import AsyncConnectionPool
ModuleNotFoundError: No module named 'psycopg_pool'
```
