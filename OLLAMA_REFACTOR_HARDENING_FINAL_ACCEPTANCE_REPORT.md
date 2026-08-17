# OLLAMA_REFACTOR_HARDENING_FINAL_ACCEPTANCE_REPORT

Date: 2026-07-09

## 1. Executive verdict

READY_WITH_RISKS.

The current uploaded bundle is materially the corrected active Ollama orchestration bundle, not a sidecar-only mismatch. This pass reconciled remaining active gaps: legacy Opus/Sonnet metric/router names, stale sidecars/prompt backups, and an Agent Mode final-answer verifier edge.

It is ready for target-host local Ollama validation. It is not honestly ready for live user traffic until the target deployment passes full backend startup, live selftest, one harmless Agent Mode tool smoke, and one live master → worker → analyst → verifier → final smoke.

## 2. Files changed

Code/test/report changes in this hardening pass:

- `app/main.py`
- `app/model_routing_metrics_router.py` new
- `app/agent/complexity_detector.py`
- `app/agent/agents/agentic_rag.py`
- `app/agent_mode/agent.py`
- `app/config.py` comment-only cleanup
- `tests/test_ollama_source_contracts.py`
- `tests/test_ollama_legacy_routing_cleanup.py` new
- `CURRENT_OLLAMA_RECONCILIATION_GATE.md` new
- `SECRET_AND_SIDECAR_REMOVAL_MANIFEST.md` new
- hardening reports under `reports/hardening/`
- `DISTRIBUTION_SANITIZATION_MANIFEST.md` new

Quarantined out of the patched tree:

- obsolete `.ollama_patched.py` sidecars
- `app/opus_metrics_router.py`
- `app/config.ollama_active.py.bak`
- stale prompt backup files

## 3. Current provider/model-role map

| Role | Provider | Model | Context | Max output | Tool-capable |
|---|---|---:|---:|---:|---|
| primary | ollama | deepseek-r1:32b | 65536 | 8000 | False |
| efficient | ollama | deepseek-r1:32b | 65536 | 4096 | False |
| complex | ollama | deepseek-r1:70b | 65536 | 8000 | False |
| tool_worker | ollama | llama3.2:latest | 65536 | 4096 | True |
| analyst | ollama | deepseek-r1:32b | 65536 | 8000 | False |
| verifier | ollama | deepseek-r1:32b | 65536 | 4096 | False |
| title | ollama | deepseek-r1:32b | 65536 | 512 | False |
| summary | ollama | deepseek-r1:32b | 65536 | 4096 | False |

## 4. Prompt identity result

PASS. Active prompt paths do not claim Claude, Sonnet, Opus, Anthropic, or Bedrock identity. Prompt contract contains evidence discipline, tool discipline, methodology selection, and verifier requirements.

## 5. Legacy routing cleanup result

PASS_WITH_RISKS. Active routing now reports provider-neutral `complex` / `primary` metrics through `/internal/model-routing-stats`. Deprecated compatibility wrappers remain in `chat_models.py` and `complexity_detector.py` but active Agent Mode/chat paths do not call `get_model(model_arn=...)` or `use_opus` routing.

## 6. Secret/sidecar hygiene result

PASS_WITH_RISKS. Active config has no hardcoded bearer-token literal values. Obsolete Ollama sidecars and stale prompt backups were quarantined. Six inactive `app/config.py.backup*` files in the source scan contained token-looking literals; the downloadable tarball excludes backup/env files, and real values should be rotated before redistributing the original bundle. No token values were printed.

## 7. Tool registry/methodology result

PASS_WITH_RISKS. Methodology selector passed and all required templates are present. Registry inventory source is active, but runtime registry selftest is unverified in this sandbox because `langchain_core` is not installed.

## 8. MCOP packet/orchestration result

PASS_WITH_RISKS. Durable packet helpers, child `ToolEvidencePacket` output, analyst packet creation, verifier report writing, and deterministic orchestration smoke all passed. Live Agent Mode orchestration with real tools remains unverified.

## 9. Verifier gate result

PASS_WITH_RISKS. Active Agent Mode now routes no-tool final answers through a verifier node that writes `VERIFIER_REPORT.json` and appends verification limits when warranted. Live LangGraph execution was not possible in this sandbox.

## 10. Context budget result

PASS. Context budgeting derives from resolved model roles and reports configured context plus system/tool/output/history budgets. Actual Ollama offload/context must be confirmed on the target host with `ollama ps`.

## 11. Selftest endpoint JSON summary

Full JSON: `reports/hardening/selftest_current.json`

```json
{
  "provider_config": {
    "PLLM_PROVIDER": "ollama",
    "ELLM_PROVIDER": "ollama",
    "pure_ollama_chat_mode": true
  },
  "ollama_reachable": false,
  "langchain_ollama_available": false,
  "primary_model_bind_status": {
    "role": "primary",
    "status": "skipped",
    "reason": "langchain-ollama import unavailable"
  },
  "tool_worker_model_bind_status": {
    "role": "tool_worker",
    "status": "skipped",
    "reason": "langchain-ollama import unavailable"
  },
  "tool_registry_duplicate_status": {
    "status": "unverified",
    "error": "tool registry selftest: No module named 'langchain_core'"
  },
  "prompt_identity_status": {
    "status": "pass",
    "identity_leaks": [],
    "contract_present": true
  },
  "methodology_selector_status": "pass",
  "mcop_packet_status": "pass",
  "context_budget_status": "pass",
  "aws_refresh_disabled_or_skipped_for_pure_ollama": true,
  "backend_startup_status": "selftest_route_loaded"
}
```

## 12. Backend startup result

NOT VERIFIED / environment-bound failure in this sandbox.

```text
app_main_import_exit=1
ModuleNotFoundError: No module named 'psycopg_pool'
```

`requirements.txt` includes `psycopg[binary,pool]==3.2.9`, so this is a sandbox dependency gap rather than an Ollama-specific code failure.

## 13. Agent Mode smoke test results

- Selftest-only: PARTIAL PASS. Direct selftest function loaded and returned structured JSON, but Ollama reachability and `langchain_ollama` binding were unavailable in this sandbox.
- Harmless read-only tool: NOT RUN. Live LangGraph/LangChain/Ollama dependencies are unavailable here.
- Mini orchestration: PARTIAL PASS. Deterministic master → worker → analyst → verifier → final smoke passed without live LLM/tool calls.

## 14. Commands run with exit codes

See `reports/hardening/verification_commands.log`.

Key results:

```text
config_role_resolution EXIT_CODE: 0
py_compile EXIT_CODE: 0
fake_chatollama_factory EXIT_CODE: 0
selftest_direct EXIT_CODE: 0
app_main_import EXIT_CODE: 1 (missing psycopg_pool in sandbox)
restart_script_syntax EXIT_CODE: 0
orchestration_smoke EXIT_CODE: 0
context_budget_roles EXIT_CODE: 0
targeted pytest: 11 passed, exit 0
```

## 15. Known gaps

1. `langchain_ollama`, `langchain_core`, `langgraph`, and `psycopg_pool` are unavailable in this sandbox.
2. Ollama base URL was not reachable from this sandbox.
3. Full FastAPI backend startup was not proven here.
4. Runtime tool-registry duplicate selftest was unverified because registry import needs `langchain_core`.
5. A real Agent Mode harmless-tool smoke was not run.
6. A live master → worker → analyst → verifier → final smoke with Ollama was not run.
7. Inactive config backup files in the source scan contained token-looking literals; values were not printed, and the downloadable tarball excludes backup/env files.

## 16. Recommended next work

On the target host, run:

```bash
pip install -r requirements.txt
python -c "import langchain_ollama; print('langchain_ollama import OK')"
python -c "import psycopg_pool; print('psycopg_pool import OK')"
ollama list
ollama ps || true
PLLM_PROVIDER=ollama ELLM_PROVIDER=ollama python - <<'PY'
from app.config import Settings
from app.core.llm.model_roles import resolve_all_model_roles
s = Settings()
print(s.PLLM_PROVIDER, s.ELLM_PROVIDER)
for role, cfg in resolve_all_model_roles(s).items():
    print(role, cfg.to_dict())
PY
bash /home/jakebot/Jakes-agent/restart-dishchat.sh
curl -s http://localhost:8000/rest/api/v1/health/ollama-agent-selftest | jq .
```

Then run the three required Agent Mode smoke prompts and confirm packet/verifier files are written.
