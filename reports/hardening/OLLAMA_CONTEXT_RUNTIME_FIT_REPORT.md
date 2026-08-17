# OLLAMA_CONTEXT_RUNTIME_FIT_REPORT

Date: 2026-07-09

## Verdict
PASS_WITH_RISKS.

Context budgeting derives from active model roles and reports role, model, configured context, reserved budgets, and remaining history/tool budgets. This sandbox cannot verify actual Ollama runtime context/offload.

## Static/runtime results in this artifact

| Role | Model | Configured context | History budget | Result |
|---|---:|---:|---:|---|
| primary | deepseek-r1:32b | 65536 | 42791 | PASS |
| complex | deepseek-r1:70b | 65536 | 42791 | PASS |
| tool_worker | llama3.2:latest | 65536 | 46695 | PASS |
| analyst | deepseek-r1:32b | 65536 | 42791 | PASS |
| verifier | deepseek-r1:32b | 65536 | 46695 | PASS |
| title | deepseek-r1:32b | 65536 | 50279 | PASS |
| summary | deepseek-r1:32b | 65536 | 46695 | PASS |

## Selftest result

`context_budget_status.status == pass` in `reports/hardening/selftest_current.json`.

## Runtime recommendations for target host

- Confirm actual context with `ollama ps` after a model is loaded.
- Confirm the target model supports the configured context and has enough VRAM/RAM.
- Keep `OLLAMA_CTX_LEN`, `PLLM_CTX_LEN`, and `ELLM_CTX_LEN` aligned with the selected roles.
- Do not assume 200k context unless explicitly configured and observed in runtime.
- Watch prompt/compression logs for role, model, configured context, reserved output budget, reserved system/tool budget, and remaining history budget.
