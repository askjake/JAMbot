# Ollama Provider Activation Report — DishIP Actual Bundle

## Verdict
PASS_WITH_RISKS.

This actual running bundle already had Ollama enabled in active `app/config.py`, active `ChatOllama` factory support, `langchain-ollama>=0.3.0`, and a tool-capable Ollama model path. I preserved that working control path and added role-native factory support on top.

## What changed

- Added provider-neutral model roles while retaining existing Ollama synthesis/tool model separation.
- `get_model(role=...)` now supports explicit roles and uses tuple cache keys containing provider, role, model, base URL, context length, and max output.
- `get_tool_model(role="tool_worker")` resolves the tool-capable Ollama model.
- AWS credential refresh now explicitly skips pure Ollama chat mode before creating the background task.
- Hardcoded bearer-like values in active config and sidecar config were moved to env-derived headers; values are not printed.

## Verification

- Config validation with `PLLM_PROVIDER=ollama ELLM_PROVIDER=ollama`: exit 0.
- Role resolver imports and resolves all required roles: exit 0.
- Modified module `py_compile`: exit 0.
- Targeted tests: 9 passed.

## Remaining risk

The execution sandbox lacks `langchain_ollama` and `psycopg_pool`, so I verified import/source/config/test behavior here but did not instantiate a real local model from this sandbox. The uploaded bundle itself already contains the runtime dependency declaration.
