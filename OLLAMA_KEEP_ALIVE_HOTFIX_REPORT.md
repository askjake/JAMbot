# OLLAMA_KEEP_ALIVE_HOTFIX_REPORT

## Verdict
PATCHED_FOR_TARGET_TESTING.

## Why
The backend logs showed repeated Ollama model-role creation and long gaps around model invocation. That does not prove weights are re-downloaded; it can also be Ollama re-loading an unloaded model from disk/CPU/VRAM. The active code did not pass an Ollama `keep_alive` value into `ChatOllama`, so the daemon could unload models on its default lifecycle.

## Changes
- Added `OLLAMA_KEEP_ALIVE` to active config with default `1h`.
- Added role-specific keep-alive resolution via `MODEL_ROLE_<ROLE>_KEEP_ALIVE` and `MODEL_ROLES_CONFIG[role].keep_alive`.
- Added `keep_alive` to `ModelRoleConfig`, role cache keys, and selftest role-map output.
- Passed `keep_alive` into `ChatOllama(**kwargs)`.
- Preserved keep-alive through Bedrock `model_arn` compatibility wrapper construction.
- Added Ollama response metadata logging for `load_duration` and `total_duration` so reload behavior is visible.
- Extended the Ollama selftest endpoint to call `/api/ps`, report running models, expected role models, missing expected models, and expected models not currently running.
- Added tests for default and role-specific keep-alive resolution.

## Verification
- `python -m py_compile app/config.py app/core/llm/model_roles.py app/core/llm/chat_models.py app/agent/agents/agentic_rag.py app/health/router.py tests/test_ollama_model_roles.py` passed.
- `PYTHONPATH=. PLLM_PROVIDER=ollama ELLM_PROVIDER=ollama pytest -q tests/test_ollama_model_roles.py` passed: 3 passed.

## Target-host checks still required
- Confirm `ollama list` contains every role model.
- Confirm `/rest/api/v1/health/ollama-agent-selftest` shows expected role models and running models.
- Confirm backend logs include `keep_alive=1h` when creating Ollama model roles.
- Confirm subsequent model responses report much lower `load_ms` after the first warm call.
- Confirm `ollama ps` shows loaded models with future `expires_at` values.
