# LEGACY_ROUTING_CLEANUP_REPORT

Date: 2026-07-09

## Verdict
PASS_WITH_RISKS.

Active Ollama/chat paths already used provider-neutral roles. This pass cleaned the remaining confusing active Opus/Sonnet router and metrics labels without removing Bedrock compatibility.

## Changes made

- Replaced active `app.opus_metrics_router` import/use in `app/main.py` with `app.model_routing_metrics_router`.
- Added `app/model_routing_metrics_router.py` with primary endpoints `/internal/model-routing-stats` and `/internal/model-routing-stats/reset`.
- Kept deprecated aliases `/internal/legacy-opus-routing-stats` and `/internal/legacy-opus-routing-stats/reset` for dashboard compatibility.
- Updated `app/agent/complexity_detector.py` metrics from `opus_requests` / `sonnet_requests` to `complex_requests` / `primary_requests`.
- Added `should_use_complex_role_for_context()` and retained `should_use_opus_for_context()` only as a deprecated compatibility wrapper.
- Updated source-contract tests and added `tests/test_ollama_legacy_routing_cleanup.py`.

## Active-code result

| Check | Result |
|---|---|
| Agent Mode calls `get_model(role="complex")` | PASS |
| Active chat path calls `choose_model_role_for_context()` | PASS |
| Active chat path avoids `use_opus` | PASS |
| Active metrics endpoint is provider-neutral | PASS |
| `use_opus` occurrences remain only in compatibility wrappers | PASS_WITH_RISKS |

## Verification

```text
PYTHONPATH=. PLLM_PROVIDER=ollama ELLM_PROVIDER=ollama pytest -q tests/test_ollama_legacy_routing_cleanup.py tests/test_ollama_source_contracts.py
PASS as part of targeted suite: 11 passed
```

## Known risk

Some Bedrock compatibility files and comments still mention Bedrock/Claude family names. They are not prompt identity or active Ollama routing, but should remain isolated to AWS compatibility code.
