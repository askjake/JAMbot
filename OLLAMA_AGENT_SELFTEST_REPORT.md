# Ollama Agent Selftest Report

## Verdict
PASS_WITH_RISKS.

Added `/rest/api/v1/health/ollama-agent-selftest`.

## Selftest fields

The endpoint reports provider config, Ollama reachability, local model list if accessible, role mapping, `langchain-ollama` import status, model bind status, verifier status, tool registry status, prompt identity status, methodology status, MCOP packet status, evidence ledger status, context budget status, AWS refresh skip status, and backend startup route status.

A redacted sample JSON is stored at `reports/ollama_agent_selftest_sample_diship.json`.
