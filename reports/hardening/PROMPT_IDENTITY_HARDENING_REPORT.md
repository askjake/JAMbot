# PROMPT_IDENTITY_HARDENING_REPORT

Date: 2026-07-09

## Verdict
PASS.

Active prompt identity is already provider-neutral and this pass confirmed it remained clean.

## Active prompt paths checked

- `app/agent/agents/prompts/chat_system_prompt.txt`
- `app/agent_mode/adaptive_system_prompt.py`
- `app/health/router.py` prompt identity selftest helper

## Required prompt contract

| Requirement | Result |
|---|---|
| Does not identify as Claude/Sonnet/Opus/Anthropic/Bedrock in active prompts | PASS |
| Includes evidence discipline | PASS |
| Includes tool discipline | PASS |
| Includes methodology selection | PASS |
| Includes final verifier requirement | PASS |
| Warns against broad tool fishing / unsupported root-cause claims | PASS |

## Hygiene

Old prompt backup files that could confuse glob/path-based prompt loaders were quarantined out of the patched working tree and listed in `SECRET_AND_SIDECAR_REMOVAL_MANIFEST.md`.

## Verification

```text
rg -n "Claude|Sonnet|Opus|Anthropic|Bedrock" app/agent/agents/prompts/chat_system_prompt.txt app/agent_mode/adaptive_system_prompt.py
# no output

PYTHONPATH=. PLLM_PROVIDER=ollama ELLM_PROVIDER=ollama pytest -q tests/test_ollama_prompt_identity.py
PASS as part of targeted suite: 11 passed
```
