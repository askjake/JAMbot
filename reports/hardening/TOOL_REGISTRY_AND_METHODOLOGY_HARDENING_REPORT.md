# TOOL_REGISTRY_AND_METHODOLOGY_HARDENING_REPORT

Date: 2026-07-09

## Verdict
PASS_WITH_RISKS.

The active source contains live registry inventory and methodology selection, but this sandbox lacks `langchain_core`, so the runtime registry selftest could not be imported here.

## Active-code findings

| Requirement | Evidence/result |
|---|---|
| Tool inventory generated from registry, not pasted list | `app/agent/agents/tools/registry.py` exposes `get_tool_inventory()` |
| Duplicate detector exists | registry selftest source contains duplicate-status handling |
| Provider-specific tool-name sanitation exists | registry source includes Ollama stable-name policy and Bedrock 64-character limit handling |
| Methodology selector exists | `app/agent/methodology.py` |
| Required methodology templates exist | PASS: all 8 observed in selftest |
| Methodology selftest | PASS |
| Runtime registry import in sandbox | UNVERIFIED: `langchain_core` missing |

## Required methodology templates verified

- `repo_code_review`
- `backend_runtime_debug`
- `qos_switchback_investigation`
- `epg_schedule_metadata_check`
- `dva_stb_firmware_workflow`
- `web_internal_research`
- `artifact_generation`
- `generic_engineering`

## Selftest result excerpt

```text
methodology_selector_status pass
tool_registry_duplicate_status unverified: missing langchain_core in sandbox
```

## Required target-host follow-up

Install dependencies from `requirements.txt`, then call `/rest/api/v1/health/ollama-agent-selftest` and confirm `tool_registry_duplicate_status.status == pass`.
