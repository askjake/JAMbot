# Secret Rotation Notes

No secret values are printed here.

## Changes made

- Active config no longer embeds bearer-token literals in MCP headers.
- Ollama sidecar config no longer embeds the pasted beta-report bearer literal.
- Config now references environment variables for bearer-style headers and existing auth keys.

## Environment variables referenced

- `MASTER_KEY`
- `GRASSHOPPER_AUTH_KEY`
- `BETAREPORT_MCP_BEARER_TOKEN`
- `JIRA_MCP_BEARER_TOKEN`
- `CONFLUENCE_MCP_BEARER_TOKEN`
- `STBHEALTH_MCP_BEARER_TOKEN`
- `QODO_CONTEXT_MCP_BEARER_TOKEN`
- `NETRA_MCP_BEARER_TOKEN` if the commented Netra config is re-enabled

Rotate any token that may have existed in prior committed config or sidecar files before deployment.
