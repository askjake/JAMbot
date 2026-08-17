# Tool Registry and Methodology Report

## Verdict
PASS_WITH_RISKS.

The actual registry now has inventory and audit helpers without removing existing MCP tools.

## What changed

- Added `get_tool_inventory()`, `detect_duplicate_tools()`, `detect_prompt_tool_drift()`, and `get_tool_registry_selftest()` to the active registry.
- Bedrock 64-character tool-name truncation is now provider-specific.
- Ollama path preserves stable tool names unless deterministic collision prefixing is required.
- Added deterministic methodology selector and required templates.

## Remaining risk

The sandbox cannot import the full live registry dependencies, so full MCP schema inventory should be verified in the running deployment through the new selftest endpoint.
