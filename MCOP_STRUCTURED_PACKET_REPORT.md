# MCOP Structured Packet Report

## Verdict
PASS_WITH_RISKS.

Patched the existing MCOP path in place.

## What changed

- Child `max_iters` is passed into state and used by node/router.
- Child prompt requires a `ToolEvidencePacket` JSON final response.
- Child result contains facts, inferences, gaps, errors, raw artifacts, packet path, and next step.
- Children call `role="tool_worker"`.
- Artifacts are scoped to the child task directory instead of whole-workspace mtime scans.
- Added repeated identical tool-call stop guard.
- Added parent `agent_read_packet` tool.

## Verification

Source contract tests passed. Full child LangGraph execution remains deployment-side verification.
