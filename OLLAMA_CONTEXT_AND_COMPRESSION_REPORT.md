# Ollama Context and Compression Report

## Verdict
PASS.

Context budgeting now derives from the active model role and uses a 64k local target unless overridden.

## Sample budget

Default primary role selftest reported configured context `65536`, output reserve `8000`, history budget `42791`, and tool result budget `2674`.

## Changes

- Added `effective_context_budget()`.
- Dynamic tool budget uses role-derived context.
- Active chat trimming uses provider-aware budget.
- Context selftest passes.
