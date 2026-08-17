# Tool Execution Audit Schema

Schema version: `tool_execution_audit.v1`
Module: `app/agent/tool_execution_audit.py`

## Purpose

Chat SSE deltas, assistant response text, and generic "Tools called" log lines
are model-authored or best-effort. They are not proof that a tool executed.
This schema defines independent, server-generated evidence that is authoritative
for tool execution.

Every tool call emitted by the model produces exactly one audit event, written
after the outcome is known, so `executed`, `duration_ms`, and `error_type`
describe reality rather than intent.

## Event shape

```json
{
  "schema_version": "tool_execution_audit.v1",
  "event_id": "tea-<uuid4hex>",
  "timestamp": "2026-08-03T22:30:36Z",
  "request_id": "req-<uuid4hex>",
  "thread_id_digest": "thr-<24 hex>",
  "tool_call_id": "call_abc",
  "tool_name": "diship_backend_tool_inventory_status",
  "binding_signature": "bind-0007-<16 hex>",
  "tool_was_bound": true,
  "binding_known": true,
  "required_capabilities": ["heavy_tools_authorized"],
  "authorization_flags": {
    "operator_authorized": false,
    "heavy_tools_authorized": false,
    "persistence_authorized": false,
    "mutation_authorized": false
  },
  "decision": "ALLOWED",
  "result_code": "EXECUTED",
  "executed": true,
  "paired_tool_result": true,
  "argument_field_names": ["scope"],
  "rewritten_fields": [],
  "bounded_fields": [],
  "duration_ms": 4,
  "error_type": null,
  "enforcement_enabled": true,
  "graph_node": "tools"
}
```

## Field semantics

| Field | Meaning |
| --- | --- |
| `event_id` | Unique audit record identifier. |
| `request_id` | Server-owned HTTP request correlation ID. Never a model argument. |
| `thread_id_digest` | One-way truncated digest of the conversation ID. See below. |
| `tool_call_id` | Identity of the model-emitted call; pairs with the ToolMessage. |
| `binding_signature` | Signature of the exact model-facing binding for the turn. |
| `tool_was_bound` | Whether the called tool was in that binding. |
| `binding_known` | Whether the binding was recorded by the model node for this turn. |
| `required_capabilities` | Authorization capability names the call required. |
| `authorization_flags` | The four server-side booleans. Never tokens. |
| `decision` | `ALLOWED` / `BLOCKED` / `REWRITTEN` / `FAILED`. |
| `result_code` | Enumerated outcome; see table below. |
| `executed` | Whether an executor actually ran. Blocked calls are always `false`. |
| `paired_tool_result` | Whether exactly one ToolMessage was produced. |
| `argument_field_names` | Argument **names** only. Never values. |
| `rewritten_fields` | Field names the gate rewrote (policy downgrades). |
| `bounded_fields` | Field names clamped to a hard ceiling. |
| `duration_ms` | Integer milliseconds of executor time. |
| `error_type` | Bare exception class name, or `null`. Never a message. |
| `enforcement_enabled` | Whether gate blocks were enforced or only observed. |
| `graph_node` | Graph node that produced the event. |

## Decisions

```
ALLOWED     the gate permitted the call and it ran
BLOCKED     the gate refused; nothing executed
REWRITTEN   the call ran with server-rewritten or clamped arguments
FAILED      the call was permitted but the executor raised
```

## Result codes

```
EXECUTED
BLOCKED_UNBOUND_TOOL
BLOCKED_HEAVY_UNAUTHORIZED
BLOCKED_PERSISTENCE_UNAUTHORIZED
BLOCKED_MUTATION_UNAUTHORIZED
BLOCKED_OPERATOR_UNAUTHORIZED
BLOCKED_TOOL_NOT_EXECUTABLE
REWRITTEN_PERSIST_FALSE
REWRITTEN_LIMIT_BOUNDED
REWRITTEN_AUTHORIZATION_FIELD_DROPPED
FAILED_EXECUTOR_EXCEPTION
NO_TOOL_CALL
```

The repository execution gate (`app/agent/tool_execution_gate.py`) keeps its own
established codes (`TOOL_NOT_IN_LAST_BINDING`, `TOOL_AUTHORIZATION_REQUIRED`,
`TOOL_EXECUTION_ALLOWED_WITH_REWRITE`, `BOUND_TOOL_NOT_EXECUTABLE`).
`classify_gate_decision()` is the single mapping point between the two
vocabularies, so gate semantics stay intact while audit semantics stay explicit.

## Absolute redaction contract

Audit events must never contain raw prompt text, assistant text, tool argument
values, tool-result bodies, authorization tokens, authorization hashes, token
prefixes or suffixes, webhook URLs, API keys, database URLs, email addresses,
raw thread IDs, model reasoning, or exception messages.

Two independent guards enforce this:

1. A strict key allowlist (`_ALLOWED_EVENT_KEYS`). Unknown keys are dropped by
   `sanitize_event()` before serialization.
2. A forbidden-key pattern that drops anything matching `token`, `secret`,
   `password`, `credential`, `webhook`, `api_key`, `cookie`, `bearer`, `hash`,
   and similar — **without reading the value**.

Authorization material is never hashed. A digest of a secret is still
secret-derived, so token-shaped fields are dropped by name. `error_type` is
accepted only if it matches a bare class-name pattern; anything message-like
becomes `UnsafeErrorTypeRedacted`.

## Thread digest

`thread_id_digest = "thr-" + sha256("tool_execution_audit.v1:thread:" + thread_id)[:24]`

The digested input is the server-generated conversation/session identifier
only. Thread IDs are internal UUIDs, not credentials. No prompt text, user
attribute, or credential is part of the digest input, so the digest cannot
expose credential material. Its only purpose is grouping events for one
conversation without persisting the raw identifier.

## Validation

`validate_event()` returns a list of problems and is exposed to operators via:

```bash
python scripts/active/query_tool_execution_audit.py --validate --limit 100
```

It rejects wrong schema versions, non-enumerated decisions or result codes,
non-boolean flags, non-allowlisted keys, forbidden keys, non-integer durations,
message-like `error_type` values, and any record claiming execution while
blocked.
