# Tool Execution Observability

## What is authoritative

| Signal | Authoritative? |
| --- | --- |
| Assistant response text | No |
| Chat SSE stream deltas | No |
| Generic "Tools called" log lines | No |
| `tool_execution_audit.v1` events | **Yes** |
| Paired ToolMessage for each emitted call | **Yes** |

## Execution path

```
HTTP chat request
  -> RequestCorrelationMiddleware        (server-owned request_id, ContextVar)
  -> agent graph invocation
  -> call_model()                        (records exact model-facing binding)
  -> model emits tool calls
  -> AuditedToolNode                     (gate decision, enforcement, execution)
  -> paired ToolMessage per emitted call
  -> tool_execution_audit.v1 event per emitted call
  -> visible response completion
```

`AuditedToolNode` replaces the bare `ToolNode` in the primary chat graph
(`app/agent/agents/agentic_rag.py`). Allowed calls still execute through the
real tool object using the standard `tool_call` invocation contract, so
ToolMessage content shape is unchanged.

## Request correlation

`RequestCorrelationMiddleware` accepts a strictly validated `X-Request-ID`
header or generates `req-<uuid4hex>`, and binds a mutable `AuditRequestState`
into a `ContextVar`. The object is shared by reference, so the model node can
record the binding and a sibling graph task still observes it. The request ID is
returned in the `X-Request-ID` response header.

The correlation object is server-owned. It is never exposed as a model-editable
tool parameter, and mutable graph state is never accepted as a tool argument.

The ContextVar is intentionally not reset in a `finally` block: SSE chat
responses keep producing tool calls after middleware `dispatch` returns. Each
request already runs in its own context copy, so nothing leaks across requests.

## Coverage

Covered today: parent-agent tools in the primary chat graph, management
facades, blocked calls, rewritten calls, bounded calls, and executor
exceptions. Entry points without HTTP correlation (cron, tests, background
jobs) still emit events using a detached correlation state, so evidence is
never silently dropped.

Not covered by this layer: MCOP child-graph checkpoint propagation and
parent/child checkpoint state. That belongs to the separate graph-integration
phase. When the binding for a turn was not recorded, `binding_known` is
`false` and the executor inventory is used, so unknown binding state cannot
manufacture false `BLOCKED_UNBOUND_TOOL` decisions.

## Storage

| Property | Value |
| --- | --- |
| Storage path | `var/tool_execution_audit/` (override `DISHCHAT_TOOL_AUDIT_DIR`) |
| Active file | `tool_execution_audit.jsonl` |
| Rotated files | `tool_execution_audit.<UTC stamp>.jsonl` |
| Rotation trigger | active file size + incoming record would exceed max bytes |
| Max bytes | 5,000,000 (`DISHCHAT_TOOL_AUDIT_MAX_BYTES`, floor 1024) |
| Retention | 10 rotated files (`DISHCHAT_TOOL_AUDIT_MAX_FILES`, floor 1) |
| Directory mode | `0700` |
| File mode | `0600` |
| Write mode | single `O_APPEND` write of one bounded line (<= 4096 bytes) |
| Git tracked | No — `var/` is ignored |

`var/` is runtime-only and git-ignored. Audit evidence is never committed.

## Failure behaviour

Audit I/O never blocks or breaks chat execution:

* `ToolExecutionAuditSink.record()` catches every exception, returns `False`,
  and increments `write_failure_count` — the audit-write failure metric.
* At most one warning per 60-second window is logged, and only the exception
  class name is logged, so no recursive logging loop and no data leak occur.
* `AuditedToolNode` additionally guards the sink call, so an injected or
  misconfigured sink cannot turn an audit problem into a chat outage.
* An audit failure can never cause a blocked call to execute: enforcement is
  decided before, and independently of, any audit write.

Check the metric with:

```bash
python scripts/active/query_tool_execution_audit.py --storage
```

## Enforcement switch

`DISHCHAT_TOOL_AUDIT_ENFORCE=0` degrades to observe-only: gate decisions are
still recorded, but blocks are not enforced. It exists purely as an operator
rollback control. The default is enforcement enabled. The active value is
recorded on every event as `enforcement_enabled`.

## Operator inspection

A CLI is used rather than an HTTP endpoint because it adds no network surface
that a prompt could reach.

```bash
python scripts/active/query_tool_execution_audit.py --limit 20
python scripts/active/query_tool_execution_audit.py --request-id req-<hex>
python scripts/active/query_tool_execution_audit.py --thread-digest thr-<hex>
python scripts/active/query_tool_execution_audit.py --tool-call-id call_abc
python scripts/active/query_tool_execution_audit.py --tool-name list_dates
python scripts/active/query_tool_execution_audit.py --decision BLOCKED
python scripts/active/query_tool_execution_audit.py --result-code BLOCKED_HEAVY_UNAUTHORIZED
python scripts/active/query_tool_execution_audit.py --since 2026-08-03T00:00:00Z --until 2026-08-04T00:00:00Z
python scripts/active/query_tool_execution_audit.py --validate --limit 100
python scripts/active/query_tool_execution_audit.py --storage
```

Default limit is 50 and the hard maximum is 500. The CLI cannot return argument
values or tool-result bodies because they were never stored; it additionally
refuses to print a set of forbidden keys as defence in depth.
