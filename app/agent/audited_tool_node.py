"""Audited tool-execution node.

This node replaces a bare ``ToolNode`` in the primary chat graph.  It is the
only place where a model-emitted tool call becomes a real executor invocation,
which makes it the correct place to generate authoritative evidence.

Responsibilities
----------------
1. Evaluate every emitted call against the exact last model-facing binding and
   the current server-side authorization booleans (``tool_execution_gate``).
2. Enforce the gate decision.  Blocked calls are never executed.
3. Execute allowed calls through the real tool so ToolMessage content shape is
   identical to the upstream ``ToolNode`` contract.
4. Emit exactly one paired ToolMessage per emitted call.
5. Emit exactly one redacted audit event per emitted call, *after* the outcome
   is known, so ``executed``, ``duration_ms`` and ``error_type`` are real.

The audit sink never raises.  An audit failure therefore cannot execute a
blocked call and cannot crash an otherwise safe allowed call.
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Iterable, Mapping, Sequence

from app.agent.tool_execution_audit import (
    DECISION_FAILED,
    RESULT_FAILED_EXECUTOR_EXCEPTION,
    SCOPE_PARENT,
    ToolExecutionAuditSink,
    build_event,
    classify_gate_decision,
    enforcement_enabled,
    get_audit_sink,
    require_audit_request_state,
)
from app.agent.operational_identity import typed_host_attribution
from app.agent.tool_execution_gate import (
    GateDecision,
    evaluate_tool_call,
    make_tool_message,
    paired_result_payload,
)

logger = logging.getLogger(__name__)

_AUTHORIZATION_ARGUMENT_NAMES = {
    "heavy_auth_token",
    "operator_auth_token",
    "authorization_token",
    "mutation_auth_token",
}


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or getattr(tool, "__name__", "") or "")


def _emitted_tool_calls(state: Any) -> list[dict[str, Any]]:
    messages = []
    if isinstance(state, Mapping):
        messages = list(state.get("messages") or [])
    else:
        messages = list(getattr(state, "messages", None) or [])
    if not messages:
        return []
    last = messages[-1]
    calls = getattr(last, "tool_calls", None)
    if not calls and isinstance(last, Mapping):
        calls = last.get("tool_calls")
    return [dict(call) for call in (calls or []) if isinstance(call, Mapping)]


def _bounded_field_names(decision: GateDecision) -> tuple[str, ...]:
    """Rewrites that are numeric-ceiling bounds rather than policy downgrades."""
    policy_rewrites = {"persist", "allow_heavy", *_AUTHORIZATION_ARGUMENT_NAMES}
    return tuple(name for name in decision.rewritten_arguments if name not in policy_rewrites)


def _server_context_identifiers(state: Any, config: Any) -> dict[str, Any]:
    """Collect context identifiers from server-owned graph/config state.

    Values are used only for equality/type checks and are never copied into an
    audit event.
    """
    out: dict[str, Any] = {}
    configurable = {}
    if isinstance(config, Mapping):
        configurable = config.get("configurable") or {}
    elif config is not None:
        configurable = getattr(config, "configurable", {}) or {}
    if isinstance(configurable, Mapping):
        for key in ("chat_id", "workspace_id", "thread_id", "task_id", "child_run_id"):
            if configurable.get(key) is not None:
                out[key] = configurable.get(key)
    if isinstance(state, Mapping):
        for key in ("chat_id", "workspace_id", "thread_id", "task_id", "child_run_id"):
            if key not in out and state.get(key) is not None:
                out[key] = state.get(key)
    return out


def _server_host_attribution(state: Any, config: Any) -> dict[str, Any]:
    """Return typed, host-only attribution supplied by server configuration."""
    candidates: list[Any] = []
    if isinstance(config, Mapping):
        configurable = config.get("configurable") or {}
        if isinstance(configurable, Mapping):
            candidates.append(configurable.get("operational_identity"))
    if isinstance(state, Mapping):
        candidates.append(state.get("operational_identity"))
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return typed_host_attribution(candidate)
    return typed_host_attribution({})


def _server_argument_sources(state: Any, config: Any) -> dict[str, str]:
    """Read bounded server-generated argument provenance, never model text."""
    candidates: list[Any] = []
    if isinstance(config, Mapping):
        candidates.append((config.get("configurable") or {}).get("argument_sources"))
    if isinstance(state, Mapping):
        candidates.append(state.get("argument_sources"))
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return {str(k)[:128]: str(v)[:128] for k, v in list(candidate.items())[:40]}
    return {}


class AuditedToolNode:
    """Gate-enforcing, audit-emitting executor for model-emitted tool calls."""

    def __init__(
        self,
        tools: Iterable[Any],
        *,
        sink: ToolExecutionAuditSink | None = None,
        node_name: str = "tools",
    ) -> None:
        self._tools_by_name: dict[str, Any] = {}
        for tool in tools or ():
            name = _tool_name(tool)
            if name and name not in self._tools_by_name:
                self._tools_by_name[name] = tool
        self._sink = sink
        self._node_name = node_name

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools_by_name))

    def _sink_or_default(self) -> ToolExecutionAuditSink:
        return self._sink or get_audit_sink()

    async def ainvoke(self, state: Any, config: Any = None) -> dict[str, Any]:
        calls = _emitted_tool_calls(state)
        if not calls:
            return {"messages": []}

        request_state = require_audit_request_state()
        enforce = enforcement_enabled()

        # When the model-facing binding for this turn was recorded by the model
        # node, it is authoritative.  When it is unknown (a graph or entrypoint
        # that does not record it), the executor inventory is used so unknown
        # binding state cannot manufacture false "unbound" blocks.
        if request_state.binding_known and request_state.bound_tool_names:
            bound_names: Sequence[str] = request_state.bound_tool_names
        else:
            bound_names = self.tool_names

        context_identifiers = _server_context_identifiers(state, config)
        argument_sources = _server_argument_sources(state, config)
        host_attribution = _server_host_attribution(state, config)
        messages: list[Any] = []
        for call in calls:
            message = await self._handle_call(
                call,
                bound_names=bound_names,
                request_state=request_state,
                enforce=enforce,
                config=config,
                context_identifiers=context_identifiers,
                argument_sources=argument_sources,
                host_attribution=host_attribution,
            )
            messages.append(message)
        return {"messages": messages}

    async def _handle_call(
        self,
        call: Mapping[str, Any],
        *,
        bound_names: Sequence[str],
        request_state: Any,
        enforce: bool,
        config: Any,
        context_identifiers: Mapping[str, Any],
        argument_sources: Mapping[str, str],
        host_attribution: Mapping[str, Any],
    ) -> Any:
        decision = evaluate_tool_call(
            call,
            last_bound_tool_names=bound_names,
            authorization_flags=request_state.authorization_flags,
            context_identifiers=context_identifiers,
            argument_sources=argument_sources,
            execution_constraints=getattr(request_state, "execution_constraints", {}),
        )
        bounded_fields = _bounded_field_names(decision)
        authorization_field_dropped = any(
            name in _AUTHORIZATION_ARGUMENT_NAMES for name in decision.rewritten_arguments
        )
        audit_decision, result_code = classify_gate_decision(
            gate_result_code=decision.result_code,
            allowed=decision.allowed,
            missing_authorizations=decision.missing_authorizations,
            rewritten_fields=decision.rewritten_arguments,
            bounded_fields=bounded_fields,
            authorization_field_dropped=authorization_field_dropped,
        )

        executed = False
        error_type: str | None = None
        duration_ms = 0.0
        message: Any

        blocked = not decision.allowed
        if blocked and not enforce:
            # Observe-only rollback mode: the decision is still recorded, but the
            # call proceeds.  Enforcement is on by default.
            logger.warning(
                "tool_execution_audit: enforcement disabled; recording %s for tool=%s without blocking",
                result_code,
                decision.tool_name,
            )
            blocked = False

        if blocked:
            message = make_tool_message(paired_result_payload(decision))
        else:
            tool = self._tools_by_name.get(decision.tool_name)
            if tool is None:
                unavailable = GateDecision(
                    **{**decision.to_dict(), "allowed": False, "result_code": "BOUND_TOOL_NOT_EXECUTABLE"}
                )
                audit_decision, result_code = classify_gate_decision(
                    gate_result_code="BOUND_TOOL_NOT_EXECUTABLE",
                    allowed=False,
                    missing_authorizations=(),
                )
                message = make_tool_message(paired_result_payload(unavailable))
            else:
                started = time.perf_counter()
                try:
                    message = await self._invoke(tool, call, decision, config)
                    executed = True
                except Exception as exc:  # noqa: BLE001
                    # Only the class name is retained.  Exception text can carry
                    # URLs, headers, credentials, or user data.
                    error_type = type(exc).__name__
                    audit_decision = DECISION_FAILED
                    result_code = RESULT_FAILED_EXECUTOR_EXCEPTION
                    message = make_tool_message(paired_result_payload(decision, error=error_type))
                finally:
                    duration_ms = (time.perf_counter() - started) * 1000.0

        event = build_event(
            request_id=request_state.request_id,
            thread_id_digest=request_state.thread_id_digest,
            tool_call_id=decision.tool_call_id,
            tool_name=decision.tool_name,
            binding_signature=request_state.binding_signature,
            tool_was_bound=decision.tool_name in set(bound_names),
            binding_known=bool(request_state.binding_known),
            required_capabilities=decision.required_authorizations,
            authorization_flags=request_state.authorization_flags,
            decision=audit_decision,
            result_code=result_code,
            executed=executed,
            paired_tool_result=message is not None,
            argument_field_names=decision.original_argument_names,
            rewritten_fields=decision.rewritten_arguments,
            bounded_fields=bounded_fields,
            duration_ms=duration_ms,
            error_type=error_type,
            enforcement_enabled=enforce,
            graph_node=self._node_name,
            # Scope identity travels on the correlation state, so the
            # parent graph and the MCOP child graph share one gate.
            scope=getattr(request_state, "scope", SCOPE_PARENT),
            parent_request_id=getattr(request_state, "parent_request_id", ""),
            child_run_id=getattr(request_state, "child_run_id", ""),
            snapshot_status=getattr(request_state, "snapshot_status", ""),
            snapshot_policy_version=getattr(
                request_state, "snapshot_policy_version", 0
            ),
            snapshot_registry_generation=getattr(
                request_state, "snapshot_registry_generation", ""
            ),
            current_registry_generation=getattr(
                request_state, "current_registry_generation", ""
            ),
            generation_match=getattr(request_state, "generation_match", True),
            **dict(host_attribution),
        )
        # A sink failure is recorded as a metric inside the sink and must not
        # change execution behaviour in either direction.  The extra guard keeps
        # an injected or misconfigured sink from turning an audit problem into a
        # chat outage, and from ever causing a blocked call to run.
        try:
            self._sink_or_default().record(event)
        except Exception as exc:  # noqa: BLE001 - audit must never break chat
            logger.warning(
                "tool_execution_audit sink raised (type=%s); execution outcome unchanged",
                type(exc).__name__,
            )
        return message

    async def _invoke(self, tool: Any, call: Mapping[str, Any], decision: GateDecision, config: Any) -> Any:
        """Execute through the real tool, preserving upstream ToolMessage shape."""
        tool_call = {
            "name": decision.tool_name,
            "args": dict(decision.effective_args),
            "id": decision.tool_call_id or str(call.get("id") or ""),
            "type": "tool_call",
        }
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(tool_call, config=config)
        if hasattr(tool, "invoke"):
            value = tool.invoke(tool_call, config=config)
            return await value if inspect.isawaitable(value) else value
        value = tool(**tool_call["args"])
        return await value if inspect.isawaitable(value) else value


def make_audited_tool_node(
    tools: Iterable[Any],
    *,
    sink: ToolExecutionAuditSink | None = None,
    node_name: str = "tools",
):
    """Return an async LangGraph node function backed by ``AuditedToolNode``."""
    node = AuditedToolNode(tools, sink=sink, node_name=node_name)

    async def audited_tools(state: Any, config: Any = None) -> dict[str, Any]:
        return await node.ainvoke(state, config=config)

    audited_tools.audited_tool_node = node  # type: ignore[attr-defined]
    return audited_tools
