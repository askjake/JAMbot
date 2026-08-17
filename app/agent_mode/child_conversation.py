"""
Multi-Conversation Orchestration Protocol (MCOP) - Child Conversation Engine
=============================================================================

Core primitive: Executes a single sub-task in a completely fresh LangGraph
thread with its own context window, sharing the parent's workspace filesystem.

Author: Jacob Montgomery
Created: 2026-06-23
Version: 0.1.0 (MVP)
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Any, Sequence

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from app.config import get_settings
from app.core.llm import get_model
from app.agent.utils import set_model_config
from app.agent.db_utils import get_checkpointer
from app.agent_mode.thought_interceptor import interceptor
from app.agent_mode.task_capability_plan import (
    BLOCKED_CHILD_MISSING_CAPABILITY,
    TaskCapabilityPlan,
    evaluate_child_capability_plan,
)
from app.agent_mode.orchestration_packets import (
    normalize_worker_packet_from_summary,
    try_parse_tool_evidence_packet,
)
from app.agent.tool_execution_policy import (
    append_tool_execution_failure,
    binding_audit,
    build_compact_tool_execution_system_prompt,
    build_tool_execution_failed_message,
    get_all_executor_tools,
    get_scoped_tools_for_prompt,
    rank_tools_for_retry,
    should_force_tool_retry,
)
from app.agent.audited_tool_node import make_audited_tool_node
from app.agent.tool_execution_audit import record_model_facing_binding
from app.agent_mode.child_tool_policy import (
    build_child_policy_snapshot,
    child_audit_scope,
    child_safe_tools,
    current_registry_generation,
    fail_closed_policy,
    narrow_policy_for_task,
    reconcile_snapshot_with_registry,
    validate_child_policy_snapshot,
)

# Children get ALL parent tools EXCEPT the MCOP spawn tools (no recursion).
# Loaded dynamically from the registry so MCP tools (Qodo, JIRA, RTR, S3, etc.)
# are available to children, not just filesystem tools.
# Late import inside _get_child_tools() to avoid circular import

# MCOP tool names to EXCLUDE from children (prevents recursive spawning)
_MCOP_SPAWN_TOOL_NAMES = frozenset({
    "agent_spawn_task",
    "agent_spawn_parallel",
    "agent_check_tasks",
    "agent_read_task_result",
    "agent_read_packet",
})


def _filter_child_tools(tools):
    """Remove recursive MCOP spawn tools from a candidate child binding list."""
    return [t for t in tools if hasattr(t, "name") and t.name not in _MCOP_SPAWN_TOOL_NAMES]


def _get_child_tools(prompt: str | None = None, *, has_prior_tool_results: bool = False):
    """
    Return methodology-scoped child tools, falling back to the full executor
    inventory only when no prompt is available (ToolNode construction path).

    Model-facing binding stays small for local Ollama; ToolNode keeps the broad
    executor inventory so emitted calls can be executed.
    """
    if prompt:
        scoped, _plan = get_scoped_tools_for_prompt(prompt, has_prior_tool_results=has_prior_tool_results)
        return _filter_child_tools(scoped)
    return _filter_child_tools(get_all_executor_tools())

settings = get_settings()
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

MCOP_MAX_DEPTH = 1          # Children cannot spawn grandchildren
MCOP_MAX_CHILDREN = int(getattr(settings, "MCOP_MAX_CHILDREN", 5))
MCOP_CHILD_MAX_ITERS = int(getattr(settings, "MCOP_CHILD_MAX_ITERS", 5))
MCOP_PARALLEL_LIMIT = int(getattr(settings, "MCOP_PARALLEL_LIMIT", 3))
MCOP_RESULT_MAX_TOKENS = int(getattr(settings, "MCOP_RESULT_MAX_TOKENS", 2000))
MCOP_DIR = "_mcop"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ChildResult:
    """Result of a child conversation execution."""
    task_id: str
    status: str  # "completed" | "failed" | "partial" | "running"
    artifacts: List[str] = field(default_factory=list)
    summary: str = ""
    tokens_used: int = 0
    iterations_used: int = 0
    error: Optional[str] = None
    facts: List[dict] = field(default_factory=list)
    inferences: List[dict] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    raw_artifacts: List[str] = field(default_factory=list)
    next_recommended_step: str = ""
    packet_path: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    # Phase D2 bounded parent-facing completion contract.  No policy
    # object, no child messages, no graph state, no tool arguments and no
    # raw tool outputs cross back to the parent.
    child_run_id: str = ""
    toolsets_permitted: List[str] = field(default_factory=list)
    tools_executed: List[str] = field(default_factory=list)
    audit_event_ids: List[str] = field(default_factory=list)
    capability_preflight: dict = field(default_factory=dict)
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
    
    @classmethod
    def from_json(cls, data: str) -> "ChildResult":
        return cls(**json.loads(data))


@dataclass  
class OrchestrationState:
    """Master state for all child tasks under a parent."""
    parent_chat_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tasks: dict = field(default_factory=dict)  # task_id -> ChildResult
    total_tokens: int = 0
    
    def to_json(self) -> str:
        serializable = {
            "parent_chat_id": self.parent_chat_id,
            "created_at": self.created_at,
            "tasks": {k: asdict(v) for k, v in self.tasks.items()},
            "total_tokens": self.total_tokens,
        }
        return json.dumps(serializable, indent=2)
    
    @classmethod
    def load(cls, parent_chat_id: str) -> "OrchestrationState":
        """Load orchestration state from workspace, or create new."""
        ws = _get_mcop_dir(parent_chat_id)
        state_file = ws / "orchestration_state.json"
        if state_file.exists():
            data = json.loads(state_file.read_text())
            state = cls(
                parent_chat_id=data["parent_chat_id"],
                created_at=data["created_at"],
                total_tokens=data.get("total_tokens", 0),
            )
            for tid, tdata in data.get("tasks", {}).items():
                state.tasks[tid] = ChildResult(**tdata)
            return state
        return cls(parent_chat_id=parent_chat_id)
    
    def save(self):
        """Persist orchestration state to workspace."""
        ws = _get_mcop_dir(self.parent_chat_id)
        state_file = ws / "orchestration_state.json"
        state_file.write_text(self.to_json())


# ============================================================================
# WORKSPACE HELPERS
# ============================================================================

import os
_BASE_WORKDIR = Path(os.environ.get("AGENT_MODE_WORKDIR", "/tmp/home_agent"))


def _get_workspace(chat_id: str) -> Path:
    """Get workspace root for a chat."""
    ws = _BASE_WORKDIR / chat_id
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _get_mcop_dir(chat_id: str) -> Path:
    """Get or create the _mcop directory for orchestration state."""
    ws = _get_workspace(chat_id)
    mcop = ws / MCOP_DIR
    mcop.mkdir(parents=True, exist_ok=True)
    return mcop


def _get_task_dir(chat_id: str, task_id: str) -> Path:
    """Get or create a task-specific directory."""
    mcop = _get_mcop_dir(chat_id)
    task_dir = mcop / f"task_{task_id}"
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


# ============================================================================
# CHILD CONVERSATION LANGGRAPH
# ============================================================================

class ChildState(TypedDict):
    messages: Annotated[list, add_messages]
    chat_id: str       # parent's chat_id (for workspace access)
    task_id: str
    iterations: int
    max_iters: int
    tokens_used: int
    # Phase D2: immutable bounded parent policy upper bound for this run.
    # Ephemeral only -- it lives in MemorySaver for one child run and is
    # never written to the parent's encrypted Postgres checkpoint.
    child_policy: Any
    task_capability_plan: Any


def _tool_call_signature(call: Any) -> str:
    if isinstance(call, dict):
        name = call.get("name", "")
        args = call.get("args", {})
    else:
        name = getattr(call, "name", "")
        args = getattr(call, "args", {})
    try:
        args_text = json.dumps(args, sort_keys=True)
    except Exception:
        args_text = str(args)
    return f"{name}:{args_text}"


def _has_repeated_identical_tool_call(messages: list) -> bool:
    seen = set()
    for msg in messages:
        calls = getattr(msg, "tool_calls", None) or []
        for call in calls:
            signature = _tool_call_signature(call)
            if signature in seen:
                return True
            seen.add(signature)
    return False


def _last_human_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                return content
            return str(content or "")
    return ""


def _build_child_system_prompt(
    task_id: str,
    chat_id: str,
    max_iters: int,
    current_iter: int = 0,
    reserve_iters: int = 3,
    bound_tool_names: Sequence[str] = (),
) -> str:
    """Build child system prompt with three-layer iteration budget awareness.

    Layer 1 — Planning block  : iteration 0 only — how to allocate the budget.
    Layer 2 — Countdown line  : every iteration  — exact position in budget.
    Layer 3 — Wrap-up inject  : last reserve_iters — hard directive to write output.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    bound = {str(name) for name in bound_tool_names if str(name)}
    writer_tool = (
        "agent_run_python" if "agent_run_python" in bound
        else ("agent_run_shell" if "agent_run_shell" in bound else "")
    )
    if writer_tool:
        output_instruction = (
            f"Write findings to the task workspace using {writer_tool} (chat_id={chat_id})."
        )
        final_write_instruction = (
            f"If you have not saved the output, use {writer_tool} now (chat_id={chat_id})."
        )
    else:
        output_instruction = (
            "No file-writing tool is bound. Return the complete structured evidence packet inline; "
            "do not claim that an artifact was written."
        )
        final_write_instruction = output_instruction

    # Guard: reserve must leave at least 1 research iteration
    reserve = min(reserve_iters, max_iters - 1)
    remaining_after = max_iters - current_iter - 1   # remaining AFTER this iteration
    research_end = max_iters - reserve
    in_reserve = (max_iters - current_iter) <= reserve
    is_last = current_iter >= max_iters - 1

    # ── Layer 2: countdown (always shown) ───────────────────────────────────
    if is_last:
        phase_label = "OUTPUT \u2014 FINAL \ud83d\udd34"
    elif in_reserve:
        phase_label = "OUTPUT \u26a0\ufe0f"
    elif (max_iters - current_iter) == reserve + 1:
        phase_label = "RESEARCH \u2014 finish up"
    else:
        phase_label = "RESEARCH"

    countdown = (
        f"Iteration : {current_iter + 1} / {max_iters}  |  "
        f"Remaining after this: {remaining_after}  |  "
        f"Phase: {phase_label}"
    )

    # ── Layer 1: planning block (iteration 0 only) ──────────────────────────
    if current_iter == 0:
        planning_block = (
            f"\nITERATION BUDGET PLANNING:\n"
            f"You have {max_iters} total iterations. Allocate them as follows:\n\n"
            f"  RESEARCH PHASE  (iterations 1–{research_end}):\n"
            f"    Make only calls supported by the bound capability plan. Prioritise\n"
            f"    the highest-value evidence first.\n\n"
            f"  OUTPUT PHASE  (iterations {research_end + 1}–{max_iters}) ← RESERVED:\n"
            f"    {output_instruction}\n"
            f"    Include findings, classifications, evidence references, and known gaps.\n"
            f"    Final text response must be a concise structured packet for the parent.\n\n"
            f"You MUST produce the evidence packet before your budget runs out.\n"
            f"Do not wait for the WRAP-UP warning to start finalizing.\n"
        )
    else:
        planning_block = ""

    # ── Layer 3: wrap-up / rules section ────────────────────────────────────
    if is_last:
        rules_section = (
            f"\n🔴 LAST ITERATION ({current_iter + 1}/{max_iters}).\n"
            f"{final_write_instruction}\n"
            f"Return whatever verified evidence is available, even if partial.\n"
            f"Then return your structured summary. This is your last chance.\n"
        )
    elif in_reserve:
        rules_section = (
            f"\n⚠️  OUTPUT PHASE — {remaining_after} iteration(s) remaining after this one.\n"
            f"STOP all research and unrelated calls.\n"
            f"{output_instruction}\n\n"
            f"If research is incomplete, preserve the gaps explicitly.\n\n"
            f"Required before you finish:\n"
            f"  1. Produce the evidence packet using only bound tools\n"
            f"  2. Return: status + concise evidence-backed summary for the parent\n"
        )
    else:
        rules_section = (
            f"\nRULES:\n"
            f"- Complete the assigned task efficiently\n"
            f"- Use only tools in the declared, preflighted binding\n"
            f"- {output_instruction}\n"
            f"- Be concise — your packet will be returned to the parent agent\n"
            f"- Do NOT ask questions — preserve an explicit gap when blocked\n"
            f"- Do NOT explain reasoning at length — execute and cite evidence\n"
            f"- Do not claim an artifact exists unless a bound tool actually wrote it\n"
        )

    packet_contract = (
        "\nSTRUCTURED EVIDENCE PACKET CONTRACT:\n"
        "Your final response must be JSON for a ToolEvidencePacket with fields: "
        "packet_type='tool_evidence', task_id, worker_role, status, tool_families_used, "
        "tools_called, raw_artifacts, facts, inferences, gaps, errors, next_recommended_step. "
        "Facts must cite tool/file/command references. Inferences must state what facts they depend on. "
        "Do not spawn child conversations. Do not repeat identical tool calls; change scope/query or stop with a gap.\n"
    )

    return (
        f"You are a focused task executor. You have been spawned to complete ONE specific task.\n\n"
        f"Date      : {now}\n"
        f"Workspace : {chat_id}\n"
        f"Task ID   : {task_id}\n"
        f"Task dir  : {_get_task_dir(chat_id, task_id)}\n"
        f"{countdown}\n"
        f"{planning_block}"
        f"{rules_section}"
        f"{packet_contract}"
    )


async def _child_agent_node(state: ChildState, config=None):
    """The LLM node for child conversations."""
    chat_id = state["chat_id"]
    task_id = state["task_id"]
    iterations = state.get("iterations", 0)
    messages = state["messages"]
    max_iters = int(state.get("max_iters", MCOP_CHILD_MAX_ITERS))
    
    interceptor.thought(
        f"[Child {task_id}] Iteration {iterations+1}/{max_iters}", 
        "child_thinking"
    )
    
    model = get_model(role="tool_worker")
    set_model_config(model, {"temperature": 0.5})  # Lower temp for focused execution

    prompt_text = _last_human_text(messages)
    has_prior_tool_results = any(isinstance(m, ToolMessage) for m in messages)
    child_tools, tool_plan = get_scoped_tools_for_prompt(
        prompt_text,
        has_prior_tool_results=has_prior_tool_results,
    )
    # Phase D2: the per-turn model binding is intersected with the parent
    # policy upper bound before it ever reaches the model.  The execution
    # gate would block an unauthorized call anyway; keeping it out of the
    # schema means the model is never invited to attempt it.
    _child_policy = state.get("child_policy")
    _task_plan = state.get("task_capability_plan")
    if not isinstance(_task_plan, TaskCapabilityPlan):
        _task_plan = TaskCapabilityPlan.from_values(plan_required=False)
    required_names = tuple(_task_plan.required_tools)
    child_tools = child_safe_tools(child_tools, _child_policy)
    if required_names:
        child_tools = child_safe_tools(
            child_tools, _child_policy, task_required_names=required_names
        )
    # Explicit required families/tools are permitted to add only concrete tools
    # already inside the parent snapshot. This prevents methodology selection
    # from accidentally hiding a declared capability while preserving the
    # parent policy as an upper bound.
    if _task_plan.required_toolsets or required_names:
        broad_required = child_safe_tools(
            _get_child_tools(), _child_policy, task_required_names=required_names or None
        )
        existing = {getattr(tool, "name", "") for tool in child_tools}
        for tool in broad_required:
            name = str(getattr(tool, "name", "") or "")
            if name and name not in existing:
                child_tools.append(tool)
                existing.add(name)
    child_tools = _filter_child_tools(child_tools)
    system = SystemMessage(content=_build_child_system_prompt(
        task_id, chat_id, max_iters,
        current_iter=iterations,
        reserve_iters=settings.MCOP_CHILD_RESERVE_ITERS,
        bound_tool_names=[getattr(tool, "name", "") for tool in child_tools],
    ))
    llm_with_tools = model.bind_tools(child_tools)

    # Log consolidated message state (matches parent call_model format)
    msg_types = {}
    for msg in messages:
        t = type(msg).__name__
        msg_types[t] = msg_types.get(t, 0) + 1
    type_summary = " ".join(f"{k}={v}" for k, v in msg_types.items())
    audit = binding_audit(
        role="tool_worker",
        model_name=getattr(settings, "PLLM_TOOL_MODEL", "tool_worker"),
        tools=child_tools,
        plan=tool_plan,
        bound_model=llm_with_tools,
    )
    logger.info(
        "Child %s iter=%s/%s | %s msgs [%s] | tool_binding_audit=%s",
        task_id,
        iterations + 1,
        max_iters,
        len(messages),
        type_summary,
        json.dumps(audit, sort_keys=True, default=str),
    )

    policy = SystemMessage(content=build_compact_tool_execution_system_prompt(tool_plan))
    _record_child_binding(child_tools)
    response = await llm_with_tools.ainvoke([system, policy, *messages], config=config)
    turn_tokens = _response_token_count(response)

    if should_force_tool_retry(
        prompt=prompt_text,
        plan=tool_plan,
        tools=child_tools,
        messages=messages,
        response=response,
    ):
        ledger_path = append_tool_execution_failure(
            session_id=chat_id,
            prompt=prompt_text,
            plan=tool_plan,
            response_text=str(getattr(response, "content", "")),
            reason="child_zero_tool_calls_initial",
            audit=audit,
        )
        retry_tools = rank_tools_for_retry(tool_plan.methodology, child_tools, limit=3)
        retry_tools = child_safe_tools(retry_tools, _child_policy)
        if required_names:
            retry_tools = child_safe_tools(
                retry_tools, _child_policy, task_required_names=required_names
            )
        retry_llm = model.bind_tools(retry_tools)
        retry_policy = SystemMessage(content=build_compact_tool_execution_system_prompt(tool_plan, retry=True))
        _record_child_binding(retry_tools)
        retry_response = await retry_llm.ainvoke([system, retry_policy, *messages], config=config)
        turn_tokens += _response_token_count(retry_response)
        if not getattr(retry_response, "tool_calls", None):
            ledger_path = append_tool_execution_failure(
                session_id=chat_id,
                prompt=prompt_text,
                plan=tool_plan,
                response_text=str(getattr(retry_response, "content", "")),
                reason="child_zero_tool_calls_retry",
                audit=audit,
            )
            retry_response = AIMessage(content=build_tool_execution_failed_message(tool_plan, ledger_path))
        response = retry_response

    return {
        "messages": [response],
        "iterations": iterations + 1,
        "tokens_used": int(state.get("tokens_used", 0) or 0) + turn_tokens,
    }


def _response_token_count(response: Any) -> int:
    """Return a best-effort token total from one model response.

    LangChain providers expose usage in slightly different places. Prefer the
    normalized ``usage_metadata`` contract, then fall back to common provider
    response-metadata shapes. Missing usage remains zero rather than fabricating
    an estimate.
    """
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if total is not None:
            try:
                return max(0, int(total))
            except Exception:
                pass
        try:
            return max(0, int(usage.get("input_tokens") or 0)) + max(
                0, int(usage.get("output_tokens") or 0)
            )
        except Exception:
            pass

    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        candidates = [
            metadata.get("usage_metadata"),
            metadata.get("token_usage"),
            metadata.get("usage"),
        ]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            for key in ("total_tokens", "totalTokenCount", "total_token_count"):
                if candidate.get(key) is not None:
                    try:
                        return max(0, int(candidate[key]))
                    except Exception:
                        pass
            input_value = next(
                (
                    candidate.get(key)
                    for key in ("input_tokens", "prompt_tokens", "inputTokenCount")
                    if candidate.get(key) is not None
                ),
                0,
            )
            output_value = next(
                (
                    candidate.get(key)
                    for key in ("output_tokens", "completion_tokens", "outputTokenCount")
                    if candidate.get(key) is not None
                ),
                0,
            )
            try:
                return max(0, int(input_value or 0)) + max(0, int(output_value or 0))
            except Exception:
                continue
    return 0


def _bounded_child_summary(text: str) -> str:
    """Create a parent-facing preview only after authoritative packet parsing."""
    value = str(text or "")
    max_chars = MCOP_RESULT_MAX_TOKENS * 4  # bounded display only; not evidence storage
    if len(value) > max_chars:
        return value[:max_chars] + "\n... [SUMMARY TRUNCATED; FULL PACKET/RAW RESPONSE IS DURABLE]"
    return value


def _record_child_binding(tools) -> None:
    """Record the exact child model-facing binding for the execution gate.

    The child executor inventory is deliberately broader than the per-turn
    model binding, so the gate must be told which exact tools this specific
    model invocation was actually given.
    """
    try:
        record_model_facing_binding(
            bound_tool_names=[
                _name
                for _name in (getattr(_t, "name", "") for _t in (tools or []))
                if _name
            ]
        )
    except Exception as exc:  # noqa: BLE001 - correlation is never fatal
        logger.warning(
            "mcop child binding record failed: %s", type(exc).__name__
        )


def _executed_tool_names(messages, limit: int = 40) -> list:
    """Bounded list of tool names the child actually executed."""
    names = []
    for message in messages or []:
        if not isinstance(message, ToolMessage):
            continue
        name = str(getattr(message, "name", "") or "")
        if name and name not in names:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def _child_audit_event_ids(child_run_id: str, limit: int = 40) -> list:
    """Bounded audit event IDs for this child run.

    Correlation only -- no audit record bodies cross back to the parent.
    Never raises: an inspection failure must not fail a child run.
    """
    if not child_run_id:
        return []
    try:
        from app.agent.tool_execution_audit import get_audit_sink

        ids = []
        for event in get_audit_sink().iter_events(max_records=2000):
            if event.get("child_run_id") != child_run_id:
                continue
            event_id = str(event.get("event_id") or "")
            if event_id and event_id not in ids:
                ids.append(event_id)
            if len(ids) >= limit:
                break
        return ids
    except Exception as exc:  # noqa: BLE001 - correlation is never fatal
        logger.warning(
            "mcop child audit correlation failed: %s", type(exc).__name__
        )
        return []


def _child_route(state: ChildState) -> str:
    """Route child: continue with tools or end."""
    iters = state.get("iterations", 0)
    max_iters = int(state.get("max_iters", MCOP_CHILD_MAX_ITERS))
    if _has_repeated_identical_tool_call(state.get("messages", [])):
        logger.warning("Child %s stopped due to repeated identical tool call", state.get("task_id", "?"))
        return END
    last = state["messages"][-1]
    has_tools = hasattr(last, "tool_calls") and bool(last.tool_calls)
    
    if iters >= max_iters:
        return END
    elif has_tools:
        return "tools"
    else:
        return END


def _build_child_graph(policy=None, task_capability_plan: TaskCapabilityPlan | None = None):
    """Build a fresh LangGraph for a child conversation.

    ``policy`` is the immutable parent snapshot.  When supplied, even the
    broad executor inventory is reduced to the parent-permitted surface.
    """
    child_tools = _get_child_tools()
    required_names = tuple((task_capability_plan or TaskCapabilityPlan.from_values(plan_required=False)).required_tools)
    
    workflow = StateGraph(ChildState)
    workflow.add_node("agent", _child_agent_node)
    # Phase D0: the child no longer executes model-emitted calls through a
    # raw ToolNode.  It uses the same audited, gated executor as the parent
    # graph, so exact-binding, authorization, persistence, mutation and
    # heavy-argument enforcement all apply inside child runs too.
    bounded_child_tools = child_safe_tools(child_tools, policy)
    if required_names:
        bounded_child_tools = child_safe_tools(
            bounded_child_tools, policy, task_required_names=required_names
        )
    workflow.add_node(
        "tools",
        make_audited_tool_node(
            bounded_child_tools,
            node_name="mcop_child_tools",
        ),
    )
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", _child_route, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    
    # Use in-memory checkpointer for children (ephemeral)
    from langgraph.checkpoint.memory import MemorySaver
    return workflow.compile(checkpointer=MemorySaver())


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

async def run_child_conversation(
    parent_chat_id: str,
    task_id: str,
    prompt: str,
    context_files: Optional[List[str]] = None,
    max_iters: int = None,
    task_capability_plan: TaskCapabilityPlan | None = None,
) -> ChildResult:
    """
    Execute a sub-task in a completely fresh conversation context.
    
    This is the core primitive of MCOP. It:
    1. Creates a fresh LangGraph thread (clean context window)
    2. Injects ONLY the task prompt + optional file contents as context
    3. Runs the child agent to completion (or iteration limit)
    4. Extracts the final response as a summary
    5. Saves the result to _mcop/task_{id}/result.json
    6. Cleans up the child's LangGraph state
    
    Args:
        parent_chat_id: The parent's workspace (shared filesystem)
        task_id: Unique identifier for this task
        prompt: The child's instructions (self-contained)
        context_files: Optional file paths to inject as context
        max_iters: Override child iteration limit
    
    Returns:
        ChildResult with status, artifacts, and summary
    """
    effective_max_iters = max_iters or MCOP_CHILD_MAX_ITERS
    task_capability_plan = task_capability_plan or TaskCapabilityPlan.from_values()
    
    task_dir = _get_task_dir(parent_chat_id, task_id)
    child_thread_id = f"child_{task_id}_{uuid.uuid4().hex[:8]}"
    
    # Record start
    result = ChildResult(
        task_id=task_id,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    
    # Save the prompt for debugging/auditing
    (task_dir / "prompt.txt").write_text(prompt)
    
    interceptor.thought(f"Spawning child conversation: {task_id}", "child_spawn")
    
    try:
        # Build the child's input message
        input_parts = []
        
        # Inject context from files if provided
        if context_files:
            for fpath in context_files:
                full_path = _get_workspace(parent_chat_id) / fpath
                if full_path.exists():
                    content = full_path.read_text()
                    # Truncate very large files
                    if len(content) > 50000:
                        content = content[:50000] + "\n... [TRUNCATED]"
                    input_parts.append(f'<context_file path="{fpath}">\n{content}\n</context_file>')
                else:
                    input_parts.append(f'<context_file path="{fpath}">[FILE NOT FOUND]</context_file>')
        
        # Add the task prompt
        input_parts.append(f"<task>\n{prompt}\n</task>")
        
        full_input = "\n\n".join(input_parts)
        
        # Build and run the child graph
        # ── Phase D2: bounded parent-to-child policy inheritance ──────────
        # The snapshot is captured from the parent's effective state for the
        # current turn.  It is an upper bound: the child narrows it for the
        # task and can never broaden it.
        child_policy = build_child_policy_snapshot()
        _problems = validate_child_policy_snapshot(child_policy)
        if _problems:
            logger.warning(
                "mcop_child_policy malformed (%s); failing closed",
                ",".join(_problems[:4]),
            )
            child_policy = fail_closed_policy(child_policy.child_run_id)
        _live_generation = current_registry_generation()
        child_policy, _generation_match = reconcile_snapshot_with_registry(
            child_policy, current_generation=_live_generation
        )
        child_policy = narrow_policy_for_task(
            child_policy,
            task_required_toolsets=task_capability_plan.required_toolsets,
            task_required_tools=task_capability_plan.required_tools,
        )
        preflight_tools = _filter_child_tools(
            child_safe_tools(
                _get_child_tools(),
                child_policy,
                task_required_names=task_capability_plan.required_tools or None,
            )
        )
        capability_decision = evaluate_child_capability_plan(
            plan=task_capability_plan,
            parent_policy=child_policy,
            candidate_tool_names=[getattr(tool, "name", "") for tool in preflight_tools],
        )
        result.capability_preflight = capability_decision.to_dict()
        logger.info(
            "mcop_child_policy %s",
            json.dumps(
                {
                    "task_id": task_id,
                    "generation_match": _generation_match,
                    **child_policy.as_safe_dict(),
                },
                sort_keys=True,
                default=str,
            ),
        )
        if not capability_decision.allowed:
            result.status = "blocked"
            result.gaps = [
                "Missing child capabilities: " + ", ".join(capability_decision.missing_capabilities)
                if capability_decision.missing_capabilities else "Child capability plan was not satisfiable."
            ]
            result.errors = [BLOCKED_CHILD_MISSING_CAPABILITY]
            result.next_recommended_step = "Bind the exact missing capability in the parent or revise the declared child plan."
            result.child_run_id = child_policy.child_run_id
            result.toolsets_permitted = list(child_policy.eligible_toolsets)[:24]
            result.finished_at = datetime.now(timezone.utc).isoformat()
            blocked_packet = normalize_worker_packet_from_summary(
                task_id=task_id,
                summary="",
                status="blocked",
                artifacts=[],
                error=BLOCKED_CHILD_MISSING_CAPABILITY,
            )
            blocked_packet.gaps = list(result.gaps)
            blocked_packet.next_recommended_step = result.next_recommended_step
            packet_file = task_dir / "tool_evidence_packet.json"
            packet_file.write_text(json.dumps(asdict(blocked_packet), indent=2, sort_keys=True) + "\n")
            result.packet_path = str(packet_file.relative_to(_get_workspace(parent_chat_id)))
            (task_dir / "result.json").write_text(result.to_json())
            orchestration = OrchestrationState.load(parent_chat_id)
            orchestration.tasks[task_id] = result
            orchestration.save()
            return result
        graph = _build_child_graph(child_policy, task_capability_plan)
        
        config = {
            "configurable": {
                "thread_id": child_thread_id,
            },
            "recursion_limit": max(10, effective_max_iters * 4),
        }
        
        # Execute the child conversation under the captured snapshot.
        with child_audit_scope(
            child_policy,
            generation_match=_generation_match,
            current_generation=_live_generation,
        ):
            final_state = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=full_input)],
                    "chat_id": parent_chat_id,
                    "task_id": task_id,
                    "iterations": 0,
                    "max_iters": effective_max_iters,
                    "tokens_used": 0,
                    "child_policy": child_policy,
                    "task_capability_plan": task_capability_plan,
                },
                config=config,
            )
        
        # Extract results
        messages = final_state.get("messages", [])
        iterations_used = final_state.get("iterations", 0)
        tokens_used = int(final_state.get("tokens_used", 0) or 0)

        # Extract the full final response. Structured evidence is parsed from this
        # authoritative text BEFORE any parent-facing preview is truncated.
        full_summary = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if isinstance(msg.content, str):
                    full_summary = msg.content
                elif isinstance(msg.content, list):
                    full_summary = " ".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in msg.content
                    )
                break

        parsed_packet = try_parse_tool_evidence_packet(full_summary, task_id)
        summary = _bounded_child_summary(full_summary)

        # Scan only the child task directory; do not mtime-scan the whole parent workspace.
        artifacts = []
        for f in task_dir.rglob("*"):
            if f.is_file() and f.name not in {"result.json", "tool_evidence_packet.json"}:
                artifacts.append(str(f.relative_to(_get_workspace(parent_chat_id))))

        if parsed_packet is None:
            # Preserve malformed/prose output durably, but never promote it to a
            # successful ToolEvidencePacket. The parent can inspect this artifact
            # while the task remains partial and therefore cannot masquerade as
            # verified completion.
            if full_summary:
                raw_final = task_dir / "unparsed_final_response.txt"
                raw_final.write_text(full_summary)
                raw_rel = str(raw_final.relative_to(_get_workspace(parent_chat_id)))
                if raw_rel not in artifacts:
                    artifacts.append(raw_rel)
            parsed_packet = normalize_worker_packet_from_summary(
                task_id=task_id, summary=summary, status="partial", artifacts=artifacts
            )
            parsed_packet.gaps.append(
                "Child final response was not a valid ToolEvidencePacket; raw response was preserved for review."
            )
        packet_data = asdict(parsed_packet)
        packet_data["task_id"] = task_id
        if not packet_data.get("raw_artifacts"):
            packet_data["raw_artifacts"] = artifacts
        packet_file = task_dir / "tool_evidence_packet.json"
        packet_file.write_text(json.dumps(packet_data, indent=2, sort_keys=True) + "\n")

        # Finalize result without amplifying a blocked/partial packet into completed.
        packet_status = str(packet_data.get("status") or "partial").lower()
        result.status = {
            "complete": "completed",
            "completed": "completed",
            "success": "completed",
            "blocked": "blocked",
            "failed": "failed",
            "error": "failed",
            "partial": "partial",
        }.get(packet_status, "partial")
        result.summary = summary
        # Phase D2 bounded completion contract.
        result.child_run_id = child_policy.child_run_id
        result.toolsets_permitted = list(child_policy.eligible_toolsets)[:24]
        result.tools_executed = _executed_tool_names(messages)
        result.audit_event_ids = _child_audit_event_ids(child_policy.child_run_id)
        result.artifacts = artifacts
        result.raw_artifacts = packet_data.get("raw_artifacts", artifacts)
        result.facts = packet_data.get("facts", [])
        result.inferences = packet_data.get("inferences", [])
        result.gaps = packet_data.get("gaps", [])
        result.errors = packet_data.get("errors", [])
        result.next_recommended_step = packet_data.get("next_recommended_step", "")
        result.packet_path = str(packet_file.relative_to(_get_workspace(parent_chat_id)))
        result.iterations_used = iterations_used
        result.tokens_used = tokens_used
        result.finished_at = datetime.now(timezone.utc).isoformat()

        interceptor.thought(
            f"Child {task_id} finished status={result.status}: {len(artifacts)} artifacts, "
            f"{iterations_used} iters, {tokens_used} tokens",
            "child_complete"
        )
        
    except Exception as e:
        result.status = "failed"
        result.error = str(e)
        result.errors = [str(e)]
        failed_packet = normalize_worker_packet_from_summary(task_id=task_id, summary="", status="failed", artifacts=[], error=str(e))
        packet_file = task_dir / "tool_evidence_packet.json"
        packet_file.write_text(json.dumps(asdict(failed_packet), indent=2, sort_keys=True) + "\n")
        result.packet_path = str(packet_file.relative_to(_get_workspace(parent_chat_id)))
        result.finished_at = datetime.now(timezone.utc).isoformat()
        logger.error(f"Child conversation {task_id} failed: {e}", exc_info=True)
        interceptor.thought(f"Child {task_id} FAILED: {e}", "child_error")
    
    # Save result
    (task_dir / "result.json").write_text(result.to_json())
    
    # Update orchestration state
    state = OrchestrationState.load(parent_chat_id)
    state.tasks[task_id] = result
    state.total_tokens += result.tokens_used
    state.save()
    
    return result


# ============================================================================
# PARALLEL EXECUTION HELPER
# ============================================================================

_semaphore = asyncio.Semaphore(MCOP_PARALLEL_LIMIT)


async def run_parallel_tasks(
    parent_chat_id: str,
    tasks: List[dict],
) -> List[ChildResult]:
    """
    Execute multiple child conversations in parallel (with concurrency limit).
    
    Args:
        parent_chat_id: Parent workspace
        tasks: List of task definitions, each containing:
            - task_id: str
            - prompt: str
            - context_files: Optional[List[str]]
            - max_iters: Optional[int]
    
    Returns:
        List of ChildResults (in same order as input)
    """
    if len(tasks) > MCOP_MAX_CHILDREN:
        raise ValueError(
            f"Cannot spawn {len(tasks)} children (max: {MCOP_MAX_CHILDREN})"
        )
    
    async def _run_with_semaphore(task_def):
        async with _semaphore:
            return await run_child_conversation(
                parent_chat_id=parent_chat_id,
                task_id=task_def["task_id"],
                prompt=task_def["prompt"],
                context_files=task_def.get("context_files"),
                max_iters=task_def.get("max_iters"),
                task_capability_plan=TaskCapabilityPlan.from_values(
                    required_toolsets=task_def.get("required_toolsets", ()),
                    required_tools=task_def.get("required_tools", ()),
                    required_capabilities=task_def.get("required_capabilities", ()),
                    expected_artifact_types=task_def.get("expected_artifact_types", ()),
                    write_required=bool(task_def.get("write_required", False)),
                    network_required=bool(task_def.get("network_required", False)),
                    repository_access_required=bool(task_def.get("repository_access_required", False)),
                    plan_required=bool(task_def.get("plan_required", True)),
                ),
            )
    
    interceptor.thought(
        f"Running {len(tasks)} child tasks in parallel (limit: {MCOP_PARALLEL_LIMIT})",
        "parallel_spawn"
    )
    
    results = await asyncio.gather(
        *[_run_with_semaphore(t) for t in tasks],
        return_exceptions=True,
    )
    
    # Convert exceptions to failed results
    final_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            final_results.append(ChildResult(
                task_id=tasks[i]["task_id"],
                status="failed",
                error=str(r),
            ))
        else:
            final_results.append(r)
    
    return final_results
