"""
MCOP Orchestrator Tools - Agent-facing interface for multi-conversation orchestration.
=====================================================================================

These tools allow the parent agent to spawn isolated child conversations,
check their status, and collect results.

Author: Jacob Montgomery
Created: 2026-06-23
Version: 0.1.0 (MVP)
"""

import asyncio
import hashlib
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Optional, List

from langchain.tools import tool

from app.agent_mode.thought_interceptor import interceptor
from app.agent.tool_execution_audit import current_audit_request_state
from app.agent_mode.task_capability_plan import TaskCapabilityPlan
from app.agent_mode.child_conversation import (
    run_child_conversation,
    run_parallel_tasks,
    OrchestrationState,
    ChildResult,
    _get_mcop_dir,
    _get_task_dir,
    MCOP_MAX_CHILDREN,
)

logger = logging.getLogger(__name__)

# Parent-facing MCOP responses are intentionally bounded. Full child evidence
# remains durable under _mcop/ and can be read in explicit chunks.
_TERMINAL_CHILD_STATUSES = frozenset({
    "completed", "failed", "partial", "cancelled", "blocked",
})
_TASK_STATUS_FILTERS = frozenset({"all", "active", "terminal"})
_DEFAULT_TASK_PAGE_SIZE = 10
_MAX_TASK_PAGE_SIZE = 50
_DEFAULT_TEXT_CHUNK_CHARS = 6000
_MAX_TEXT_CHUNK_CHARS = 12000
_PARENT_RESPONSE_MAX_CHARS = 12000
_TASK_SUMMARY_PREVIEW_CHARS = 240
_PARALLEL_SUMMARY_PREVIEW_CHARS = 500
_ERROR_PREVIEW_CHARS = 600
_ARTIFACT_PREVIEW_LIMIT = 20
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _normalized_child_status(result: ChildResult) -> str:
    return str(getattr(result, "status", "") or "").strip().lower()


def _is_terminal_child(result: ChildResult) -> bool:
    return _normalized_child_status(result) in _TERMINAL_CHILD_STATUSES


def _task_capacity(state: OrchestrationState) -> dict:
    active_ids = [
        task_id
        for task_id, result in state.tasks.items()
        if not _is_terminal_child(result)
    ]
    terminal_count = len(state.tasks) - len(active_ids)
    return {
        "max_active_children": MCOP_MAX_CHILDREN,
        "active_task_count": len(active_ids),
        "terminal_task_count": terminal_count,
        "available_slots": max(0, MCOP_MAX_CHILDREN - len(active_ids)),
        "active_task_ids": active_ids,
    }


def _bounded_text(value: object, *, offset: int = 0, max_chars: int = _DEFAULT_TEXT_CHUNK_CHARS) -> dict:
    text = str(value or "")
    try:
        offset = max(0, int(offset or 0))
    except Exception:
        offset = 0
    try:
        max_chars = int(max_chars or _DEFAULT_TEXT_CHUNK_CHARS)
    except Exception:
        max_chars = _DEFAULT_TEXT_CHUNK_CHARS
    max_chars = max(1, min(max_chars, _MAX_TEXT_CHUNK_CHARS))
    chunk = text[offset:offset + max_chars]
    next_offset = offset + len(chunk)
    truncated = next_offset < len(text)
    return {
        "chunk": chunk,
        "offset": offset,
        "returned_chars": len(chunk),
        "total_chars": len(text),
        "truncated": truncated,
        "next_offset": next_offset if truncated else None,
    }


def _compact_child_result(result: ChildResult, *, preview_chars: int) -> dict:
    summary = _bounded_text(result.summary, offset=0, max_chars=preview_chars)
    error = _bounded_text(result.error, offset=0, max_chars=_ERROR_PREVIEW_CHARS)
    artifacts = list(result.artifacts or [])
    raw_artifacts = list(result.raw_artifacts or [])
    return {
        "task_id": result.task_id,
        "stable_task_id": result.task_id,
        "status": result.status,
        "iterations_used": result.iterations_used,
        "tokens_used": result.tokens_used,
        "artifacts_created": artifacts[:_ARTIFACT_PREVIEW_LIMIT],
        "artifacts_count": len(artifacts),
        "artifacts_truncated": len(artifacts) > _ARTIFACT_PREVIEW_LIMIT,
        "raw_artifacts_count": len(raw_artifacts),
        "facts_count": len(result.facts or []),
        "inferences_count": len(result.inferences or []),
        "gaps_count": len(result.gaps or []),
        "errors_count": len(result.errors or []),
        "next_recommended_step": _bounded_text(
            result.next_recommended_step, offset=0, max_chars=_TASK_SUMMARY_PREVIEW_CHARS
        )["chunk"],
        "packet_path": result.packet_path,
        "summary_preview": summary["chunk"],
        "summary_chars_total": summary["total_chars"],
        "summary_truncated": summary["truncated"],
        "error_preview": error["chunk"] or None,
        "error_chars_total": error["total_chars"],
        "detail_reader": "agent_read_task_result",
        "packet_reader": "agent_read_packet",
    }


def _full_child_result_payload(result: ChildResult) -> dict:
    payload = {
        "task_id": result.task_id,
        "stable_task_id": result.task_id,
        "status": result.status,
        "iterations_used": result.iterations_used,
        "tokens_used": result.tokens_used,
        "artifacts_created": result.artifacts,
        "raw_artifacts": result.raw_artifacts,
        "facts": result.facts,
        "inferences": result.inferences,
        "gaps": result.gaps,
        "errors": result.errors,
        "next_recommended_step": result.next_recommended_step,
        "packet_path": result.packet_path,
        "summary": result.summary,
    }
    if result.error:
        payload["error"] = result.error
    return payload


def _bounded_child_response(result: ChildResult, *, preview_chars: int) -> dict:
    """Preserve the legacy full payload when small; compact only when needed."""
    full = _full_child_result_payload(result)
    serialized_chars = len(json.dumps(full, default=str))
    if serialized_chars <= _PARENT_RESPONSE_MAX_CHARS:
        full["response_truncated"] = False
        full["response_chars_total"] = serialized_chars
        return full
    compact = _compact_child_result(result, preview_chars=preview_chars)
    compact.update({
        "response_truncated": True,
        "response_chars_total": serialized_chars,
        "required_action": (
            "Use agent_read_task_result for bounded summary/detail metadata or "
            "agent_read_packet for durable evidence chunks."
        ),
    })
    return compact


def _normalize_page(offset: int, limit: int) -> tuple[int, int]:
    try:
        offset = max(0, int(offset or 0))
    except Exception:
        offset = 0
    try:
        limit = int(limit or _DEFAULT_TASK_PAGE_SIZE)
    except Exception:
        limit = _DEFAULT_TASK_PAGE_SIZE
    return offset, max(1, min(limit, _MAX_TASK_PAGE_SIZE))


def _validate_task_id(task_id: str) -> str:
    value = str(task_id or "").strip()
    if not _TASK_ID_RE.fullmatch(value) or ".." in value:
        raise ValueError("Invalid task_id; use only letters, numbers, '.', '_' or '-' (max 128 chars).")
    return value


def _safe_task_file(chat_id: str, task_id: str, filename: str) -> Path:
    safe_task_id = _validate_task_id(task_id)
    mcop_root = _get_mcop_dir(chat_id).resolve()
    candidate = (mcop_root / f"task_{safe_task_id}" / filename).resolve()
    if mcop_root not in candidate.parents:
        raise ValueError("Resolved task path escaped the MCOP workspace.")
    return candidate


def _safe_packet_path(chat_id: str, *, task_id: str = "", packet_path: str = "") -> Path:
    mcop_root = _get_mcop_dir(chat_id).resolve()
    if packet_path:
        workspace_root = mcop_root.parent.resolve()
        candidate = (workspace_root / str(packet_path)).resolve()
        if mcop_root not in candidate.parents:
            raise ValueError("packet_path must resolve inside the current _mcop workspace.")
        return candidate
    if task_id:
        return _safe_task_file(chat_id, task_id, "tool_evidence_packet.json")
    raise ValueError("Provide either task_id or packet_path.")


def _parent_only_block_payload(tool_name: str) -> str | None:
    state = current_audit_request_state()
    constraints = getattr(state, "execution_constraints", {}) if state is not None else {}
    if not bool((constraints or {}).get("mcop_children_forbidden", False)):
        return None
    return json.dumps({
        "ok": False,
        "schema": "diship_gated_tool_result.v1",
        "result_code": "BLOCKED_PARENT_ONLY_TURN",
        "tool_name": tool_name,
        "write_performed": False,
        "mcop_children_spawned": 0,
        "required_action": "Continue in the parent thread without spawning children.",
    }, sort_keys=True)


@tool("agent_spawn_task")
async def agent_spawn_task(
    chat_id: str,
    task_prompt: str,
    task_id: str = "",
    context_files: str = "[]",
    max_iters: int = 5,
    required_toolsets: str = "[]",
    required_tools: str = "[]",
    required_capabilities: str = "[]",
    expected_artifact_types: str = "[]",
    write_required: bool = False,
    network_required: bool = False,
    repository_access_required: bool = False,
) -> str:
    """Spawn an isolated child conversation to handle a sub-task.
    
    The child gets a FRESH context window (no history bloat) and only the
    declared capabilities that also survive the parent policy and live registry
    intersection. It shares the workspace filesystem only when a permitted
    writing tool is actually bound.
    
    USE THIS WHEN:
    - The task has multiple independent sub-steps
    - Each step would generate significant tool output (logs, file contents)
    - Sub-steps don't require your intermediate reasoning to proceed
    
    DO NOT USE WHEN:
    - The task is simple enough to do in one pass
    - You need to interact with the user between steps
    - Steps are tightly coupled (B needs A's reasoning, not just A's output)
    
    Parameters:
      - chat_id: Your workspace identifier (MUST be your current chat_id)
      - task_prompt: Clear, self-contained instructions for the child agent.
                     Include ALL context the child needs — it has no memory of 
                     your conversation. Be specific about expected outputs.
      - task_id: Optional unique ID for the task (auto-generated if empty)
      - context_files: JSON array of file paths the child should read first.
                       Paths are relative to workspace root.
                       Example: '["repo/README.md", "config.json"]'
      - max_iters: Maximum iterations for the child (default: 5)
      - required_toolsets: JSON array/CSV of exact families required by the task
      - required_tools: JSON array/CSV of exact qualified or raw tool names
      - required_capabilities: JSON array/CSV of semantic capabilities
      - expected_artifact_types: JSON array/CSV; implies write_required
      - write_required/network_required/repository_access_required: hard gates

    A missing or unsatisfied capability plan returns
    BLOCKED_CHILD_MISSING_CAPABILITY without invoking the child model.
    
    Returns:
      JSON summary of the child's work including status, artifacts, and summary.
    """
    parent_only = _parent_only_block_payload("agent_spawn_task")
    if parent_only is not None:
        return parent_only

    # Generate task_id if not provided
    if not task_id:
        task_id = f"t{uuid.uuid4().hex[:6]}"
    
    # Parse context files
    try:
        files = json.loads(context_files) if context_files else []
    except json.JSONDecodeError:
        files = []
    
    # Check ACTIVE child capacity. Terminal history is durable evidence and does
    # not consume execution slots. Unknown/nonterminal states fail closed as active.
    state = OrchestrationState.load(chat_id)
    capacity = _task_capacity(state)
    if capacity["available_slots"] <= 0:
        return json.dumps({
            "ok": False,
            "schema": "mcop_child_capacity.v2",
            "result_code": "MAX_ACTIVE_CHILDREN_REACHED",
            "error": f"Maximum active child tasks reached ({MCOP_MAX_CHILDREN}).",
            **capacity,
            "total_task_history": len(state.tasks),
            "required_action": "Wait for or inspect an active child; completed/failed history is retained but does not block new spawns.",
        }, sort_keys=True)
    
    interceptor.thought(f"Spawning child task: {task_id}", "orchestrator")
    
    capability_plan = TaskCapabilityPlan.from_values(
        required_toolsets=required_toolsets,
        required_tools=required_tools,
        required_capabilities=required_capabilities,
        expected_artifact_types=expected_artifact_types,
        write_required=write_required,
        network_required=network_required,
        repository_access_required=repository_access_required,
        plan_required=True,
    )

    # Run the child conversation (blocking — waits for completion)
    result = await run_child_conversation(
        parent_chat_id=chat_id,
        task_id=task_id,
        prompt=task_prompt,
        context_files=files,
        max_iters=max_iters,
        task_capability_plan=capability_plan,
    )
    
    # Preserve legacy full result shape when it fits the parent response budget.
    # Large child results become an explicit compact index with durable readers.
    output = _bounded_child_response(result, preview_chars=_PARALLEL_SUMMARY_PREVIEW_CHARS)
    output.update({
        "ok": _normalized_child_status(result) == "completed",
        "schema": "mcop_child_spawn_result.v2",
        "result_code": "CHILD_COMPLETED" if _normalized_child_status(result) == "completed" else "CHILD_NOT_COMPLETED",
        "result": {
            "task_id": result.task_id,
            "status": result.status,
            "packet_path": result.packet_path,
        },
    })
    return json.dumps(output, indent=2, default=str)


@tool("agent_spawn_parallel")
async def agent_spawn_parallel(
    chat_id: str,
    tasks_json: str,
) -> str:
    """Spawn multiple child conversations to run in parallel.
    
    Each child gets a FRESH context window and runs independently.
    All children share the workspace filesystem.
    
    Parameters:
      - chat_id: Your workspace identifier
      - tasks_json: JSON array of task objects. Each object must have:
            - "task_id": unique ID (or omit for auto-generated)
            - "prompt": self-contained instructions
            - "context_files": optional array of file paths to inject
            - "max_iters": optional iteration budget (default: 5)
            - "required_toolsets", "required_tools", "required_capabilities"
            - "expected_artifact_types", "write_required", "network_required"
            - "repository_access_required" (capability plan is mandatory)
        
        Example:
        '[
            {"prompt": "Clone the repo and list all Python files", "context_files": []},
            {"prompt": "Analyze requirements.txt and check for vulnerabilities"}
        ]'
    
    Returns:
      JSON array of results from all child tasks.
    """
    parent_only = _parent_only_block_payload("agent_spawn_parallel")
    if parent_only is not None:
        return parent_only

    try:
        tasks = json.loads(tasks_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})
    
    if not isinstance(tasks, list):
        return json.dumps({"error": "tasks_json must be a JSON array"})
    
    if len(tasks) > MCOP_MAX_CHILDREN:
        return json.dumps({
            "ok": False,
            "schema": "mcop_child_capacity.v2",
            "result_code": "PARALLEL_BATCH_TOO_LARGE",
            "error": f"Too many tasks ({len(tasks)}). Maximum active children: {MCOP_MAX_CHILDREN}",
        })

    state = OrchestrationState.load(chat_id)
    capacity = _task_capacity(state)
    if len(tasks) > capacity["available_slots"]:
        return json.dumps({
            "ok": False,
            "schema": "mcop_child_capacity.v2",
            "result_code": "INSUFFICIENT_ACTIVE_CHILD_SLOTS",
            "requested_children": len(tasks),
            **capacity,
            "total_task_history": len(state.tasks),
        }, sort_keys=True)
    
    # Ensure each task has a task_id
    for i, task in enumerate(tasks):
        if "task_id" not in task or not task["task_id"]:
            task["task_id"] = f"t{uuid.uuid4().hex[:6]}"
        if "prompt" not in task:
            return json.dumps({"error": f"Task {i} missing 'prompt' field"})
    
    interceptor.thought(
        f"Spawning {len(tasks)} parallel children", "orchestrator"
    )
    
    # Execute all tasks in parallel
    results = await run_parallel_tasks(
        parent_chat_id=chat_id,
        tasks=tasks,
    )
    
    # Parallel aggregation is always compact so N child payloads cannot multiply
    # into an unbounded parent ToolMessage. Detail remains durable per task.
    entries = [
        _compact_child_result(r, preview_chars=_PARALLEL_SUMMARY_PREVIEW_CHARS)
        for r in results
    ]
    return json.dumps({
        "ok": all(_normalized_child_status(r) == "completed" for r in results),
        "schema": "mcop_parallel_result.v2",
        "result_code": "PARALLEL_CHILDREN_COMPLETE" if all(_normalized_child_status(r) == "completed" for r in results) else "PARALLEL_CHILDREN_MIXED_STATUS",
        "result_count": len(entries),
        "results": entries,
        "response_truncated": any(entry["summary_truncated"] for entry in entries),
    }, indent=2, default=str)


@tool("agent_check_tasks")
def agent_check_tasks(
    chat_id: str,
    offset: int = 0,
    limit: int = _DEFAULT_TASK_PAGE_SIZE,
    status_filter: str = "all",
) -> str:
    """Check child-task status with bounded pagination and explicit capacity.

    Terminal tasks remain durable history but do not consume active child slots.
    Use offset/limit to page large task histories. status_filter is all, active,
    or terminal.
    """
    state = OrchestrationState.load(chat_id)
    capacity = _task_capacity(state)
    offset, limit = _normalize_page(offset, limit)
    status_filter = str(status_filter or "all").strip().lower()
    if status_filter not in _TASK_STATUS_FILTERS:
        return json.dumps({
            "ok": False,
            "schema": "mcop_task_status.v2",
            "result_code": "INVALID_STATUS_FILTER",
            "error": "status_filter must be one of: all, active, terminal",
        }, sort_keys=True)

    if not state.tasks:
        return json.dumps({
            "ok": True,
            "schema": "mcop_task_status.v2",
            "result_code": "NO_TASKS",
            "status": "no_tasks",
            "parent_chat_id": state.parent_chat_id,
            "total_tasks": 0,
            "returned_count": 0,
            "offset": 0,
            "limit": limit,
            "truncated": False,
            "next_offset": None,
            **capacity,
            "result": {"total_tasks": 0, "active_task_count": 0},
            "message": "No child tasks have been spawned yet.",
        }, sort_keys=True)

    items = list(state.tasks.items())
    if status_filter == "active":
        items = [(tid, result) for tid, result in items if not _is_terminal_child(result)]
    elif status_filter == "terminal":
        items = [(tid, result) for tid, result in items if _is_terminal_child(result)]

    total_matching = len(items)
    page = items[offset:offset + limit]
    next_offset = offset + len(page)
    truncated = next_offset < total_matching
    tasks = {
        tid: {
            "status": result.status,
            "iterations_used": result.iterations_used,
            "tokens_used": result.tokens_used,
            "artifacts": list(result.artifacts or [])[:_ARTIFACT_PREVIEW_LIMIT],
            "artifacts_count": len(result.artifacts or []),
            "packet_path": result.packet_path,
            "facts_count": len(result.facts or []),
            "inferences_count": len(result.inferences or []),
            "gaps_count": len(result.gaps or []),
            "errors_count": len(result.errors or []),
            "summary_preview": _bounded_text(
                result.summary, offset=0, max_chars=_TASK_SUMMARY_PREVIEW_CHARS
            )["chunk"],
            "summary_chars_total": len(str(result.summary or "")),
            "summary_truncated": len(str(result.summary or "")) > _TASK_SUMMARY_PREVIEW_CHARS,
            "error_preview": _bounded_text(
                result.error, offset=0, max_chars=_ERROR_PREVIEW_CHARS
            )["chunk"] or None,
        }
        for tid, result in page
    }
    output = {
        "ok": True,
        "schema": "mcop_task_status.v2",
        "result_code": "TASK_STATUS_OK",
        "parent_chat_id": state.parent_chat_id,
        "total_tasks": len(state.tasks),
        "matching_tasks": total_matching,
        "returned_count": len(page),
        "offset": offset,
        "limit": limit,
        "status_filter": status_filter,
        "truncated": truncated,
        "next_offset": next_offset if truncated else None,
        "total_tokens_used": state.total_tokens,
        **capacity,
        "tasks": tasks,
        "result": {
            "total_tasks": len(state.tasks),
            "matching_tasks": total_matching,
            "active_task_count": capacity["active_task_count"],
            "terminal_task_count": capacity["terminal_task_count"],
        },
    }
    return json.dumps(output, indent=2, default=str)


@tool("agent_read_task_result")
def agent_read_task_result(
    chat_id: str,
    task_id: str,
    summary_offset: int = 0,
    max_chars: int = _DEFAULT_TEXT_CHUNK_CHARS,
) -> str:
    """Read a completed child result without injecting an unbounded ToolMessage.

    Small results preserve the legacy full response. Large results return compact
    metadata plus a bounded summary chunk. Continue with summary_offset=next_offset
    or use agent_read_packet for the durable structured evidence.
    """
    try:
        result_file = _safe_task_file(chat_id, task_id, "result.json")
    except Exception as exc:
        return json.dumps({
            "ok": False,
            "schema": "mcop_task_result.v2",
            "result_code": "INVALID_TASK_PATH",
            "error": str(exc),
        }, sort_keys=True)

    if not result_file.exists():
        return json.dumps({
            "ok": False,
            "schema": "mcop_task_result.v2",
            "result_code": "TASK_RESULT_NOT_FOUND",
            "error": f"No result found for task '{task_id}'. Either the task has not completed or the ID is wrong.",
        }, sort_keys=True)

    try:
        result = ChildResult.from_json(result_file.read_text())
        full = {
            "task_id": result.task_id,
            "status": result.status,
            "iterations_used": result.iterations_used,
            "tokens_used": result.tokens_used,
            "artifacts_created": result.artifacts,
            "raw_artifacts": result.raw_artifacts,
            "packet_path": result.packet_path,
            "facts": result.facts,
            "inferences": result.inferences,
            "gaps": result.gaps,
            "errors": result.errors,
            "next_recommended_step": result.next_recommended_step,
            "summary": result.summary,
            "error": result.error,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
        }
        serialized_chars = len(json.dumps(full, default=str))
        if serialized_chars <= _PARENT_RESPONSE_MAX_CHARS and int(summary_offset or 0) == 0:
            full.update({
                "ok": True,
                "schema": "mcop_task_result.v2",
                "result_code": "TASK_RESULT_OK",
                "response_truncated": False,
                "response_chars_total": serialized_chars,
                "result": {"task_id": result.task_id, "status": result.status},
            })
            return json.dumps(full, indent=2, default=str)

        summary = _bounded_text(result.summary, offset=summary_offset, max_chars=max_chars)
        compact = _compact_child_result(result, preview_chars=_TASK_SUMMARY_PREVIEW_CHARS)
        compact.update({
            "ok": True,
            "schema": "mcop_task_result.v2",
            "result_code": "TASK_RESULT_CHUNKED",
            "response_truncated": True,
            "response_chars_total": serialized_chars,
            "summary": summary["chunk"],
            "summary_offset": summary["offset"],
            "summary_returned_chars": summary["returned_chars"],
            "summary_next_offset": summary["next_offset"],
            "summary_complete": not summary["truncated"],
            "result": {"task_id": result.task_id, "status": result.status},
            "required_action": "Continue with summary_offset=summary_next_offset or read packet chunks for structured evidence.",
        })
        return json.dumps(compact, indent=2, default=str)
    except Exception as exc:
        return json.dumps({
            "ok": False,
            "schema": "mcop_task_result.v2",
            "result_code": "TASK_RESULT_READ_ERROR",
            "error": f"Failed to read result: {exc}",
        }, sort_keys=True)


@tool("agent_read_packet")
def agent_read_packet(
    chat_id: str,
    task_id: str = "",
    packet_path: str = "",
    offset: int = 0,
    max_chars: int = _DEFAULT_TEXT_CHUNK_CHARS,
) -> str:
    """Read a structured MCOP evidence packet as a bounded text chunk.

    The response reports total size, SHA-256, truncation, and next_offset. packet_path
    is restricted to the current chat's _mcop workspace to prevent path traversal.
    """
    try:
        path = _safe_packet_path(chat_id, task_id=task_id, packet_path=packet_path)
        if not path.exists():
            return json.dumps({
                "ok": False,
                "schema": "mcop_packet_chunk.v2",
                "result_code": "PACKET_NOT_FOUND",
                "error": f"Packet not found: {path}",
            }, sort_keys=True)
        raw = path.read_text()
        window = _bounded_text(raw, offset=offset, max_chars=max_chars)
        return json.dumps({
            "ok": True,
            "schema": "mcop_packet_chunk.v2",
            "result_code": "PACKET_CHUNK_OK",
            "packet_path": str(path),
            "sha256": hashlib.sha256(raw.encode()).hexdigest(),
            **window,
            "result": {
                "packet_path": str(path),
                "returned_chars": window["returned_chars"],
                "total_chars": window["total_chars"],
            },
        }, indent=2, default=str)
    except Exception as exc:
        return json.dumps({
            "ok": False,
            "schema": "mcop_packet_chunk.v2",
            "result_code": "PACKET_READ_ERROR",
            "error": f"Failed to read packet: {exc}",
        }, sort_keys=True)
