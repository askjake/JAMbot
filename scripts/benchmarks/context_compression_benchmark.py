#!/usr/bin/env python3
"""Deterministic context-compression, continuity and evidence-integrity benchmark.

Phase D3B3.

This harness exercises the *actual* production parent-path compression stack
used by ``app.agent.agents.agentic_rag.call_model`` rather than a reimplementation:

    Step 1  sanitize_tool_messages
    Step 2  truncate_messages
              -> message-count cap
              -> apply_tiered_compression
              -> apply_token_efficiency_layer
              -> truncate_large_messages
                   -> generate_progressive_levels        (aged ToolMessages)
                   -> compress_tool_message_content      (large ToolMessages)
              -> trim_messages (provider budget)
              -> sanitize_tool_messages
              -> ensure_bedrock_converse_message_shape
    Step 3  cleanup_cachept / aggressive_cachept
    Step 5  ensure_bedrock_converse_message_shape (bedrock role)

Authorization / activation state is resolved through the real Phase D1
checkpoint functions (``load_tool_policy_state`` + ``parse_authorization_delta``
+ ``merge_authorization_state``), never by re-scanning the retained window.

Token counts use the same offline counter production uses
(``app.message.compression.count_message_tokens``, tiktoken cl100k_base).  That
counter is a PROXY for Claude/Bedrock tokens, not an authoritative tokenizer.

The harness is offline, deterministic, uses no production database, no secrets,
no real user data, and executes no live tools.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# ── Real production components ────────────────────────────────────────────────
from app.agent.agents import agentic_rag as AR  # noqa: E402
from app.agent.utils import CACHE_POINT_BLOCK, aggressive_cachept, cleanup_cachept  # noqa: E402
from app.agent.tool_policy_state import (  # noqa: E402
    AUTHORIZATION_KEYS,
    load_tool_policy_state,
    merge_authorization_state,
    parse_authorization_delta,
)
from app.message.compression import count_message_tokens  # noqa: E402
from app.message.message_tiering import apply_tiered_compression  # noqa: E402
from app.message.token_efficiency_adapter import apply_token_efficiency_layer  # noqa: E402
from app.message.tool_message_compressor import compress_tool_message_content  # noqa: E402
from app.tools.progressive_tool_memory import generate_progressive_levels  # noqa: E402

SCHEMA_VERSION = "d3b3_context_compression_benchmark.v1"

# Strings that must never appear in model-visible context (leak canaries).
LEAK_CANARIES: dict[str, str] = {
    "audit_event_id": "AUDITEVT-D3B3-CANARY-0001",
    "internal_request_id": "REQ-INTERNAL-D3B3-CANARY-0002",
    "checkpoint_internal": "CKPT-ENCMETA-D3B3-CANARY-0003",
    "webhook_metadata": "WEBHOOK-TARGET-D3B3-CANARY-0004",
    "policy_state_map": "RAWPOLICYMAP-D3B3-CANARY-0005",
}


# ─────────────────────────────────────────────────────────────────────────────
# Measurement
# ─────────────────────────────────────────────────────────────────────────────
def _content_to_text(content: Any) -> str:
    """Flatten message content to text for byte/char accounting."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                if "text" in block and isinstance(block["text"], str):
                    parts.append(block["text"])
                else:
                    parts.append(json.dumps(block, sort_keys=True))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def serialize_context(messages: Sequence[BaseMessage]) -> str:
    """Stable textual projection of the model-visible message list."""
    out: list[str] = []
    for msg in messages:
        role = msg.__class__.__name__
        text = _content_to_text(getattr(msg, "content", ""))
        extra = ""
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            extra = " tool_calls=" + ",".join(
                f"{tc.get('name')}#{tc.get('id')}" for tc in tool_calls
            )
        tcid = getattr(msg, "tool_call_id", None)
        if tcid:
            extra += f" tool_call_id={tcid}"
        out.append(f"[{role}{extra}]\n{text}")
    return "\n".join(out)


def _safe_tokens(messages: Sequence[BaseMessage]) -> int:
    try:
        return int(count_message_tokens(list(messages)))
    except Exception:
        return len(serialize_context(messages)) // 4


def measure(messages: Sequence[BaseMessage]) -> dict[str, Any]:
    """Exact bytes/chars/counts plus proxy token estimates."""
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    other_msgs = [m for m in messages if not isinstance(m, ToolMessage)]
    text = serialize_context(messages)
    tool_text = serialize_context(tool_msgs)
    other_text = serialize_context(other_msgs)

    cachepoints = 0
    for m in messages:
        content = getattr(m, "content", None)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, Mapping) and "cachePoint" in block:
                    cachepoints += 1

    ai_tool_calls = sum(
        len(getattr(m, "tool_calls", None) or []) for m in messages if isinstance(m, AIMessage)
    )
    no_cp = serialize_context([m for m in messages])
    return {
        "messages": len(messages),
        "tool_messages": len(tool_msgs),
        "non_tool_messages": len(other_msgs),
        "ai_tool_calls": ai_tool_calls,
        "chars": len(text),
        "bytes": len(text.encode("utf-8")),
        "est_tokens": _safe_tokens(messages),
        "tool_chars": len(tool_text),
        "tool_bytes": len(tool_text.encode("utf-8")),
        "tool_est_tokens": _safe_tokens(tool_msgs),
        "non_tool_chars": len(other_text),
        "non_tool_bytes": len(other_text.encode("utf-8")),
        "non_tool_est_tokens": _safe_tokens(other_msgs),
        "cachepoints": cachepoints,
        "cachepoint_block_chars": cachepoints * len(json.dumps(CACHE_POINT_BLOCK, sort_keys=True)),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def pct_reduction(base: int, new: int) -> float:
    if base <= 0:
        return 0.0
    return round((base - new) * 100.0 / base, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic synthetic content generators
# ─────────────────────────────────────────────────────────────────────────────
def repetitive_log(receiver: str, source_id: str, lines: int, marker: str) -> str:
    """Deterministic, highly repetitive STB-style log payload."""
    head = [
        f"receiver_id={receiver}",
        f"source_id={source_id}",
        f"marker={marker}",
        "log_type=stbCtrl",
    ]
    body: list[str] = []
    for i in range(lines):
        sev = "ERROR" if i % 17 == 0 else ("WARN" if i % 5 == 0 else "INFO")
        body.append(
            f"[{sev}]<04/12/2026 11:{(i // 60) % 60:02d}:{i % 60:02d}.000 MDT> "
            f"stbCtrl tune_attempt seq={i} freq=573000000 state=ACQUIRING "
            f"snr=12.{i % 10} lock=false retry={i % 4} src={source_id}"
        )
    tail = [f"end_of_segment source_id={source_id} total_lines={lines}"]
    return "\n".join(head + body + tail)


def json_status_payload(source_id: str, entries: int) -> str:
    return json.dumps(
        {
            "source_id": source_id,
            "status": "ok",
            "entries": [
                {"idx": i, "name": f"item_{i:04d}", "state": "READY", "bytes": 1024 + i}
                for i in range(entries)
            ],
        },
        indent=2,
    )


def _ai_tool_call(name: str, call_id: str, args: dict[str, Any] | None = None) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "id": call_id, "args": args or {}}],
    )


def _pair(name: str, call_id: str, result: str, args: dict[str, Any] | None = None) -> list[BaseMessage]:
    """One AI tool call plus exactly one paired ToolMessage."""
    return [
        _ai_tool_call(name, call_id, args),
        ToolMessage(content=result, tool_call_id=call_id, name=name),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Fixture model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Question:
    qid: str
    kind: str          # context_fact | policy_state | evidence_label | source_identity | absent
    prompt: str
    expect: Any
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Fixture:
    name: str
    description: str
    messages: list[BaseMessage]
    large: bool
    prompt: str                                  # current-turn user text
    stored_policy: dict[str, Any] = field(default_factory=dict)
    questions: list[Question] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    expansion: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Case 1 — small control
# ─────────────────────────────────────────────────────────────────────────────
def fixture_small_control() -> Fixture:
    msgs: list[BaseMessage] = [
        HumanMessage(content="What does the acronym EPG mean in set-top box software?"),
        AIMessage(content="EPG stands for Electronic Program Guide - the on-screen listing of channels and scheduled programming."),
        HumanMessage(content="And what is an OTA tuner used for?"),
        AIMessage(content="An OTA (over-the-air) tuner receives free broadcast television signals through an antenna rather than satellite."),
    ]
    return Fixture(
        name="case1_small_control",
        description="Short generic exchange, no tools, no authorization changes.",
        messages=msgs,
        large=False,
        prompt="Summarise what we discussed.",
        questions=[
            Question("c1q1", "context_fact", "What does EPG stand for?", ["Electronic Program Guide"]),
            Question("c1q2", "context_fact", "What does an OTA tuner receive?", ["antenna"]),
        ],
        notes="Control: compression must not expand or distort a short conversation.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Case 2 — medium read-only investigation
# ─────────────────────────────────────────────────────────────────────────────
def fixture_medium_investigation() -> Fixture:
    msgs: list[BaseMessage] = [
        HumanMessage(
            content="Investigate receiver R1954841480 for guide load failures on 2026-04-12. "
                    "Read-only please."
        ),
    ]
    msgs += _pair("list_dates", "call_c2_01",
                  json_status_payload("SRC-C2-DATES", 12),
                  {"receiver_id": "R1954841480"})
    msgs += _pair("list_files", "call_c2_02",
                  json_status_payload("SRC-C2-FILES", 40),
                  {"receiver_id": "R1954841480", "date": "2026-04-12"})
    msgs += _pair("search_logs", "call_c2_03",
                  repetitive_log("R1954841480", "SRC-C2-SEARCH", 120, "guide_load_fail"),
                  {"receiver_id": "R1954841480", "pattern": "guide"})
    msgs.append(AIMessage(
        content="Observed 7 guide load failures on 2026-04-12 between 11:00 and 11:20 MDT. "
                "UNRESOLVED QUESTION: whether the sddp_log for the same window shows a "
                "matching backend timeout, because sddp_log was not in the uploaded profile."
    ))
    msgs.append(HumanMessage(content="Check the receiver health summary too."))
    msgs += _pair("get_summary", "call_c2_04",
                  json_status_payload("SRC-C2-SUMMARY", 25),
                  {"receiver_id": "R1954841480"})
    msgs.append(AIMessage(
        content="Health summary retrieved. The unresolved sddp_log question remains open."
    ))
    return Fixture(
        name="case2_medium_investigation",
        description="Read-only receiver-log investigation with several tool results and one unresolved question.",
        messages=msgs,
        large=False,
        prompt="Continue the investigation.",
        stored_policy={"active_toolsets": ["s3_receiver_logs"],
                       "requested_extra_tools": []},
        questions=[
            Question("c2q1", "context_fact", "Which receiver was investigated?", ["R1954841480"]),
            Question("c2q2", "context_fact", "Which date was investigated?", ["2026-04-12"]),
            Question("c2q3", "context_fact", "What question remained unresolved?", ["sddp_log"]),
        ],
        notes="Domain continuity and unresolved-question preservation.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Case 3 — large repetitive tool-output workload  (primary large case)
# ─────────────────────────────────────────────────────────────────────────────
def fixture_large_tool_output() -> Fixture:
    msgs: list[BaseMessage] = []
    expansion_target = "SRC-C3-SEG07"
    for turn in range(1, 9):
        msgs.append(HumanMessage(
            content=f"Read log segment {turn} for receiver R1958038467 and report tune failures."
        ))
        src = f"SRC-C3-SEG{turn:02d}"
        payload = repetitive_log("R1958038467", src, 900, f"segment_{turn}")
        msgs += _pair("read_log", f"call_c3_{turn:02d}", payload,
                      {"receiver_id": "R1958038467", "filename": f"seg{turn}.gz"})
        msgs.append(AIMessage(
            content=f"Segment {turn} ({src}) parsed: repeated ACQUIRING states with lock=false."
        ))
    # A final large JSON status payload on the current turn.
    msgs.append(HumanMessage(content="Now pull the consolidated status payload."))
    msgs += _pair("get_summary", "call_c3_final",
                  json_status_payload("SRC-C3-FINAL", 1200),
                  {"receiver_id": "R1958038467"})
    msgs.append(AIMessage(
        content="Consolidated payload (SRC-C3-FINAL) retrieved. No successful lock observed in any segment."
    ))
    return Fixture(
        name="case3_large_tool_output",
        description="Many large, highly repetitive log/search outputs plus a large JSON status payload.",
        messages=msgs,
        large=True,
        prompt="Summarise the tune failure pattern across all segments.",
        questions=[
            Question("c3q1", "context_fact", "Which receiver was analysed?", ["R1958038467"]),
            Question("c3q2", "context_fact", "Was a successful lock observed?", ["No successful lock"]),
        ],
        evidence={"segments": [f"SRC-C3-SEG{i:02d}" for i in range(1, 9)] + ["SRC-C3-FINAL"]},
        expansion={
            "tool_call_id": "call_c3_07",
            "tool_name": "read_log",
            "source_id": expansion_target,
            "expect_substring": f"end_of_segment source_id={expansion_target}",
        },
        notes="Primary large token case: tool-result compression, dedup, targeted expansion.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Case 4 — long authorization lifecycle
# ─────────────────────────────────────────────────────────────────────────────
def fixture_authorization_lifecycle() -> Fixture:
    msgs: list[BaseMessage] = [
        HumanMessage(content="Build a log capsule for receiver R1948258330 covering 2026-04-12."),
        AIMessage(content="build_log_capsule is a heavy tool and is not currently authorized. "
                          "It is recorded as requested and pending authorization."),
        HumanMessage(content="Also prepare an incident scene for the same window."),
        AIMessage(content="build_incident_scene is not available upstream in this runtime. "
                          "It is recorded as unavailable."),
        HumanMessage(content="Heavy tools authorized."),
        AIMessage(content="Heavy tool authorization recorded. build_log_capsule is now eligible."),
    ]
    # Filler turns with tool traffic so the history is long enough to compress.
    for turn in range(1, 7):
        msgs.append(HumanMessage(content=f"Show me diagnostic step {turn}."))
        msgs += _pair("search_logs", f"call_c4_{turn:02d}",
                      repetitive_log("R1948258330", f"SRC-C4-{turn:02d}", 400, f"diag_{turn}"),
                      {"receiver_id": "R1948258330"})
        msgs.append(AIMessage(content=f"Diagnostic step {turn} complete."))
    msgs.append(HumanMessage(content="Heavy tools are no longer authorized."))
    msgs.append(AIMessage(content="Heavy tool authorization revoked. build_log_capsule is no longer eligible."))
    return Fixture(
        name="case4_authorization_lifecycle",
        description="Heavy request without authorization, pending, grant, eligible, revocation, follow-up.",
        messages=msgs,
        large=True,
        prompt="Continue.",
        stored_policy={
            "authorization_flags": {
                "heavy_tools_authorized": False,
                "persistence_authorized": False,
                "operator_authorized": False,
                "mutation_authorized": False,
            },
            "requested_extra_tools": ["build_log_capsule", "build_incident_scene"],
            "pending_authorization_extra_tools": ["build_log_capsule"],
            "unavailable_extra_tools": ["build_incident_scene"],
        },
        questions=[
            Question("c4q1", "policy_state", "Which authorization was later revoked?",
                     {"heavy_tools_authorized": False}),
            Question("c4q2", "policy_state", "Which exact tool remained requested/pending?",
                     {"requested_contains": "build_log_capsule"}),
            Question("c4q3", "policy_state", "Which Incident Scene tool was unavailable?",
                     {"unavailable_contains": "build_incident_scene"}),
        ],
        notes="Checkpoint state, not retained prose, must control authorization.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Case 5 — operational repository workflow (D3B2A regression)
# ─────────────────────────────────────────────────────────────────────────────
REPO_TARGET = "git@gitlab.com:dish-cloud/dt/sse/datasolutions/cas/ai-test.git"
REPO_COMMIT = "3a191e7163382c342999eacd4f3d6ed5535b8000"


def fixture_operational_workflow() -> Fixture:
    msgs: list[BaseMessage] = [
        HumanMessage(content=f"Clone {REPO_TARGET}, review it, set up an isolated venv and run its tests."),
        AIMessage(content="This is a repository checkout and local deployment workflow. "
                          "agent_git_clone, agent_run_shell and agent_create_venv are requested and "
                          "pending authorization because they are privileged."),
        HumanMessage(content="Operator authorized. Heavy tools authorized."),
        AIMessage(content="Operational authorization recorded."),
    ]
    msgs += _pair("agent_git_clone", "call_c5_01",
                  f"Cloned {REPO_TARGET} at commit {REPO_COMMIT}\n" + json_status_payload("SRC-C5-CLONE", 30),
                  {"repo_url": REPO_TARGET})
    msgs.append(AIMessage(content=f"Clone complete at commit {REPO_COMMIT}. Starting protocol and security review."))
    msgs += _pair("agent_run_shell", "call_c5_02",
                  repetitive_log("n/a", "SRC-C5-REVIEW", 300, "protocol_review"),
                  {"command": "grep -RIn pattern"})
    msgs.append(HumanMessage(content="Continue."))
    msgs += _pair("agent_create_venv", "call_c5_03",
                  json_status_payload("SRC-C5-VENV", 60), {"python_bin": "python3.11"})
    msgs.append(AIMessage(content="Isolated venv created; dependencies installed."))
    msgs.append(HumanMessage(content="Run the next step."))
    msgs += _pair("agent_run_shell", "call_c5_04",
                  "205 passed in 41.2s\n" + repetitive_log("n/a", "SRC-C5-TESTS", 500, "pytest_run"),
                  {"command": "pytest -q"})
    msgs.append(AIMessage(content="Repository tests: 205 passed. Cleanup command defined as "
                                  "'rm -rf /tmp/d3b3-ai-test-clone' for the isolated clone only. "
                                  "Constraint: no production infrastructure may be modified."))
    msgs.append(HumanMessage(content="Revoke operator authorization and heavy tool authorization."))
    msgs.append(AIMessage(content="Operational authorization revoked; further privileged execution is blocked."))
    return Fixture(
        name="case5_operational_workflow",
        description="Repository checkout / local deploy workflow with generic continuations and revocation.",
        messages=msgs,
        large=True,
        prompt="Continue.",
        stored_policy={
            "active_toolsets": ["agent_mode"],
            "requested_extra_tools": [
                "agent_mode:agent_git_clone",
                "agent_mode:agent_run_shell",
                "agent_mode:agent_create_venv",
            ],
            "authorization_flags": {
                "heavy_tools_authorized": False,
                "persistence_authorized": False,
                "operator_authorized": False,
                "mutation_authorized": False,
            },
        },
        questions=[
            Question("c5q1", "context_fact", "What repository was requested?", ["ai-test.git"]),
            Question("c5q2", "context_fact", "Which commit was tested?", [REPO_COMMIT[:12]]),
            Question("c5q3", "context_fact", "What cleanup command was defined?", ["rm -rf /tmp/d3b3-ai-test-clone"]),
            Question("c5q4", "context_fact", "What constraint prohibited production modification?",
                     ["no production infrastructure"]),
            Question("c5q5", "policy_state", "Which authorization was later revoked?",
                     {"operator_authorized": False, "heavy_tools_authorized": False}),
        ],
        notes="Direct D3B2A regression: workflow must survive compression and generic continuation.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Case 6 — parent/child evidence workflow
# ─────────────────────────────────────────────────────────────────────────────
def fixture_parent_child() -> Fixture:
    msgs: list[BaseMessage] = [
        HumanMessage(content="Investigate two receivers in parallel: R1955245852 (logs) and R1947782731 (QoS)."),
        AIMessage(content="Dispatching two MCOP children with disjoint read-only task families."),
    ]
    # Child A — receiver logs
    msgs += _pair("mcop_child_task", "call_c6_A",
                  "child_run_id=CHILD-A-0001 family=s3_receiver_logs\n"
                  + repetitive_log("R1955245852", "SRC-C6-CHILDA", 500, "child_a_logs")
                  + "\nCHILD-A RESULT: 3 stbCtrl restarts observed.",
                  {"task_id": "childA", "family": "s3_receiver_logs"})
    # Child B — QoS
    msgs += _pair("mcop_child_task", "call_c6_B",
                  "child_run_id=CHILD-B-0002 family=qos\n"
                  + json_status_payload("SRC-C6-CHILDB", 200)
                  + "\nCHILD-B RESULT: failed - QoS partition unavailable for the window."
                    "\nCHILD-B UNRESOLVED: whether the QoS gap is ingestion lag or genuine absence.",
                  {"task_id": "childB", "family": "qos"})
    msgs.append(AIMessage(
        content="Parent synthesis: CHILD-A-0001 (s3_receiver_logs) observed 3 stbCtrl restarts on "
                "R1955245852. CHILD-B-0002 (qos) failed; its unresolved question is whether the QoS "
                "gap is ingestion lag or genuine absence. Children remain isolated; no cross-child "
                "tool state was shared."
    ))
    for turn in range(1, 5):
        msgs.append(HumanMessage(content=f"Expand on synthesis point {turn}."))
        msgs += _pair("search_logs", f"call_c6_{turn:02d}",
                      repetitive_log("R1955245852", f"SRC-C6-{turn:02d}", 350, f"synth_{turn}"),
                      {"receiver_id": "R1955245852"})
        msgs.append(AIMessage(content=f"Synthesis point {turn} expanded."))
    return Fixture(
        name="case6_parent_child",
        description="Parent plus two MCOP children with different read-only families, one failure, one unresolved question.",
        messages=msgs,
        large=True,
        prompt="Continue the synthesis.",
        questions=[
            Question("c6q1", "context_fact", "Which child produced the stbCtrl restart result?", ["CHILD-A-0001"]),
            Question("c6q2", "context_fact", "Which child failed?", ["CHILD-B-0002"]),
            Question("c6q3", "context_fact", "What question remained unresolved for the failing child?",
                     ["ingestion lag"]),
        ],
        evidence={"children": ["CHILD-A-0001", "CHILD-B-0002"]},
        notes="Child attribution and isolation must survive compression.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Case 7 — evidence summary and targeted expansion
# ─────────────────────────────────────────────────────────────────────────────
EVIDENCE_ITEMS = [
    ("SRC-C7-0001", "OBSERVED", "stbCtrl logged tune_attempt seq=41 with lock=false at 11:04:12 MDT."),
    ("SRC-C7-0002", "OBSERVED", "sddp_log recorded backend_timeout=1 at 11:04:15 MDT."),
    ("SRC-C7-0003", "DERIVED", "Tuner state transitioned ACQUIRING -> FAILED without an intermediate LOCKED state."),
    ("SRC-C7-0004", "CORRELATED", "Backend timeout events co-occur with tune failures within a 3 second window."),
    ("SRC-C7-0005", "HYPOTHESIS", "A backend guide timeout may be preventing tuner lock completion."),
    ("SRC-C7-0006", "MISSING_EVIDENCE", "qt_gui log family was not uploaded for this window; UI-side confirmation is unavailable."),
    ("SRC-C7-0007", "QUALITY_WARNING", "Only 2 of 5 expected log families were present; coverage is partial."),
]


def fixture_evidence_summary() -> Fixture:
    rendered = "\n".join(f"[{label}] {sid}: {text}" for sid, label, text in EVIDENCE_ITEMS)
    msgs: list[BaseMessage] = [
        HumanMessage(content="Summarise the incident evidence for receiver R1948258330 and label every fact."),
    ]
    msgs += _pair("diagnose_receiver_from_logs", "call_c7_01",
                  "EVIDENCE INDEX\n" + rendered + "\n\nRAW REGION\n"
                  + repetitive_log("R1948258330", "SRC-C7-RAW", 800, "evidence_raw"),
                  {"receiver_id": "R1948258330"})
    msgs.append(AIMessage(content="Evidence index:\n" + rendered
                          + "\n\nNo causal conclusion is asserted; SRC-C7-0005 remains a hypothesis."))
    for turn in range(1, 4):
        msgs.append(HumanMessage(content=f"Add supporting detail {turn}."))
        msgs += _pair("search_logs", f"call_c7_s{turn:02d}",
                      repetitive_log("R1948258330", f"SRC-C7-S{turn:02d}", 400, f"support_{turn}"),
                      {"receiver_id": "R1948258330"})
        msgs.append(AIMessage(content=f"Supporting detail {turn} added; labels unchanged."))
    return Fixture(
        name="case7_evidence_summary",
        description="Capsule/scene-like evidence with observed/derived/correlated/hypothesis labels and targeted expansion.",
        messages=msgs,
        large=True,
        prompt="Restate the labelled evidence.",
        questions=[
            Question("c7q1", "evidence_label", "Which evidence was observed versus hypothesised?",
                     {"observed": ["SRC-C7-0001", "SRC-C7-0002"], "hypothesis": ["SRC-C7-0005"]}),
            Question("c7q2", "context_fact", "What evidence was missing?", ["qt_gui"]),
            Question("c7q3", "absent", "No unsupported causal claim may appear.",
                     ["confirmed root cause", "proven cause", "definitively caused"]),
        ],
        evidence={"items": [{"source_id": s, "label": l, "text": t} for s, l, t in EVIDENCE_ITEMS]},
        expansion={
            "tool_call_id": "call_c7_01",
            "tool_name": "diagnose_receiver_from_logs",
            "source_id": "SRC-C7-RAW",
            "expect_substring": "end_of_segment source_id=SRC-C7-RAW",
        },
        notes="Evidence-category separation and resolvable identities.",
    )


ALL_FIXTURES: list[Callable[[], Fixture]] = [
    fixture_small_control,
    fixture_medium_investigation,
    fixture_large_tool_output,
    fixture_authorization_lifecycle,
    fixture_operational_workflow,
    fixture_parent_child,
    fixture_evidence_summary,
]


# ─────────────────────────────────────────────────────────────────────────────
# Modes — all use real production functions
# ─────────────────────────────────────────────────────────────────────────────
def _settings():
    from app.config import get_settings
    return get_settings()


def _bedrock_shape_required() -> bool:
    return bool(AR._active_provider_requires_bedrock_shape())


def mode_a_raw(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Baseline: provider-valid but no semantic or tool-result compression."""
    out = AR.sanitize_tool_messages(list(messages))
    if _bedrock_shape_required():
        out = AR.ensure_bedrock_converse_message_shape(out)
    return out


def mode_b_production(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Exact current production parent pipeline (call_model steps 1-5)."""
    out = AR.sanitize_tool_messages(list(messages))
    out = AR.truncate_messages(out, AR.MAX_MESSAGES)
    cleanup_cachept(out)
    out = aggressive_cachept(out, _settings().MAX_CACHEPOINT_CNT)
    if _bedrock_shape_required():
        out = AR.ensure_bedrock_converse_message_shape(out)
    return out


def mode_c_tool_only(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Tool-result compression only; all other history retained."""
    out: list[BaseMessage] = []
    for msg in AR.sanitize_tool_messages(list(messages)):
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            if count_message_tokens([msg]) > AR.TOOL_MESSAGE_COMPRESS_TOKEN_THRESHOLD:
                res = compress_tool_message_content(
                    tool_name=getattr(msg, "name", None) or "unknown",
                    content=msg.content,
                    tool_call_id=getattr(msg, "tool_call_id", None),
                    max_chars=AR.TOOL_COMPRESSED_MAX_CHARS,
                    first_lines=AR.TOOL_FIRST_LINES,
                )
                out.append(AR._copy_tool_message_with_content(msg, res.content))
                continue
        out.append(msg)
    out = AR.sanitize_tool_messages(out)
    if _bedrock_shape_required():
        out = AR.ensure_bedrock_converse_message_shape(out)
    return out


def mode_d_progressive_tool(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Progressive tool memory plus tool-result compression (real truncate_large_messages)."""
    out = AR.sanitize_tool_messages(list(messages))
    out = AR.truncate_large_messages(out, max_tokens_per_message=50000)
    out = AR.sanitize_tool_messages(out)
    if _bedrock_shape_required():
        out = AR.ensure_bedrock_converse_message_shape(out)
    return out


AUTH_BEARING_MARKERS = (
    "Heavy tools authorized",
    "Heavy tools are no longer authorized",
    "Operator authorized",
    "Revoke operator authorization",
    "heavy_tools_authorized",
    "operator_authorized",
    "build_log_capsule",
    "build_incident_scene",
    "Clone git@gitlab.com",
)


def drop_policy_bearing_messages(messages: list[BaseMessage]) -> tuple[list[BaseMessage], int]:
    """Physically remove every message whose text carries authorization or
    original-request wording.  This simulates the worst realistic compression
    outcome for policy-relevant prose."""
    kept: list[BaseMessage] = []
    dropped = 0
    for msg in messages:
        text = _content_to_text(getattr(msg, "content", ""))
        if any(marker in text for marker in AUTH_BEARING_MARKERS):
            dropped += 1
            continue
        kept.append(msg)
    return kept, dropped


def mode_e_checkpoint_compressed(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Production pipeline over history with all policy-bearing prose removed."""
    stripped, _ = drop_policy_bearing_messages(list(messages))
    return mode_b_production(stripped)


MODES: dict[str, Callable[[list[BaseMessage]], list[BaseMessage]]] = {
    "A_raw": mode_a_raw,
    "B_production": mode_b_production,
    "C_tool_only": mode_c_tool_only,
    "D_progressive_tool": mode_d_progressive_tool,
    "E_checkpoint_compressed": mode_e_checkpoint_compressed,
}


# ─────────────────────────────────────────────────────────────────────────────
# Tool-call / ToolMessage integrity and provider ordering
# ─────────────────────────────────────────────────────────────────────────────
def check_tool_message_integrity(messages: Sequence[BaseMessage]) -> dict[str, Any]:
    problems: list[str] = []
    call_index: dict[str, int] = {}
    result_index: dict[str, list[int]] = {}

    for idx, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            for tc in (getattr(msg, "tool_calls", None) or []):
                cid = tc.get("id")
                if not cid:
                    problems.append(f"index {idx}: AI tool call without id")
                    continue
                if cid in call_index:
                    problems.append(f"duplicate AI tool call id {cid}")
                call_index[cid] = idx
        elif isinstance(msg, ToolMessage):
            cid = getattr(msg, "tool_call_id", None)
            if not cid:
                problems.append(f"index {idx}: ToolMessage without tool_call_id")
                continue
            result_index.setdefault(cid, []).append(idx)

    for cid, idxs in result_index.items():
        if len(idxs) > 1:
            problems.append(f"duplicate ToolMessage for {cid}")
        if cid not in call_index:
            problems.append(f"orphaned ToolMessage {cid}")
        elif idxs[0] < call_index[cid]:
            problems.append(f"ToolMessage {cid} precedes its AI call")

    for cid in call_index:
        if cid not in result_index:
            problems.append(f"AI tool call {cid} has no ToolMessage")

    leading_tool = bool(messages) and isinstance(messages[0], ToolMessage)
    if leading_tool:
        problems.append("first message is a ToolMessage")

    return {
        "ai_tool_calls": len(call_index),
        "tool_messages": sum(len(v) for v in result_index.values()),
        "paired": len([c for c in call_index if c in result_index]),
        "problems": problems,
        "pass": not problems,
    }


def run_provider_repair_scenarios() -> dict[str, Any]:
    """Malformed but recoverable histories through the real repair functions."""
    scenarios: dict[str, list[BaseMessage]] = {}

    scenarios["orphaned_tool_message"] = [
        HumanMessage(content="hello"),
        ToolMessage(content="orphan result", tool_call_id="ghost_1", name="search_logs"),
        AIMessage(content="done"),
    ]
    scenarios["duplicate_tool_message"] = [
        HumanMessage(content="hello"),
        _ai_tool_call("search_logs", "dup_1"),
        ToolMessage(content="first", tool_call_id="dup_1", name="search_logs"),
        ToolMessage(content="second", tool_call_id="dup_1", name="search_logs"),
        AIMessage(content="done"),
    ]
    scenarios["ai_call_without_result"] = [
        HumanMessage(content="hello"),
        _ai_tool_call("search_logs", "lost_1"),
        AIMessage(content="done"),
    ]
    scenarios["partial_retry_sequence"] = [
        HumanMessage(content="hello"),
        AIMessage(content="", tool_calls=[
            {"name": "search_logs", "id": "r1", "args": {}},
            {"name": "list_files", "id": "r2", "args": {}},
        ]),
        ToolMessage(content="only one result", tool_call_id="r1", name="search_logs"),
        AIMessage(content="retrying"),
        _ai_tool_call("search_logs", "r3"),
        ToolMessage(content="retry result", tool_call_id="r3", name="search_logs"),
    ]
    scenarios["leading_tool_message_after_trim"] = [
        ToolMessage(content="cut boundary", tool_call_id="cut_1", name="search_logs"),
        HumanMessage(content="continue"),
        AIMessage(content="ok"),
    ]
    scenarios["blocked_result_pair"] = [
        HumanMessage(content="run privileged thing"),
        _ai_tool_call("agent_run_shell", "blk_1"),
        ToolMessage(content="BLOCKED_UNAUTHORIZED_PRIVILEGED_TOOL",
                    tool_call_id="blk_1", name="agent_run_shell"),
        AIMessage(content="blocked"),
    ]

    results: dict[str, Any] = {}
    for name, msgs in scenarios.items():
        repaired = AR.sanitize_tool_messages(list(msgs))
        if _bedrock_shape_required():
            repaired = AR.ensure_bedrock_converse_message_shape(repaired)
        integ = check_tool_message_integrity(repaired)
        fabricated = any(
            isinstance(m, ToolMessage)
            and "success" in _content_to_text(m.content).lower()
            and not any(_content_to_text(o.content).lower().find("success") >= 0
                        for o in msgs if isinstance(o, ToolMessage))
            for m in repaired
        )
        blocked_preserved = None
        if name == "blocked_result_pair":
            blocked_preserved = any(
                "BLOCKED_UNAUTHORIZED_PRIVILEGED_TOOL" in _content_to_text(m.content)
                for m in repaired
            )
        results[name] = {
            "input_messages": len(msgs),
            "repaired_messages": len(repaired),
            "provider_valid": integ["pass"],
            "problems": integ["problems"],
            "fabricated_success_result": fabricated,
            "blocked_result_preserved": blocked_preserved,
        }
    overall = all(
        r["provider_valid"] and not r["fabricated_success_result"] for r in results.values()
    ) and results["blocked_result_pair"]["blocked_result_preserved"] is True
    return {"scenarios": results, "pass": overall}


# ─────────────────────────────────────────────────────────────────────────────
# Policy / authorization invariants (real Phase D1 checkpoint path)
# ─────────────────────────────────────────────────────────────────────────────
def resolve_policy(stored_seed: Mapping[str, Any] | None, current_text: str) -> dict[str, Any]:
    stored = load_tool_policy_state(dict(stored_seed or {}))
    delta = parse_authorization_delta(current_text)
    flags = merge_authorization_state(stored["authorization_flags"], delta)
    return {
        "stored": stored,
        "delta": delta.as_dict() if hasattr(delta, "as_dict") else dict(delta),
        "flags": flags,
    }


def run_authorization_invariants() -> dict[str, Any]:
    """Each invariant removes the policy-bearing message and then re-resolves."""
    out: dict[str, Any] = {}
    observations: dict[str, Any] = {}

    # 1. Grant compressed out -> stays granted.
    granted = resolve_policy({}, "Heavy tools authorized.")["flags"]
    stored_after_grant = {"authorization_flags": granted}
    after = resolve_policy(stored_after_grant, "Continue.")["flags"]
    out["grant_compressed_out"] = {
        "granted_at_grant_turn": granted["heavy_tools_authorized"],
        "granted_after_compression": after["heavy_tools_authorized"],
        "pass": granted["heavy_tools_authorized"] is True and after["heavy_tools_authorized"] is True,
    }

    # 2. Revocation compressed out -> stays revoked.
    revoked = resolve_policy(stored_after_grant, "Heavy tools are no longer authorized.")["flags"]
    stored_after_revoke = {"authorization_flags": revoked}
    after_rev = resolve_policy(stored_after_revoke, "Continue.")["flags"]
    out["revocation_compressed_out"] = {
        "revoked_at_revoke_turn": revoked["heavy_tools_authorized"],
        "revoked_after_compression": after_rev["heavy_tools_authorized"],
        "pass": revoked["heavy_tools_authorized"] is False and after_rev["heavy_tools_authorized"] is False,
    }

    # 3. The D3B3 invariant: a compression-generated summary or codebook legend
    #    must never become the current-turn authorization source.  Compression
    #    inserts synthesised HumanMessages at the FRONT of history, so the real
    #    current user turn stays last and remains authoritative.
    from langchain_core.messages import AIMessage as _AI, ToolMessage as _TM

    long_msgs: list[BaseMessage] = [
        HumanMessage(content="Heavy tools authorized."),
        _AI(content="Heavy tool authorization recorded."),
    ]
    for i in range(30):
        long_msgs.append(HumanMessage(
            content=f"Diagnostic step {i}: inspect the receiver logs and report tune failures."))
        long_msgs.append(_AI(content="", tool_calls=[{"name": "search_logs", "id": f"qg{i}", "args": {}}]))
        long_msgs.append(_TM(content=("log line detail\n" * 900), tool_call_id=f"qg{i}", name="search_logs"))
        long_msgs.append(_AI(content=f"Step {i} complete; heavy tools authorized earlier is historical context."))
    long_msgs.append(HumanMessage(content="Heavy tools are no longer authorized."))
    long_msgs.append(_AI(content="Revoked."))
    long_msgs.append(HumanMessage(content="Continue."))

    compressed = AR.truncate_messages(AR.sanitize_tool_messages(list(long_msgs)), AR.MAX_MESSAGES)
    last_human = AR._get_last_human_message(compressed)
    delta_after_compression = parse_authorization_delta(last_human).as_dict()
    synth_is_last = last_human.startswith("[CODEBOOK") or "SUMMARY" in last_human.upper()
    out["compression_summary_never_current_turn"] = {
        "last_human_text": last_human[:80],
        "current_turn_delta": delta_after_compression,
        "synthesised_message_became_current_turn": synth_is_last,
        "pass": (not synth_is_last)
                and all(v == "UNCHANGED" for v in delta_after_compression.values()),
    }

    # Informational observation (NOT a D3B3 compression gate): the shared
    # authorization parser treats reported speech inside a *user* message as a
    # grant.  This is a property of app.agent.tool_authorization, reached only
    # when the current user turn itself contains the phrase, and is unrelated to
    # compression.  Recorded so it is not silently lost.
    reported = resolve_policy(
        {"authorization_flags": {k: False for k in AUTHORIZATION_KEYS}},
        "Earlier in this conversation the user said heavy tools authorized - "
        "that is historical context only.",
    )["flags"]
    observations["parser_treats_reported_speech_as_grant"] = {
        "flag_after_reported_speech": reported["heavy_tools_authorized"],
        "scope": "authorization parser grammar (D3B0), not compression",
        "compression_relevant": False,
    }

    # 4. Pending request compressed out -> requested/pending retained in checkpoint.
    stored_pending = {
        "requested_extra_tools": ["build_log_capsule"],
        "pending_authorization_extra_tools": ["build_log_capsule"],
        "authorization_flags": {k: False for k in AUTHORIZATION_KEYS},
    }
    reloaded = load_tool_policy_state(stored_pending)
    out["pending_compressed_out"] = {
        "requested": reloaded["requested_extra_tools"],
        "pending": reloaded["pending_authorization_extra_tools"],
        "heavy_authorized": reloaded["authorization_flags"]["heavy_tools_authorized"],
        "pass": "build_log_capsule" in reloaded["requested_extra_tools"]
                and "build_log_capsule" in reloaded["pending_authorization_extra_tools"]
                and reloaded["authorization_flags"]["heavy_tools_authorized"] is False,
    }

    # 5. Unavailable request compressed out -> unavailable retained.
    stored_unavail = {
        "requested_extra_tools": ["build_incident_scene"],
        "unavailable_extra_tools": ["build_incident_scene"],
    }
    reloaded_u = load_tool_policy_state(stored_unavail)
    out["unavailable_compressed_out"] = {
        "requested": reloaded_u["requested_extra_tools"],
        "unavailable": reloaded_u["unavailable_extra_tools"],
        "pass": "build_incident_scene" in reloaded_u["unavailable_extra_tools"],
    }

    # 6. Operational task compressed out -> workflow target restored from checkpoint.
    stored_op = {
        "active_toolsets": ["agent_mode"],
        "requested_extra_tools": [
            "agent_mode:agent_git_clone",
            "agent_mode:agent_run_shell",
            "agent_mode:agent_create_venv",
        ],
        "authorization_flags": {k: False for k in AUTHORIZATION_KEYS},
    }
    reloaded_op = load_tool_policy_state(stored_op)
    op_flags = resolve_policy(stored_op, "Continue the previous task.")["flags"]
    out["operational_task_compressed_out"] = {
        "requested": reloaded_op["requested_extra_tools"],
        "active_toolsets": reloaded_op["active_toolsets"],
        "flags_after_generic_continue": op_flags,
        "pass": set(reloaded_op["requested_extra_tools"]) == set(stored_op["requested_extra_tools"])
                and reloaded_op["active_toolsets"] == ["agent_mode"]
                and op_flags["operator_authorized"] is False,
    }

    out["pass"] = all(
        v.get("pass") for k, v in out.items()
        if isinstance(v, dict) and "pass" in v
    )
    out["observations"] = observations
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Continuity, claim safety and evidence integrity
# ─────────────────────────────────────────────────────────────────────────────
CAUSAL_CLAIM_PHRASES = (
    "confirmed root cause",
    "proven cause",
    "definitively caused",
    "the root cause is",
    "this proves",
    "conclusively caused",
)

FALSE_NEGATIVE_PHRASES = (
    "no errors were found",
    "no failures were found",
    "no evidence of any",
    "nothing was found",
    "there were no problems",
    "the log is clean",
)


def count_phrases(text: str, phrases: Sequence[str]) -> dict[str, int]:
    low = text.lower()
    return {p: low.count(p) for p in phrases if low.count(p)}


def evaluate_questions(
    fixture: Fixture,
    visible: Sequence[BaseMessage],
    policy: Mapping[str, Any],
    policy_questions_only: bool = False,
) -> dict[str, Any]:
    text = serialize_context(visible)
    details: list[dict[str, Any]] = []
    missing = 0
    incorrect = 0

    skipped: list[dict[str, str]] = []
    for q in fixture.questions:
        if policy_questions_only and q.kind != "policy_state":
            skipped.append({
                "qid": q.qid,
                "kind": q.kind,
                "reason": "mode E intentionally deletes policy-bearing narrative; "
                          "context-fact recall is not the property under test in this mode",
            })
            continue
        ok = True
        found: dict[str, Any] = {}
        if q.kind == "context_fact":
            for needle in q.expect:
                present = needle in text
                found[needle] = present
                ok = ok and present
            if not ok:
                missing += 1
        elif q.kind == "policy_state":
            for key, want in q.expect.items():
                if key == "requested_contains":
                    got = want in policy["stored"]["requested_extra_tools"]
                    found[key] = got
                    ok = ok and got is True
                    continue
                if key == "unavailable_contains":
                    got = want in policy["stored"]["unavailable_extra_tools"]
                    found[key] = got
                    ok = ok and got is True
                    continue
                got = policy["flags"].get(key)
                found[key] = got
                ok = ok and got == want
            if not ok:
                incorrect += 1
        elif q.kind == "evidence_label":
            for sid in q.expect.get("observed", []):
                present = f"[OBSERVED] {sid}" in text
                found[f"observed:{sid}"] = present
                ok = ok and present
            for sid in q.expect.get("hypothesis", []):
                labelled = f"[HYPOTHESIS] {sid}" in text
                mislabelled = f"[OBSERVED] {sid}" in text
                found[f"hypothesis:{sid}"] = labelled
                found[f"hypothesis_promoted:{sid}"] = mislabelled
                ok = ok and labelled and not mislabelled
            if not ok:
                incorrect += 1
        elif q.kind == "absent":
            for needle in q.expect:
                present = needle.lower() in text.lower()
                found[needle] = present
                ok = ok and not present
            if not ok:
                incorrect += 1
        details.append({"qid": q.qid, "kind": q.kind, "prompt": q.prompt,
                        "pass": ok, "evidence": found})

    passed = sum(1 for d in details if d["pass"])
    return {
        "skipped_not_applicable": skipped,
        "total": len(details),
        "passed": passed,
        "failed": len(details) - passed,
        "missing_expected_facts": missing,
        "incorrect_facts": incorrect,
        "details": details,
        "pass": passed == len(details),
    }


def evaluate_evidence_integrity(fixture: Fixture, visible: Sequence[BaseMessage]) -> dict[str, Any]:
    if not fixture.evidence.get("items"):
        return {"applicable": False}
    text = serialize_context(visible)
    per_item: list[dict[str, Any]] = []
    label_ok = True
    identity_ok = True
    for item in fixture.evidence["items"]:
        sid, label = item["source_id"], item["label"]
        id_present = sid in text
        label_present = f"[{label}] {sid}" in text
        promoted = label in ("HYPOTHESIS", "CORRELATED", "DERIVED") and f"[OBSERVED] {sid}" in text
        per_item.append({
            "source_id": sid, "label": label,
            "identity_resolvable": id_present,
            "label_preserved": label_present,
            "promoted_to_observed": promoted,
        })
        identity_ok = identity_ok and id_present
        label_ok = label_ok and label_present and not promoted
    warnings_present = all(
        any(i["source_id"] == sid and i["label_preserved"] for i in per_item)
        for sid, lbl, _ in EVIDENCE_ITEMS if lbl in ("MISSING_EVIDENCE", "QUALITY_WARNING")
    )
    return {
        "applicable": True,
        "items": per_item,
        "labels_preserved": label_ok,
        "identities_resolvable": identity_ok,
        "missing_and_quality_warnings_preserved": warnings_present,
        "pass": label_ok and identity_ok and warnings_present,
    }


def evaluate_targeted_expansion(fixture: Fixture) -> dict[str, Any]:
    """Targeted expansion reads the retained checkpoint message, which production
    never overwrites with compressed content (call_model returns only the new
    AIMessage)."""
    spec = fixture.expansion
    if not spec:
        return {"applicable": False}
    original = None
    for msg in fixture.messages:
        if isinstance(msg, ToolMessage) and getattr(msg, "tool_call_id", None) == spec["tool_call_id"]:
            original = msg
            break
    if original is None:
        return {"applicable": True, "pass": False, "error": "anchor ToolMessage not found"}

    progressive = generate_progressive_levels(
        tool_name=spec["tool_name"],
        raw_result=original.content,
        tool_call_id=spec["tool_call_id"],
        turn_created=0,
    )
    summary_content = progressive.get_content_for_turn(999)   # aged -> most compressed level
    full_content = progressive.full_content
    expect = spec["expect_substring"]

    summary_tokens = _safe_tokens([ToolMessage(content=summary_content, tool_call_id="x")])
    expansion_tokens = _safe_tokens([ToolMessage(content=full_content, tool_call_id="x")])
    raw_tokens = _safe_tokens([original])

    return {
        "applicable": True,
        "tool_call_id": spec["tool_call_id"],
        "source_id": spec["source_id"],
        "summary_chars": len(summary_content),
        "expansion_chars": len(full_content),
        "raw_chars": len(original.content),
        "summary_est_tokens": summary_tokens,
        "expansion_est_tokens": expansion_tokens,
        "summary_plus_expansion_est_tokens": summary_tokens + expansion_tokens,
        "raw_equivalent_est_tokens": raw_tokens,
        "expansion_returns_exact_region": expect in full_content,
        "checkpoint_original_intact": expect in original.content,
        "pass": (expect in full_content) and (expect in original.content),
    }


def evaluate_cachepoint_stability(fixture: Fixture) -> dict[str, Any]:
    """Two identical production runs must produce identical cachepoint placement
    and an identical stable prefix."""
    run1 = mode_b_production(copy.deepcopy(fixture.messages))
    run2 = mode_b_production(copy.deepcopy(fixture.messages))

    def positions(msgs: Sequence[BaseMessage]) -> list[int]:
        pos = []
        for i, m in enumerate(msgs):
            c = getattr(m, "content", None)
            if isinstance(c, list) and any(
                isinstance(b, Mapping) and "cachePoint" in b for b in c
            ):
                pos.append(i)
        return pos

    p1, p2 = positions(run1), positions(run2)
    first = p1[0] if p1 else 0
    prefix1 = serialize_context(run1[: first + 1])
    prefix2 = serialize_context(run2[: first + 1])
    h1 = hashlib.sha256(prefix1.encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(prefix2.encode("utf-8")).hexdigest()

    full1 = hashlib.sha256(serialize_context(run1).encode("utf-8")).hexdigest()
    full2 = hashlib.sha256(serialize_context(run2).encode("utf-8")).hexdigest()

    text = serialize_context(run1)
    canary_leaks = {k: (v in text) for k, v in LEAK_CANARIES.items()}

    return {
        "cachepoint_positions_run1": p1,
        "cachepoint_positions_run2": p2,
        "positions_identical": p1 == p2,
        "stable_prefix_bytes": len(prefix1.encode("utf-8")),
        "stable_prefix_est_tokens": _safe_tokens(run1[: first + 1]),
        "stable_prefix_sha256_run1": h1,
        "stable_prefix_sha256_run2": h2,
        "prefix_identical": h1 == h2,
        "full_context_identical": full1 == full2,
        "canary_leaks": canary_leaks,
        "pass": p1 == p2 and h1 == h2 and full1 == full2 and not any(canary_leaks.values()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_case(builder: Callable[[], Fixture]) -> dict[str, Any]:
    fixture = builder()
    policy = resolve_policy(fixture.stored_policy, fixture.prompt)

    modes: dict[str, Any] = {}
    baseline_metrics: dict[str, Any] | None = None
    baseline_text = ""

    for mode_name, fn in MODES.items():
        msgs = copy.deepcopy(fixture.messages)
        t0 = time.perf_counter()
        visible = fn(msgs)
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        m = measure(visible)
        text = serialize_context(visible)
        if mode_name == "A_raw":
            baseline_metrics = m
            baseline_text = text

        integ = check_tool_message_integrity(visible)
        cont = evaluate_questions(
            fixture, visible, policy,
            policy_questions_only=(mode_name == "E_checkpoint_compressed"),
        )
        ev = evaluate_evidence_integrity(fixture, visible)

        causal = count_phrases(text, CAUSAL_CLAIM_PHRASES)
        fneg = count_phrases(text, FALSE_NEGATIVE_PHRASES)
        base_causal = count_phrases(baseline_text, CAUSAL_CLAIM_PHRASES)
        base_fneg = count_phrases(baseline_text, FALSE_NEGATIVE_PHRASES)

        modes[mode_name] = {
            "metrics": m,
            "compression_duration_ms": duration_ms,
            "reduction_vs_raw": {
                "est_tokens_pct": pct_reduction(baseline_metrics["est_tokens"], m["est_tokens"]) if baseline_metrics else 0.0,
                "bytes_pct": pct_reduction(baseline_metrics["bytes"], m["bytes"]) if baseline_metrics else 0.0,
                "chars_pct": pct_reduction(baseline_metrics["chars"], m["chars"]) if baseline_metrics else 0.0,
                "tool_est_tokens_pct": pct_reduction(baseline_metrics["tool_est_tokens"], m["tool_est_tokens"]) if baseline_metrics else 0.0,
                "non_tool_est_tokens_pct": pct_reduction(baseline_metrics["non_tool_est_tokens"], m["non_tool_est_tokens"]) if baseline_metrics else 0.0,
            },
            "tool_message_integrity": integ,
            "continuity": cont,
            "evidence_integrity": ev,
            "unsupported_causal_claims": causal,
            "unsupported_causal_claims_delta": max(
                0, sum(causal.values()) - sum(base_causal.values())
            ),
            "false_negative_phrases": fneg,
            "false_negative_delta": max(0, sum(fneg.values()) - sum(base_fneg.values())),
        }

    ckpt_json = json.dumps(policy["stored"], sort_keys=True)
    t0 = time.perf_counter()
    restored = load_tool_policy_state(json.loads(ckpt_json))
    restore_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return {
        "name": fixture.name,
        "description": fixture.description,
        "large": fixture.large,
        "notes": fixture.notes,
        "raw_message_count": len(fixture.messages),
        "current_turn_prompt": fixture.prompt,
        "policy": {
            "stored_authorization_flags": policy["stored"]["authorization_flags"],
            "current_turn_delta": policy["delta"],
            "resolved_flags": policy["flags"],
            "requested_extra_tools": policy["stored"]["requested_extra_tools"],
            "pending_authorization_extra_tools": policy["stored"]["pending_authorization_extra_tools"],
            "unavailable_extra_tools": policy["stored"]["unavailable_extra_tools"],
            "active_toolsets": policy["stored"]["active_toolsets"],
            "checkpoint_serialized_bytes": len(ckpt_json.encode("utf-8")),
            "checkpoint_restore_ms": restore_ms,
            "restore_lossless": restored["requested_extra_tools"] == policy["stored"]["requested_extra_tools"],
        },
        "modes": modes,
        "targeted_expansion": evaluate_targeted_expansion(fixture),
        "cachepoint_stability": evaluate_cachepoint_stability(fixture),
    }


def token_measurement_method() -> dict[str, Any]:
    import importlib
    info: dict[str, Any] = {
        "counter": "app.message.compression.count_message_tokens",
        "tokenizer": "tiktoken/cl100k_base",
        "token_measurement": "estimated",
        "classification": "PROXY estimate, not exact Claude/Bedrock tokens",
        "rationale": "Production uses this same offline counter, so benchmark counts match "
                     "what the runtime itself budgets against.",
        "fallback_formula": "len(serialized_text) // 4 when the counter raises",
    }
    try:
        info["tiktoken_version"] = importlib.import_module("tiktoken").__version__
    except Exception:
        info["tiktoken_version"] = "unavailable"
    authoritative = False
    notes = []
    try:
        import anthropic
        info["anthropic_sdk_version"] = getattr(anthropic, "__version__", "unknown")
        client_has_legacy = hasattr(anthropic.Anthropic, "count_tokens")
        notes.append(f"anthropic.Anthropic.count_tokens present={client_has_legacy}")
        notes.append("anthropic beta messages count_tokens is a REMOTE API endpoint requiring "
                     "credentials; it is not a local authoritative tokenizer and was not called")
    except Exception as exc:
        notes.append(f"anthropic SDK probe failed: {type(exc).__name__}")
    try:
        import importlib as _il
        _il.import_module("langchain_aws")
        notes.append("langchain_aws present; exposes no local Claude tokenizer")
    except Exception:
        notes.append("langchain_aws not importable")
    info["authoritative_local_tokenizer_available"] = authoritative
    info["model_mapping_limitations"] = notes
    return info


def build_results() -> dict[str, Any]:
    started = time.time()
    cases = [run_case(b) for b in ALL_FIXTURES]

    large = [c for c in cases if c["large"]]
    large_meeting = [
        c for c in large
        if c["modes"]["B_production"]["reduction_vs_raw"]["est_tokens_pct"] >= 30.0
    ]
    agg_base = sum(c["modes"]["A_raw"]["metrics"]["est_tokens"] for c in large)
    agg_new = sum(c["modes"]["B_production"]["metrics"]["est_tokens"] for c in large)

    def all_pass(path: Callable[[dict[str, Any]], bool]) -> bool:
        return all(path(c) for c in cases)

    return {
        "schema": SCHEMA_VERSION,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "duration_s": round(time.time() - started, 2),
        "token_measurement": token_measurement_method(),
        "provider": {
            "pllm_provider": _settings().PLLM_PROVIDER,
            "bedrock_shape_required": _bedrock_shape_required(),
            "max_cachepoints": _settings().MAX_CACHEPOINT_CNT,
            "max_messages": AR.MAX_MESSAGES,
            "history_budget_tokens": AR._active_history_budget_tokens(),
            "tool_compress_token_threshold": AR.TOOL_MESSAGE_COMPRESS_TOKEN_THRESHOLD,
            "tiered_target_tokens": AR.TIERED_COMPRESSION_TARGET_TOKENS,
        },
        "modes": list(MODES),
        "cases": cases,
        "authorization_invariants": run_authorization_invariants(),
        "provider_repair": run_provider_repair_scenarios(),
        "aggregates": {
            "large_cases": len(large),
            "large_cases_meeting_30_percent": len(large_meeting),
            "aggregate_large_case_reduction_pct": pct_reduction(agg_base, agg_new),
            "tool_message_pairing_pass": all_pass(
                lambda c: all(m["tool_message_integrity"]["pass"] for m in c["modes"].values())
            ),
            "continuity_pass": all_pass(
                lambda c: all(m["continuity"]["pass"] for m in c["modes"].values())
            ),
            "evidence_integrity_pass": all_pass(
                lambda c: all(
                    (not m["evidence_integrity"].get("applicable")) or m["evidence_integrity"]["pass"]
                    for m in c["modes"].values()
                )
            ),
            "targeted_expansion_pass": all_pass(
                lambda c: (not c["targeted_expansion"].get("applicable")) or c["targeted_expansion"]["pass"]
            ),
            "cachepoint_stability_pass": all_pass(lambda c: c["cachepoint_stability"]["pass"]),
            "unsupported_causal_claims_delta": sum(
                m["unsupported_causal_claims_delta"] for c in cases for m in c["modes"].values()
            ),
            "false_negative_delta": sum(
                m["false_negative_delta"] for c in cases for m in c["modes"].values()
            ),
        },
    }


def render_markdown(results: Mapping[str, Any]) -> str:
    L: list[str] = []
    L.append("# D3B3 context compression benchmark")
    L.append("")
    L.append(f"Generated: {results['generated_utc']}  |  duration {results['duration_s']}s")
    tm = results["token_measurement"]
    L.append(f"Tokenizer: `{tm['tokenizer']}` ({tm['token_measurement']}, {tm['classification']})")
    L.append("")
    L.append("## Runtime parameters")
    for k, v in results["provider"].items():
        L.append(f"- `{k}` = `{v}`")
    L.append("")
    L.append("## Per-case model-visible input (Mode B production vs Mode A raw)")
    L.append("")
    L.append("| case | large | raw msgs | raw tokens | prod msgs | prod tokens | token reduction | tool reduction |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in results["cases"]:
        a = c["modes"]["A_raw"]["metrics"]
        b = c["modes"]["B_production"]
        L.append(
            f"| {c['name']} | {c['large']} | {a['messages']} | {a['est_tokens']:,} | "
            f"{b['metrics']['messages']} | {b['metrics']['est_tokens']:,} | "
            f"{b['reduction_vs_raw']['est_tokens_pct']}% | "
            f"{b['reduction_vs_raw']['tool_est_tokens_pct']}% |"
        )
    L.append("")
    L.append("## All modes, estimated model-visible tokens")
    L.append("")
    header = "| case | " + " | ".join(results["modes"]) + " |"
    L.append(header)
    L.append("|---" * (len(results["modes"]) + 1) + "|")
    for c in results["cases"]:
        row = [c["name"]]
        for mode in results["modes"]:
            mm = c["modes"][mode]
            row.append(f"{mm['metrics']['est_tokens']:,} ({mm['reduction_vs_raw']['est_tokens_pct']}%)")
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    agg = results["aggregates"]
    L.append("## Aggregates")
    for k, v in agg.items():
        L.append(f"- `{k}` = `{v}`")
    L.append("")
    L.append("## Authorization invariants")
    for k, v in results["authorization_invariants"].items():
        if isinstance(v, dict):
            L.append(f"- `{k}` -> pass=`{v.get('pass')}`")
    L.append("")
    L.append("## Provider message repair")
    for k, v in results["provider_repair"]["scenarios"].items():
        L.append(f"- `{k}`: provider_valid=`{v['provider_valid']}` "
                 f"fabricated_success=`{v['fabricated_success_result']}`")
    L.append("")
    L.append("## Targeted expansion")
    for c in results["cases"]:
        te = c["targeted_expansion"]
        if te.get("applicable"):
            L.append(f"- `{c['name']}`: summary {te['summary_est_tokens']:,} tok, "
                     f"expansion {te['expansion_est_tokens']:,} tok, "
                     f"raw {te['raw_equivalent_est_tokens']:,} tok, "
                     f"exact_region=`{te['expansion_returns_exact_region']}`")
    L.append("")
    return "\n".join(L)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D3B3 context compression benchmark")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--md-out", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    results = build_results()

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md_out).write_text(render_markdown(results), encoding="utf-8")
    if not args.quiet:
        agg = results["aggregates"]
        print(f"cases={len(results['cases'])} modes={len(results['modes'])}")
        for c in results["cases"]:
            b = c["modes"]["B_production"]
            print(f"  {c['name']:34s} large={str(c['large']):5s} "
                  f"raw={c['modes']['A_raw']['metrics']['est_tokens']:>8,} "
                  f"prod={b['metrics']['est_tokens']:>8,} "
                  f"red={b['reduction_vs_raw']['est_tokens_pct']:>6}%")
        print(f"large_meeting_30pct={agg['large_cases_meeting_30_percent']}/{agg['large_cases']} "
              f"aggregate={agg['aggregate_large_case_reduction_pct']}%")
        print(f"pairing={agg['tool_message_pairing_pass']} continuity={agg['continuity_pass']} "
              f"evidence={agg['evidence_integrity_pass']} expansion={agg['targeted_expansion_pass']} "
              f"cachepoint={agg['cachepoint_stability_pass']}")
        print(f"auth_invariants={results['authorization_invariants']['pass']} "
              f"repair={results['provider_repair']['pass']}")
        print(f"causal_delta={agg['unsupported_causal_claims_delta']} "
              f"false_negative_delta={agg['false_negative_delta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
