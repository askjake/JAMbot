from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import types
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest


SOURCE_ROOT = Path(os.environ["JAKE_SOURCE_ROOT"])


@dataclass
class FakeChildResult:
    task_id: str
    status: str
    artifacts: list[str] = field(default_factory=list)
    summary: str = ""
    tokens_used: int = 0
    iterations_used: int = 0
    error: str | None = None
    facts: list[dict] = field(default_factory=list)
    inferences: list[dict] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    raw_artifacts: list[str] = field(default_factory=list)
    next_recommended_step: str = ""
    packet_path: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_json(self):
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str):
        return cls(**json.loads(data))


@dataclass
class FakeState:
    parent_chat_id: str
    tasks: dict[str, FakeChildResult] = field(default_factory=dict)
    total_tokens: int = 0

    current = None

    @classmethod
    def load(cls, _chat_id: str):
        assert cls.current is not None
        return cls.current


def _install_stub(name: str, module: types.ModuleType):
    sys.modules[name] = module


def load_mcop(tmp_path: Path):
    # Stub the imports used by mcop_tools so these tests exercise only the
    # orchestration boundary code under test.
    langchain = types.ModuleType("langchain")
    langchain_tools = types.ModuleType("langchain.tools")

    def tool(_name):
        def decorator(fn):
            return fn
        return decorator

    langchain_tools.tool = tool
    _install_stub("langchain", langchain)
    _install_stub("langchain.tools", langchain_tools)

    app = types.ModuleType("app")
    agent = types.ModuleType("app.agent")
    agent_mode = types.ModuleType("app.agent_mode")
    _install_stub("app", app)
    _install_stub("app.agent", agent)
    _install_stub("app.agent_mode", agent_mode)

    interceptor_mod = types.ModuleType("app.agent_mode.thought_interceptor")
    interceptor_mod.interceptor = types.SimpleNamespace(thought=lambda *a, **k: None)
    _install_stub("app.agent_mode.thought_interceptor", interceptor_mod)

    audit_mod = types.ModuleType("app.agent.tool_execution_audit")
    audit_mod.current_audit_request_state = lambda: None
    _install_stub("app.agent.tool_execution_audit", audit_mod)

    cap_mod = types.ModuleType("app.agent_mode.task_capability_plan")

    class FakePlan:
        @classmethod
        def from_values(cls, **kwargs):
            return kwargs

    cap_mod.TaskCapabilityPlan = FakePlan
    _install_stub("app.agent_mode.task_capability_plan", cap_mod)

    child_mod = types.ModuleType("app.agent_mode.child_conversation")

    def get_mcop_dir(chat_id: str):
        path = tmp_path / str(chat_id) / "_mcop"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_task_dir(chat_id: str, task_id: str):
        path = get_mcop_dir(chat_id) / f"task_{task_id}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def run_child_conversation(**kwargs):
        return FakeChildResult(
            task_id=kwargs["task_id"],
            status="completed",
            summary="child completed",
            facts=[{"fact": "ok"}],
            packet_path=f"_mcop/task_{kwargs['task_id']}/tool_evidence_packet.json",
        )

    async def run_parallel_tasks(parent_chat_id: str, tasks: list[dict]):
        return [
            FakeChildResult(
                task_id=t["task_id"],
                status="completed",
                summary="S" * 20000,
                facts=[{"payload": "F" * 20000}],
                packet_path=f"_mcop/task_{t['task_id']}/tool_evidence_packet.json",
            )
            for t in tasks
        ]

    child_mod.run_child_conversation = run_child_conversation
    child_mod.run_parallel_tasks = run_parallel_tasks
    child_mod.OrchestrationState = FakeState
    child_mod.ChildResult = FakeChildResult
    child_mod._get_mcop_dir = get_mcop_dir
    child_mod._get_task_dir = get_task_dir
    child_mod.MCOP_MAX_CHILDREN = 10
    _install_stub("app.agent_mode.child_conversation", child_mod)

    source = SOURCE_ROOT / "app/agent_mode/mcop_tools.py"
    module_name = f"mcop_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, get_mcop_dir


def load_no_progress():
    source = SOURCE_ROOT / "app/agent/no_progress_controller.py"
    module_name = f"no_progress_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def terminal(task_id: str, *, summary: str = "done"):
    return FakeChildResult(task_id=task_id, status="completed", summary=summary)


def test_completed_history_does_not_exhaust_active_child_capacity(tmp_path):
    mcop, _ = load_mcop(tmp_path)
    FakeState.current = FakeState(
        parent_chat_id="chat",
        tasks={f"old-{i}": terminal(f"old-{i}") for i in range(14)},
    )

    raw = asyncio.run(
        mcop.agent_spawn_task(
            chat_id="chat",
            task_prompt="new work",
            task_id="new-task",
        )
    )
    payload = json.loads(raw)

    assert payload.get("task_id") == "new-task", payload
    assert "Maximum child tasks reached" not in raw


def test_task_status_is_bounded_paginated_and_explicit_about_truncation(tmp_path):
    mcop, _ = load_mcop(tmp_path)
    FakeState.current = FakeState(
        parent_chat_id="chat",
        tasks={
            f"done-{i}": terminal(f"done-{i}", summary="X" * 5000)
            for i in range(30)
        },
    )

    raw = mcop.agent_check_tasks("chat")
    payload = json.loads(raw)

    assert payload["ok"] is True
    assert payload["total_tasks"] == 30
    assert payload["active_task_count"] == 0
    assert payload["terminal_task_count"] == 30
    assert payload["returned_count"] == 10
    assert payload["truncated"] is True
    assert payload["next_offset"] == 10
    assert len(raw) < 15000
    assert all(row["summary_truncated"] is True for row in payload["tasks"].values())


def test_packet_read_is_chunked_and_path_traversal_is_rejected(tmp_path):
    mcop, get_mcop_dir = load_mcop(tmp_path)
    FakeState.current = FakeState(parent_chat_id="chat")
    mcop_root = get_mcop_dir("chat")
    task_dir = mcop_root / "task_demo"
    task_dir.mkdir(parents=True, exist_ok=True)
    packet = task_dir / "tool_evidence_packet.json"
    packet.write_text("P" * 25000)

    raw = mcop.agent_read_packet("chat", task_id="demo", max_chars=1000)
    payload = json.loads(raw)
    assert payload["ok"] is True
    assert payload["returned_chars"] == 1000
    assert payload["total_chars"] == 25000
    assert payload["truncated"] is True
    assert payload["next_offset"] == 1000
    assert len(raw) < 4000

    outside = mcop_root.parent.parent / "secret.txt"
    outside.write_text("TOP-SECRET")
    escaped = mcop.agent_read_packet("chat", packet_path="../secret.txt")
    escaped_payload = json.loads(escaped)
    assert escaped_payload["ok"] is False
    assert "TOP-SECRET" not in escaped



def test_parallel_parent_aggregation_is_bounded_and_truncation_is_explicit(tmp_path):
    mcop, _ = load_mcop(tmp_path)
    FakeState.current = FakeState(parent_chat_id="chat")

    raw = asyncio.run(
        mcop.agent_spawn_parallel(
            "chat",
            json.dumps([
                {"task_id": "a", "prompt": "A"},
                {"task_id": "b", "prompt": "B"},
            ]),
        )
    )
    payload = json.loads(raw)

    assert payload["ok"] is True
    assert payload["result_count"] == 2
    assert payload["response_truncated"] is True
    assert len(raw) < 15000
    assert all(item["summary_truncated"] is True for item in payload["results"])
    assert all("facts" not in item for item in payload["results"])
    assert all(item["facts_count"] == 1 for item in payload["results"])


def test_large_task_result_is_chunked_instead_of_dumped_whole(tmp_path):
    mcop, get_mcop_dir = load_mcop(tmp_path)
    FakeState.current = FakeState(parent_chat_id="chat")
    task_dir = get_mcop_dir("chat") / "task_big"
    task_dir.mkdir(parents=True, exist_ok=True)
    big = FakeChildResult(
        task_id="big",
        status="completed",
        summary="S" * 25000,
        facts=[{"payload": "F" * 25000}],
        packet_path="_mcop/task_big/tool_evidence_packet.json",
    )
    (task_dir / "result.json").write_text(big.to_json())

    raw = mcop.agent_read_task_result("chat", "big", max_chars=1000)
    payload = json.loads(raw)

    assert payload["ok"] is True
    assert payload["response_truncated"] is True
    assert payload["summary_returned_chars"] == 1000
    assert payload["summary_next_offset"] == 1000
    assert payload["facts_count"] == 1
    assert "facts" not in payload
    assert len(raw) < 6000

def test_legacy_success_without_ok_is_neutral_not_a_failure():
    npc = load_no_progress()
    messages = [
        {
            "type": "tool",
            "name": "internal_search",
            "content": json.dumps({
                "ok": False,
                "result_code": "TOOL_EXECUTION_ERROR",
                "tool_name": "internal_search",
            }),
        },
        {
            "type": "tool",
            "name": "agent_check_tasks",
            "content": json.dumps({
                "status": "no_tasks",
                "message": "No child tasks have been spawned yet.",
            }),
        },
    ]

    decision = npc.evaluate_no_progress(
        messages,
        required_tools=("some_other_tool",),
    )

    assert decision.stop is False
    assert decision.no_progress_attempts == 1
    assert "agent_check_tasks" not in decision.missing_tools
