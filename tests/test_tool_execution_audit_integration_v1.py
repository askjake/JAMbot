"""Integration tests for request-correlated tool-execution audit evidence.

Evidence in these tests is the audit event and the paired ToolMessage.  No test
relies on a model claiming that a tool ran, and no heavy or persistent
server-side build is invoked.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.audited_tool_node import make_audited_tool_node
from app.agent.tool_execution_audit import (
    ToolExecutionAuditSink,
    current_audit_request_state,
    query_events,
    record_model_facing_binding,
    validate_event,
)
from app.middlewares import RequestCorrelationMiddleware

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "scripts/active/query_tool_execution_audit.py"

CANARY_ARG = "R7654321"
CANARY_RESULT = "CANARY-INTEGRATION-RESULT-9de2"


class FakeTool:
    def __init__(self, name: str, *, result: str = "ok"):
        self.name = name
        self._result = result

    async def ainvoke(self, tool_call, config=None):
        return ToolMessage(content=self._result, tool_call_id=tool_call.get("id") or "", name=self.name)


def _ai(tool_name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": dict(args), "id": call_id, "type": "tool_call"}],
    )


@pytest.fixture()
def sink(tmp_path: Path) -> ToolExecutionAuditSink:
    return ToolExecutionAuditSink(tmp_path / "audit", max_bytes=200_000, max_files=3)


def _build_app(sink: ToolExecutionAuditSink) -> FastAPI:
    """Minimal app carrying the real correlation middleware and audited node."""
    app = FastAPI()
    app.add_middleware(RequestCorrelationMiddleware)

    node = make_audited_tool_node(
        [FakeTool("list_dates", result=CANARY_RESULT), FakeTool("build_incident_scene")],
        sink=sink,
    )

    @app.post("/no-tool")
    async def no_tool():
        state = current_audit_request_state()
        return {"request_id": state.request_id if state else "", "tool_calls": 0}

    @app.post("/run")
    async def run(payload: dict):
        record_model_facing_binding(
            bound_tool_names=payload.get("bound") or [],
            authorization_flags=payload.get("flags") or {},
            thread_id=payload.get("thread_id") or "",
        )
        state = current_audit_request_state()
        result = await node(
            {"messages": [_ai(payload["tool"], payload.get("args") or {}, payload["call_id"])]}
        )
        return {
            "request_id": state.request_id if state else "",
            "binding_signature": state.binding_signature if state else "",
            "messages": [
                {"tool_call_id": m.tool_call_id, "content": str(m.content)} for m in result["messages"]
            ],
        }

    return app


# --------------------------------------------------------------------------
# HTTP request correlation
# --------------------------------------------------------------------------
def test_http_request_id_is_server_owned_and_returned(sink):
    with TestClient(_build_app(sink)) as client:
        response = client.post("/no-tool")
    assert response.status_code == 200
    header_id = response.headers["X-Request-ID"]
    assert header_id.startswith("req-")
    assert response.json()["request_id"] == header_id


def test_generic_no_tool_request_emits_no_tool_audit_event(sink):
    with TestClient(_build_app(sink)) as client:
        response = client.post("/no-tool")
    assert response.status_code == 200
    assert query_events(sink=sink, request_id=response.headers["X-Request-ID"], limit=50) == []


def test_supplied_request_id_header_is_accepted_when_safe(sink):
    with TestClient(_build_app(sink)) as client:
        good = client.post("/no-tool", headers={"X-Request-ID": "ops-correlation-0001"})
        bad = client.post("/no-tool", headers={"X-Request-ID": "bad id with spaces!!"})
    assert good.headers["X-Request-ID"] == "ops-correlation-0001"
    assert bad.headers["X-Request-ID"].startswith("req-")


def test_allowed_execution_correlates_http_binding_gate_and_tool_message(sink):
    with TestClient(_build_app(sink)) as client:
        response = client.post(
            "/run",
            json={
                "tool": "list_dates",
                "args": {"receiver_id": CANARY_ARG},
                "call_id": "c-allowed",
                "bound": ["list_dates"],
                "thread_id": "integration-thread-1",
            },
        )
    body = response.json()
    request_id = response.headers["X-Request-ID"]
    assert body["request_id"] == request_id
    assert body["messages"][0]["content"] == CANARY_RESULT

    events = query_events(sink=sink, request_id=request_id, limit=10)
    assert len(events) == 1
    event = events[0]
    assert validate_event(event) == []
    assert event["executed"] is True
    assert event["paired_tool_result"] is True
    assert event["tool_call_id"] == body["messages"][0]["tool_call_id"] == "c-allowed"
    assert event["binding_signature"] == body["binding_signature"]
    assert event["decision"] == "ALLOWED"
    assert CANARY_ARG not in json.dumps(event)


def test_unbound_call_is_blocked_and_audited_over_http(sink):
    with TestClient(_build_app(sink)) as client:
        response = client.post(
            "/run",
            json={"tool": "build_incident_scene", "args": {}, "call_id": "c-unbound", "bound": ["list_dates"]},
        )
    request_id = response.headers["X-Request-ID"]
    events = query_events(sink=sink, request_id=request_id, limit=10)
    assert len(events) == 1
    assert events[0]["result_code"] == "BLOCKED_UNBOUND_TOOL"
    assert events[0]["executed"] is False
    assert response.json()["messages"][0]["tool_call_id"] == "c-unbound"


def test_unauthorized_heavy_call_is_blocked_over_http(sink):
    with TestClient(_build_app(sink)) as client:
        response = client.post(
            "/run",
            json={
                "tool": "build_incident_scene",
                "args": {"capsule_id": "cap"},
                "call_id": "c-heavy",
                "bound": ["build_incident_scene"],
                "flags": {"heavy_tools_authorized": False},
            },
        )
    events = query_events(sink=sink, request_id=response.headers["X-Request-ID"], limit=10)
    assert events[0]["result_code"] == "BLOCKED_HEAVY_UNAUTHORIZED"
    assert events[0]["executed"] is False
    assert events[0]["authorization_flags"]["heavy_tools_authorized"] is False


def test_concurrent_requests_do_not_share_correlation_state(sink):
    app = _build_app(sink)
    with TestClient(app) as client:
        def call(index: int):
            return client.post(
                "/run",
                json={
                    "tool": "list_dates",
                    "args": {"receiver_id": f"R12345{index:02d}"},
                    "call_id": f"c-{index}",
                    "bound": ["list_dates"],
                    "thread_id": f"thread-{index}",
                },
            )

        with ThreadPoolExecutor(max_workers=6) as pool:
            responses = list(pool.map(call, range(6)))

    request_ids = [response.headers["X-Request-ID"] for response in responses]
    assert len(set(request_ids)) == 6
    digests = set()
    for index, request_id in enumerate(request_ids):
        events = query_events(sink=sink, request_id=request_id, limit=10)
        assert len(events) == 1, f"request {request_id} produced {len(events)} events"
        assert events[0]["tool_call_id"] == f"c-{index}"
        digests.add(events[0]["thread_id_digest"])
    assert len(digests) == 6


# --------------------------------------------------------------------------
# Operator CLI
# --------------------------------------------------------------------------
def _cli(args: list[str], audit_dir: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["DISHCHAT_TOOL_AUDIT_DIR"] = str(audit_dir)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def _seed(audit_dir: Path) -> str:
    sink = ToolExecutionAuditSink(audit_dir, max_bytes=200_000, max_files=3)
    node = make_audited_tool_node([FakeTool("list_dates", result=CANARY_RESULT)], sink=sink)
    import asyncio

    record_model_facing_binding(bound_tool_names=["list_dates"], thread_id="cli-thread")
    state = current_audit_request_state()
    asyncio.run(node({"messages": [_ai("list_dates", {"receiver_id": CANARY_ARG}, "c-cli")]}))
    return state.request_id if state else ""


def test_cli_validate_mode_passes_on_real_records(tmp_path):
    audit_dir = tmp_path / "audit"
    _seed(audit_dir)
    result = _cli(["--validate", "--limit", "100"], audit_dir)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "PASS"
    assert report["invalid_records"] == 0
    assert report["records_checked"] >= 1


def test_cli_rejects_malformed_records(tmp_path):
    audit_dir = tmp_path / "audit"
    _seed(audit_dir)
    with (audit_dir / "tool_execution_audit.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"schema_version": "wrong.v0"}\n')
    result = _cli(["--validate", "--limit", "100"], audit_dir)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["status"] == "FAIL"
    assert report["invalid_records"] >= 1


def test_cli_output_never_contains_argument_values_or_result_bodies(tmp_path):
    audit_dir = tmp_path / "audit"
    _seed(audit_dir)
    for args in (["--limit", "50"], ["--limit", "50", "--json"], ["--storage"]):
        result = _cli(args, audit_dir)
        assert result.returncode == 0, result.stderr
        assert CANARY_ARG not in result.stdout
        assert CANARY_RESULT not in result.stdout


def test_cli_enforces_bounded_limit(tmp_path):
    audit_dir = tmp_path / "audit"
    for _ in range(3):
        _seed(audit_dir)
    result = _cli(["--limit", "1000000", "--json"], audit_dir)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["limit"] <= 500


def test_cli_filters_by_decision_and_result_code(tmp_path):
    audit_dir = tmp_path / "audit"
    _seed(audit_dir)
    allowed = _cli(["--decision", "ALLOWED", "--json"], audit_dir)
    assert allowed.returncode == 0
    assert json.loads(allowed.stdout)["count"] >= 1

    blocked = _cli(["--decision", "BLOCKED", "--json"], audit_dir)
    assert json.loads(blocked.stdout)["count"] == 0

    unknown = _cli(["--result-code", "NOT_A_REAL_CODE"], audit_dir)
    assert unknown.returncode == 2


def test_cli_rotation_visibility(tmp_path):
    audit_dir = tmp_path / "audit"
    sink = ToolExecutionAuditSink(audit_dir, max_bytes=1024, max_files=2)
    from app.agent.tool_execution_audit import build_event

    for _ in range(150):
        sink.record(build_event(tool_name="list_dates"))
    assert len(list(audit_dir.glob("tool_execution_audit.*.jsonl"))) <= 2
    result = _cli(["--limit", "20", "--json"], audit_dir)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["count"] <= 20
