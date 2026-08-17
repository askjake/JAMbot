from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "2026-07-09"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class ToolEvidencePacket:
    packet_type: str = "tool_evidence"
    task_id: str = ""
    worker_role: str = "tool_worker"
    status: str = "partial"
    tool_families_used: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    raw_artifacts: list[str] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    inferences: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    next_recommended_step: str = ""
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now)


@dataclass
class AnalysisPacket:
    packet_type: str = "analysis"
    source_task_ids: list[str] = field(default_factory=list)
    status: str = "partial"
    executive_summary: str = ""
    confirmed_facts: list[dict[str, Any]] = field(default_factory=list)
    likely_inferences: list[dict[str, Any]] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    recommended_next_steps: list[str] = field(default_factory=list)
    confidence: str = "medium"
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now)


@dataclass
class VerifierReport:
    verdict: str = "FAIL"
    unsupported_claims: list[str] = field(default_factory=list)
    missing_required_steps: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    tool_misuse: list[str] = field(default_factory=list)
    verification_gaps: list[str] = field(default_factory=list)
    recommended_fix: str = ""
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(payload) if hasattr(payload, "__dataclass_fields__") else payload
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def init_run_state(root: Path, *, run_id: str | None = None, chat_id: str = "", goal: str = "", methodology: str = "generic_engineering") -> Path:
    run_id = run_id or new_run_id()
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "RUN_STATE.json", {
        "run_id": run_id,
        "chat_id": chat_id,
        "goal": goal,
        "methodology": methodology,
        "current_phase": "initialized",
        "completed_phases": [],
        "open_questions": [],
        "active_tasks": [],
        "completed_tasks": [],
        "tools_called": [],
        "files_touched": [],
        "commands_run": [],
        "verification_status": "not_started",
        "final_classification": "",
        "updated_at": utc_now(),
    })
    write_json(run_dir / "TASKS.json", [])
    (run_dir / "EVIDENCE_LEDGER.jsonl").write_text("")
    write_json(run_dir / "VERIFIER_REPORT.json", VerifierReport())
    (run_dir / "FINAL_RESPONSE.md").write_text("")
    return run_dir


def update_run_state(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = Path(run_dir) / "RUN_STATE.json"
    state = read_json(path)
    state.update(updates)
    state["updated_at"] = utc_now()
    write_json(path, state)
    return state


def append_evidence(run_dir: Path, entry: dict[str, Any]) -> Path:
    path = Path(run_dir) / "EVIDENCE_LEDGER.jsonl"
    row = dict(entry)
    row.setdefault("created_at", utc_now())
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def write_tool_evidence_packet(run_dir: Path, packet: ToolEvidencePacket | dict[str, Any]) -> Path:
    data = asdict(packet) if isinstance(packet, ToolEvidencePacket) else dict(packet)
    task_id = data.get("task_id") or new_run_id("task")
    data.setdefault("packet_type", "tool_evidence")
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("created_at", utc_now())
    path = Path(run_dir) / "packets" / f"{task_id}.tool_evidence.json"
    write_json(path, data)
    append_evidence(Path(run_dir), {"entry_type": "packet", "task_id": task_id, "packet_path": str(path), "status": data.get("status")})
    return path


def write_analysis_packet(run_dir: Path, packet: AnalysisPacket | dict[str, Any], *, name: str = "analysis") -> Path:
    data = asdict(packet) if isinstance(packet, AnalysisPacket) else dict(packet)
    data.setdefault("packet_type", "analysis")
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("created_at", utc_now())
    path = Path(run_dir) / "packets" / f"{name}.analysis.json"
    write_json(path, data)
    append_evidence(Path(run_dir), {"entry_type": "analysis", "packet_path": str(path), "status": data.get("status")})
    return path


def read_packet(path: str | Path) -> dict[str, Any]:
    return read_json(Path(path))


def load_packets(run_dir: Path, packet_type: str | None = None) -> list[dict[str, Any]]:
    packets = []
    for path in sorted((Path(run_dir) / "packets").glob("*.json")):
        data = read_json(path)
        if packet_type is None or data.get("packet_type") == packet_type:
            data["_path"] = str(path)
            packets.append(data)
    return packets


def write_verifier_report(run_dir: Path, report: VerifierReport | dict[str, Any]) -> Path:
    data = asdict(report) if isinstance(report, VerifierReport) else dict(report)
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("created_at", utc_now())
    path = Path(run_dir) / "VERIFIER_REPORT.json"
    write_json(path, data)
    update_run_state(Path(run_dir), verification_status=data.get("verdict", "FAIL"))
    return path


def normalize_worker_packet_from_summary(*, task_id: str, summary: str, status: str = "partial", artifacts: Iterable[str] = (), error: str | None = None) -> ToolEvidencePacket:
    gaps = [] if summary.strip() else ["Child conversation produced no final summary."]
    facts = []
    if summary.strip():
        facts.append({"claim": "Child returned prose summary; packet was normalized by fallback.", "source": "child_final_message", "reference": task_id, "confidence": "high"})
        gaps.append("Summary requires analyst normalization before master synthesis.")
    errors = [error] if error else []
    return ToolEvidencePacket(task_id=task_id, status=status, raw_artifacts=list(artifacts), facts=facts, gaps=gaps, errors=errors, next_recommended_step="Review packet gaps before synthesis.")


def try_parse_tool_evidence_packet(text: str, task_id: str) -> ToolEvidencePacket | None:
    candidates = [text]
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if match:
        candidates.insert(0, match.group(1))
    for raw in candidates:
        try:
            data = json.loads(raw)
        except Exception:
            continue
        if data.get("packet_type") == "tool_evidence":
            data.setdefault("task_id", task_id)
            allowed = ToolEvidencePacket.__dataclass_fields__
            return ToolEvidencePacket(**{k: v for k, v in data.items() if k in allowed})
    return None


def analysis_from_tool_packets(packets: Iterable[dict[str, Any]]) -> AnalysisPacket:
    packet_list = list(packets)
    facts: list[dict[str, Any]] = []
    inferences: list[dict[str, Any]] = []
    gaps: list[str] = []
    errors: list[str] = []
    source_ids: list[str] = []
    for packet in packet_list:
        source_ids.append(packet.get("task_id", ""))
        facts.extend(packet.get("facts", []))
        inferences.extend(packet.get("inferences", []))
        gaps.extend(packet.get("gaps", []))
        errors.extend(packet.get("errors", []))
    status = "failed" if errors else ("partial" if gaps else "complete")
    return AnalysisPacket(source_task_ids=[s for s in source_ids if s], status=status, executive_summary=f"Normalized {len(packet_list)} tool evidence packet(s).", confirmed_facts=facts, likely_inferences=inferences, missing_evidence=gaps, recommended_next_steps=["Address gaps before final synthesis"] if gaps or errors else [], confidence="low" if errors else ("medium" if gaps else "high"))


def verify_final_answer_against_packets(final_answer: str, packets: Iterable[dict[str, Any]], acceptance_criteria: Iterable[str] = ()) -> VerifierReport:
    text = (final_answer or "").lower()
    evidence_text = json.dumps(list(packets), sort_keys=True).lower()
    unsupported = [m for m in ("verified all", "fully proven", "guaranteed", "no remaining risks") if m in text and m not in evidence_text]
    missing = [str(c) for c in acceptance_criteria if str(c).strip() and str(c).lower() not in text and str(c).lower() not in evidence_text]
    gaps = [] if evidence_text and evidence_text != "[]" else ["No evidence packets supplied to verifier."]
    verdict = "FAIL" if unsupported or missing else ("PASS_WITH_RISKS" if gaps else "PASS")
    return VerifierReport(verdict=verdict, unsupported_claims=unsupported, missing_required_steps=missing, verification_gaps=gaps, recommended_fix="Remove unsupported claims or gather missing packet evidence." if verdict == "FAIL" else "")


def orchestration_packet_selftest(root: Path) -> dict[str, Any]:
    run_dir = init_run_state(root, chat_id="selftest", goal="packet selftest")
    packet_path = write_tool_evidence_packet(run_dir, ToolEvidencePacket(task_id="selftest_worker", status="complete", facts=[{"claim": "packet write/read works", "source": "selftest", "reference": str(run_dir), "confidence": "high"}]))
    packet = read_packet(packet_path)
    analysis_path = write_analysis_packet(run_dir, analysis_from_tool_packets([packet]))
    report = verify_final_answer_against_packets("packet write/read works", [packet])
    write_verifier_report(run_dir, report)
    return {"status": "pass" if packet_path.exists() and analysis_path.exists() else "fail", "run_dir": str(run_dir), "packet_path": str(packet_path), "analysis_path": str(analysis_path), "verdict": report.verdict}
