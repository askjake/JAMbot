from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.agent.methodology import select_methodology
from app.agent_mode.orchestration_packets import (
    ToolEvidencePacket,
    analysis_from_tool_packets,
    init_run_state,
    load_packets,
    read_packet,
    update_run_state,
    verify_final_answer_against_packets,
    write_analysis_packet,
    write_tool_evidence_packet,
    write_verifier_report,
)

ROLE_MODEL_MAPPING = {"master": "complex", "planner": "complex", "tool_worker": "tool_worker", "analyst": "analyst", "verifier": "verifier", "final_synthesizer": "complex"}


@dataclass
class ExecutiveMaster:
    run_root: Path
    chat_id: str
    goal: str

    def initialize(self) -> Path:
        methodology = select_methodology(self.goal)
        run_dir = init_run_state(self.run_root, chat_id=self.chat_id, goal=self.goal, methodology=methodology["name"])
        update_run_state(run_dir, current_phase="planning")
        return run_dir

    def create_phase_plan(self, run_dir: Path) -> list[dict[str, Any]]:
        plan = [
            {"phase": "worker_evidence", "role": ROLE_MODEL_MAPPING["tool_worker"]},
            {"phase": "analysis_normalization", "role": ROLE_MODEL_MAPPING["analyst"]},
            {"phase": "verification", "role": ROLE_MODEL_MAPPING["verifier"]},
            {"phase": "final_synthesis", "role": ROLE_MODEL_MAPPING["final_synthesizer"]},
        ]
        update_run_state(run_dir, current_phase="worker_evidence", active_tasks=plan)
        return plan

    def read_compact_packets(self, run_dir: Path) -> list[dict[str, Any]]:
        return load_packets(run_dir)


@dataclass
class ToolWorker:
    run_dir: Path
    task_id: str
    tool_families: list[str]

    def emit_packet(self, *, facts=None, inferences=None, gaps=None, errors=None, raw_artifacts=None, status="complete", next_recommended_step="") -> Path:
        packet = ToolEvidencePacket(task_id=self.task_id, status=status, tool_families_used=self.tool_families, raw_artifacts=raw_artifacts or [], facts=facts or [], inferences=inferences or [], gaps=gaps or [], errors=errors or [], next_recommended_step=next_recommended_step)
        return write_tool_evidence_packet(self.run_dir, packet)


@dataclass
class AnalystNormalizer:
    run_dir: Path

    def normalize(self, packet_paths: Iterable[str | Path] | None = None) -> Path:
        packets = [read_packet(p) for p in packet_paths] if packet_paths else load_packets(self.run_dir, packet_type="tool_evidence")
        return write_analysis_packet(self.run_dir, analysis_from_tool_packets(packets))


@dataclass
class Verifier:
    run_dir: Path

    def audit(self, final_draft: str, acceptance_criteria: Iterable[str] = ()) -> Path:
        report = verify_final_answer_against_packets(final_draft, load_packets(self.run_dir), acceptance_criteria)
        return write_verifier_report(self.run_dir, report)


@dataclass
class FinalSynthesizer:
    run_dir: Path

    def write_final(self, content: str) -> Path:
        path = Path(self.run_dir) / "FINAL_RESPONSE.md"
        path.write_text(content)
        return path


def orchestration_smoke_test(root: Path) -> dict[str, Any]:
    master = ExecutiveMaster(root, chat_id="smoke", goal="patch FastAPI backend and run tests")
    run_dir = master.initialize()
    master.create_phase_plan(run_dir)
    worker = ToolWorker(run_dir, task_id="worker_1", tool_families=["agent_mode"])
    packet_path = worker.emit_packet(facts=[{"claim": "worker emitted compact evidence", "source": "smoke", "reference": str(run_dir), "confidence": "high"}], next_recommended_step="normalize packets")
    analysis_path = AnalystNormalizer(run_dir).normalize([packet_path])
    final_path = FinalSynthesizer(run_dir).write_final("worker emitted compact evidence")
    verifier_path = Verifier(run_dir).audit("worker emitted compact evidence")
    ok = all(p.exists() for p in (packet_path, analysis_path, final_path, verifier_path))
    return {"status": "pass" if ok else "fail", "run_dir": str(run_dir), "role_model_mapping": ROLE_MODEL_MAPPING, "packet_path": str(packet_path), "analysis_path": str(analysis_path), "verifier_path": str(verifier_path), "final_path": str(final_path)}
