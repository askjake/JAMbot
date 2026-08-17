"""Pipeline orchestration, resume, checkpointing, and terminal report."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .config import Settings
from .executor import SerialExecutor, ToolClient
from .notify import send, summarize
from .phases import PhaseRunner
from .report import render_report
from .state import RunState, RunStore, utc_now

log = logging.getLogger("nightly_rca.pipeline")

PHASE_ORDER = ["0", "1", "D1", "2", "2b", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]


def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


class NightlyPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        client: ToolClient,
        state: RunState | None = None,
        store: RunStore | None = None,
    ):
        self.settings = settings
        if state is None:
            run_id = new_run_id()
            state = RunState(
                schema_version=1,
                run_id=run_id,
                mode="commit" if settings.commit else "dry_run",
                role=settings.role,
                started_at=utc_now(),
            )
        self.state = state
        self.store = store or RunStore(settings.output_dir, state.run_id)
        self.executor = SerialExecutor(client, settings, self.state, self.store)
        self.phases = PhaseRunner(settings, self.state, self.store, self.executor)

    async def run(self, *, stop_after: str | None = None) -> int:
        errors = self.settings.validate()
        if errors:
            self.state.status = "CONFIG_ERROR"
            self.state.data["configuration_errors"] = errors
            self.store.save(self.state)
            return 3

        methods: dict[str, Any] = {
            "0": self.phases.phase_0_environment,
            "1": self.phases.phase_1_source_inventory,
            "D1": self.phases.phase_d1_unresolved,
            "2": self.phases.phase_2_discovery,
            "2b": self.phases.phase_2b_code_context,
            "3": self.phases.phase_3_profile_matching,
            "4": self.phases.phase_4_profile_design,
            "5": self.phases.phase_5_profile_validation,
            "6": self.phases.phase_6_data_collection,
            "7": self.phases.phase_7_registration,
            "8": self.phases.phase_8_cases,
            "9": self.phases.phase_9_queue,
            "10": self.phases.phase_10_bundles,
            "11": self.phases.phase_11_packets,
            "12": self.phases.phase_12_fix_lineage,
            "13": self.phases.phase_13_dashboard,
            "14": self.phases.phase_14_canaries,
        }

        try:
            for phase in PHASE_ORDER:
                if phase in self.state.completed_phases:
                    continue
                self.state.current_phase = phase
                self.store.save(self.state)
                log.info("BEGIN phase %s", phase)
                if phase == "15":
                    self.phases.phase_15_finalize_local()
                else:
                    await methods[phase]()
                self.state.completed_phases.append(phase)
                self.store.save(self.state)
                log.info("END phase %s", phase)
                if stop_after == phase:
                    self.state.status = "STOPPED_AFTER_PHASE"
                    self.state.completed_at = utc_now()
                    self.store.save(self.state)
                    return 0
        except Exception as exc:
            log.exception("pipeline aborted at phase %s", self.state.current_phase)
            self.state.status = "ABORTED"
            self.state.completed_at = utc_now()
            self.state.data["fatal_error"] = f"{type(exc).__name__}: {exc}"
            self.store.save(self.state)
            return 3

        self.state.current_phase = None
        self.state.completed_at = utc_now()
        final = self.state.data.get("final", {})
        verdict = final.get("executive_verdict", "UNKNOWN")
        if verdict.startswith("PASS") or verdict.startswith("DRY_RUN_PASS"):
            self.state.status = "COMPLETE"
            exit_code = 0
        elif verdict.startswith("DEFERRED"):
            self.state.status = "COMPLETE_PENDING"
            exit_code = 0
        else:
            self.state.status = "COMPLETE_WITH_GAPS"
            exit_code = 1
        self.store.save(self.state)

        # Candidate-A T2I atlas is a local, opt-in observability artifact.
        # It must never jeopardize the core RCA result: generation failures are
        # persisted explicitly and the terminal pipeline/report still completes.
        if self.settings.t2i_atlas_enabled:
            try:
                from .t2i_atlas import generate_t2i_v3_atlas

                atlas = generate_t2i_v3_atlas(
                    self.store.events_path,
                    self.store.run_dir,
                    profile=self.settings.t2i_atlas_profile,
                    expected_font_sha256=self.settings.t2i_atlas_font_sha256,
                    expected_font_bold_sha256=self.settings.t2i_atlas_font_bold_sha256,
                )
            except Exception as exc:
                log.exception("T2I atlas generation failed")
                atlas = {
                    "status": "GENERATION_ERROR",
                    "codec": "t2i-log-atlas/3.1",
                    "profile": self.settings.t2i_atlas_profile,
                    "source_path": str(self.store.events_path),
                    "error_class": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            self.state.data["t2i_atlas"] = atlas
            self.store.save(self.state)

        report_path = self.store.write_text("FINAL_REPORT.md", render_report(self.state))
        self.state.data["final_report_path"] = str(report_path)
        self.store.save(self.state)
        if self.settings.notify:
            report_text = summarize(self.state, self.store)
            self.store.write_text("WEBHOOK_REPORT.txt", report_text)
            result = send(self.settings.webhook_url, report_text)
            if result is not None:
                log.info("webhook notification sent successfully")
            elif self.settings.webhook_url:
                log.warning("webhook notification failed — report saved to WEBHOOK_REPORT.txt")
            else:
                log.info("webhook not configured — report saved to WEBHOOK_REPORT.txt")
        return exit_code
