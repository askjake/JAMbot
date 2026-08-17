"""Runtime configuration. Secrets and write authorization are environment-only."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

_BOOTSTRAP_ENV_FILE_PATH = ""
_BOOTSTRAPPED_KEYS: set[str] = set()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "var" / "nightly_rca_v6"


# ---------------------------------------------------------------------------
# Env-file bootstrap
# ---------------------------------------------------------------------------
# When invoked directly (e.g. `python -m apps.nightly_rca.run`) the shell
# wrapper (run_nightly.sh) is bypassed and no env file is sourced.  This
# function fills that gap by loading config/nightly_rca.env from the repo
# root, but only for variables that are *not already set* in the process
# environment, so shell-sourced invocations are completely unaffected.
# ---------------------------------------------------------------------------

def _bootstrap_env_file() -> None:
    """Load config/nightly_rca.env if env vars are absent."""
    global _BOOTSTRAP_ENV_FILE_PATH, _BOOTSTRAPPED_KEYS
    # Walk up from this file to the repo root (apps/nightly_rca/config.py
    # → apps/nightly_rca/ → apps/ → repo-root/)
    env_file_override = os.getenv("NIGHTLY_RCA_ENV_FILE")
    if env_file_override:
        candidate = Path(env_file_override)
    else:
        candidate = Path(__file__).resolve().parent.parent.parent / "config" / "nightly_rca.env"

    _BOOTSTRAP_ENV_FILE_PATH = str(candidate)
    if not candidate.is_file():
        logger.debug("nightly_rca: env bootstrap skipped — file not found: %s", candidate)
        return

    candidate_keys: set[str] = set()
    try:
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key = line.partition("=")[0].strip()
            if key:
                candidate_keys.add(key)
    except OSError:
        candidate_keys = set()
    absent_before = {key for key in candidate_keys if key not in os.environ}

    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
        loaded = load_dotenv(dotenv_path=candidate, override=False)
        if loaded:
            _BOOTSTRAPPED_KEYS.update(key for key in absent_before if key in os.environ)
            logger.debug("nightly_rca: bootstrapped env from %s", candidate)
    except ImportError:
        # python-dotenv not available; parse the file manually with stdlib.
        # This is a safety net — python-dotenv is in requirements.txt.
        with candidate.open() as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        _BOOTSTRAPPED_KEYS.update(key for key in absent_before if key in os.environ)
        logger.debug("nightly_rca: bootstrapped env from %s (stdlib fallback)", candidate)


_bootstrap_env_file()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _list(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(x.strip() for x in raw.split(",") if x.strip()) if raw else default


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    output_dir: Path = Path(os.getenv("NIGHTLY_RCA_OUTPUT_DIR", str(_DEFAULT_OUTPUT_DIR)))
    role: str = os.getenv("NIGHTLY_RCA_ROLE", "operator")
    queue_id: str = os.getenv("NIGHTLY_RCA_QUEUE_ID", "main")
    commit: bool = _bool("NIGHTLY_RCA_COMMIT", False)
    notify: bool = _bool("NIGHTLY_RCA_NOTIFY", True)
    webhook_url: str = os.getenv("NIGHTLY_RCA_WEBHOOK", "")
    aws_region: str = os.getenv("NIGHTLY_RCA_AWS_REGION", os.getenv("AWS_REGION", "us-west-2"))
    report_recipients: tuple[str, ...] = field(
        default_factory=lambda: _list("NIGHTLY_RCA_RECIPIENTS")
    )
    test_receiver_id: str = os.getenv("NIGHTLY_RCA_TEST_RECEIVER", "R0000000001")
    discovery_hours: int = _int("NIGHTLY_RCA_DISCOVERY_HOURS", 24)
    max_unresolved_cases: int = _int("NIGHTLY_RCA_MAX_UNRESOLVED", 10)
    max_candidate_clusters: int = _int("NIGHTLY_RCA_MAX_CLUSTERS", 20)
    max_profiles_per_run: int = _int("NIGHTLY_RCA_MAX_PROFILES", 10)
    max_cases_per_run: int = _int("NIGHTLY_RCA_MAX_CASES", 10)
    max_upload_files: int = _int("NIGHTLY_RCA_MAX_UPLOAD_FILES", 50)
    max_runtime_seconds: int = _int("NIGHTLY_RCA_MATERIALIZER_SECONDS", 45)
    # Log acquisition wait-and-poll settings (Phase 5)
    # The cron is resumable; default to one immediate receipt reconciliation and
    # persist unfinished work for the next run instead of sleeping in-process.
    log_poll_max_attempts: int = _int("NIGHTLY_RCA_LOG_POLL_ATTEMPTS", 1)
    log_poll_interval_seconds: int = _int("NIGHTLY_RCA_LOG_POLL_INTERVAL", 0)
    log_cohort_batch_size: int = _int("NIGHTLY_RCA_COHORT_BATCH", 5)
    log_cohort_max_peers: int = _int("NIGHTLY_RCA_COHORT_MAX_PEERS", 20)
    pending_batch_size: int = max(1, min(50, _int("NIGHTLY_RCA_PENDING_BATCH_SIZE", 25)))
    history_runs: int = _int("NIGHTLY_RCA_HISTORY_RUNS", 14)
    # T2I Log Atlas v3 is intentionally opt-in until holdout/live acceptance.
    t2i_atlas_enabled: bool = _bool("NIGHTLY_RCA_T2I_ATLAS_ENABLED", False)
    t2i_atlas_profile: str = os.getenv("NIGHTLY_RCA_T2I_ATLAS_PROFILE", "candidate_a").strip()
    # D0-frozen font identities.  A font package change is representation drift,
    # so enabled production generation fails visibly rather than silently moving
    # off the validated raster family.
    t2i_atlas_font_sha256: str = os.getenv(
        "NIGHTLY_RCA_T2I_ATLAS_FONT_SHA256",
        "39c29931201f08dd89fdba4129c76288e6baeecf7c94fe4f6a757f2b50718b1b",
    ).strip().lower()
    t2i_atlas_font_bold_sha256: str = os.getenv(
        "NIGHTLY_RCA_T2I_ATLAS_FONT_BOLD_SHA256",
        "f8aa70b5d4210d77624f5b5b5cf094fadf9fd019caf5b528fa5af42fc4ac43a5",
    ).strip().lower()
    # Qualified selective visual context is independent of the legacy full-run
    # observability atlas and remains disabled until guarded post-install canary.
    t2i_selective_context_enabled: bool = _bool(
        "NIGHTLY_RCA_T2I_SELECTIVE_CONTEXT_ENABLED", False
    )
    t2i_selective_allowed_emails: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            item.lower()
            for item in _list("NIGHTLY_RCA_T2I_SELECTIVE_ALLOWED_EMAILS")
        )
    )
    t2i_runs_root: Path = Path(
        os.getenv(
            "NIGHTLY_RCA_T2I_RUNS_ROOT",
            str(_DEFAULT_OUTPUT_DIR / "runs"),
        )
    )
    t2i_selective_max_events: int = _int(
        "NIGHTLY_RCA_T2I_SELECTIVE_MAX_EVENTS", 5
    )
    server_urls: dict[str, str] = field(default_factory=lambda: {
        "s3_stb_logs": os.getenv("NIGHTLY_RCA_S3_MCP_URL", ""),
        "rtr_alerts_mcp": os.getenv("NIGHTLY_RCA_RTR_MCP_URL", ""),
        "grasshopper_mcp": os.getenv("NIGHTLY_RCA_GRASSHOPPER_MCP_URL", ""),
        # Code intelligence — optional; skipped gracefully when empty
        "code_tools_mcp": os.getenv("NIGHTLY_RCA_CODE_MCP_URL", ""),
    })

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.role != "operator":
            errors.append("NIGHTLY_RCA_ROLE must be 'operator'")
        if self.commit and not _bool("NIGHTLY_RCA_WRITE_AUTHORIZED", False):
            errors.append("commit requested without NIGHTLY_RCA_WRITE_AUTHORIZED=true")
        if self.max_upload_files > 50:
            errors.append("NIGHTLY_RCA_MAX_UPLOAD_FILES may not exceed 50")
        if self.t2i_atlas_enabled and self.t2i_atlas_profile != "candidate_a":
            errors.append(
                "NIGHTLY_RCA_T2I_ATLAS_PROFILE must be 'candidate_a' when T2I atlas is enabled"
            )
        if self.t2i_atlas_enabled or self.t2i_selective_context_enabled:
            for name, value in (
                ("NIGHTLY_RCA_T2I_ATLAS_FONT_SHA256", self.t2i_atlas_font_sha256),
                ("NIGHTLY_RCA_T2I_ATLAS_FONT_BOLD_SHA256", self.t2i_atlas_font_bold_sha256),
            ):
                if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                    errors.append(f"{name} must be a lowercase/uppercase SHA256 hex digest")
        if not 1 <= self.t2i_selective_max_events <= 5:
            errors.append("NIGHTLY_RCA_T2I_SELECTIVE_MAX_EVENTS must be within 1..5")
        if self.t2i_selective_context_enabled and not self.t2i_selective_allowed_emails:
            errors.append(
                "NIGHTLY_RCA_T2I_SELECTIVE_ALLOWED_EMAILS must be non-empty when selective context is enabled"
            )
        return errors


def _configuration_source(
    env_name: str,
    *,
    process_env: Mapping[str, str],
    bootstrapped_keys: set[str],
    cli_source: str = "",
) -> str:
    if cli_source:
        return cli_source
    if env_name in bootstrapped_keys:
        return "ENV_FILE_BOOTSTRAP"
    if env_name in process_env:
        if process_env.get("NIGHTLY_RCA_LAUNCHER_ENV_SOURCE"):
            return "LAUNCHER_ENV_FILE"
        return "PROCESS_ENV"
    return "DEFAULT"


def build_effective_configuration_provenance(
    settings: Settings,
    *,
    process_env: Mapping[str, str] | None = None,
    cli_mode_source: str = "",
    resumed: bool = False,
    env_file_path: Path | str | None = None,
    bootstrapped_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """Return value-minimized effective configuration and source provenance.

    Secrets, endpoint values, and webhook URLs are never returned.  Timezone is
    called verified only when CRON_TZ or TZ is directly present in the process
    environment; schedule arithmetic is not promoted to proof.
    """

    env = dict(os.environ if process_env is None else process_env)
    bootstrap = {str(key) for key in bootstrapped_keys}
    mode_source = "RESUMED_RUN_STATE" if resumed else (
        cli_mode_source
        or _configuration_source(
            "NIGHTLY_RCA_COMMIT",
            process_env=env,
            bootstrapped_keys=bootstrap,
        )
    )
    timezone_name = "UNKNOWN"
    timezone_evidence = "NOT_DIRECTLY_VERIFIED"
    timezone_confidence = "UNKNOWN"
    if str(env.get("CRON_TZ") or "").strip():
        timezone_name = str(env["CRON_TZ"]).strip()[:128]
        timezone_evidence = "CRON_TZ_ENV"
        timezone_confidence = "DIRECT"
    elif str(env.get("TZ") or "").strip():
        timezone_name = str(env["TZ"]).strip()[:128]
        timezone_evidence = "TZ_ENV"
        timezone_confidence = "DIRECT"

    resolved_env_file = str(
        env_file_path
        or env.get("NIGHTLY_RCA_LAUNCHER_ENV_SOURCE")
        or env.get("NIGHTLY_RCA_ENV_FILE")
        or _BOOTSTRAP_ENV_FILE_PATH
        or ""
    )[:512]
    configuration_sources = {
        "commit": mode_source,
        "output_dir": _configuration_source(
            "NIGHTLY_RCA_OUTPUT_DIR", process_env=env, bootstrapped_keys=bootstrap
        ),
        "notify": _configuration_source(
            "NIGHTLY_RCA_NOTIFY", process_env=env, bootstrapped_keys=bootstrap
        ),
        "max_upload_files": _configuration_source(
            "NIGHTLY_RCA_MAX_UPLOAD_FILES", process_env=env, bootstrapped_keys=bootstrap
        ),
        "pending_batch_size": _configuration_source(
            "NIGHTLY_RCA_PENDING_BATCH_SIZE", process_env=env, bootstrapped_keys=bootstrap
        ),
        "t2i_atlas_enabled": _configuration_source(
            "NIGHTLY_RCA_T2I_ATLAS_ENABLED", process_env=env, bootstrapped_keys=bootstrap
        ),
        "t2i_atlas_profile": _configuration_source(
            "NIGHTLY_RCA_T2I_ATLAS_PROFILE", process_env=env, bootstrapped_keys=bootstrap
        ),
        "t2i_selective_context_enabled": _configuration_source(
            "NIGHTLY_RCA_T2I_SELECTIVE_CONTEXT_ENABLED",
            process_env=env,
            bootstrapped_keys=bootstrap,
        ),
        "t2i_selective_allowed_emails": _configuration_source(
            "NIGHTLY_RCA_T2I_SELECTIVE_ALLOWED_EMAILS",
            process_env=env,
            bootstrapped_keys=bootstrap,
        ),
        "t2i_runs_root": _configuration_source(
            "NIGHTLY_RCA_T2I_RUNS_ROOT",
            process_env=env,
            bootstrapped_keys=bootstrap,
        ),
        "t2i_selective_max_events": _configuration_source(
            "NIGHTLY_RCA_T2I_SELECTIVE_MAX_EVENTS",
            process_env=env,
            bootstrapped_keys=bootstrap,
        ),
        "server_urls": "CONFIGURED_PRESENCE_ONLY",
    }
    return {
        "schema": "nightly_rca_effective_configuration.v1",
        "effective_mode": "commit" if settings.commit else "dry_run",
        "effective_mode_source": mode_source,
        "role": settings.role,
        "write_authorization_present": str(env.get("NIGHTLY_RCA_WRITE_AUTHORIZED") or "").strip().lower()
        in {"1", "true", "yes", "on"},
        "notify_enabled": bool(settings.notify),
        "webhook_configured": bool(settings.webhook_url),
        "output_dir": str(settings.output_dir),
        "max_upload_files": int(settings.max_upload_files),
        "pending_batch_size": int(settings.pending_batch_size),
        "t2i_atlas_enabled": bool(settings.t2i_atlas_enabled),
        "t2i_atlas_profile": str(settings.t2i_atlas_profile),
        "t2i_atlas_font_identity_pinned": bool(
            settings.t2i_atlas_font_sha256 and settings.t2i_atlas_font_bold_sha256
        ),
        "t2i_selective_context_enabled": bool(settings.t2i_selective_context_enabled),
        "t2i_selective_allowed_email_count": len(settings.t2i_selective_allowed_emails),
        "t2i_runs_root": str(settings.t2i_runs_root),
        "t2i_selective_max_events": int(settings.t2i_selective_max_events),
        "configured_server_families": sorted(
            name for name, value in settings.server_urls.items() if value
        ),
        "configuration_sources": configuration_sources,
        "env_file_path": resolved_env_file,
        "env_file_exists": bool(resolved_env_file and Path(resolved_env_file).is_file()),
        "cron_timezone": timezone_name,
        "cron_timezone_evidence": timezone_evidence,
        "cron_timezone_confidence": timezone_confidence,
        "cron_timezone_inference_allowed": False,
    }


def current_bootstrap_provenance() -> tuple[str, frozenset[str]]:
    return _BOOTSTRAP_ENV_FILE_PATH, frozenset(_BOOTSTRAPPED_KEYS)


MUTATING_TOOL_NAMES = frozenset({
    "register_issue_profile",
    "record_investigation_case",
    "record_case_outcome",
    "create_human_review_queue",
    "build_human_evidence_bundle",
    "export_human_adjudication_packet",
    "human_review_fix_lineage_refresh",
    "human_review_materialize_engineer_contexts",
    "grasshopper_upload_profile_logs",
})
