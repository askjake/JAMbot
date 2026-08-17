"""Typed task-continuity checkpoint selection.

Continuity is not reconstructed by scanning retained/compressed messages.  The
current user turn is evaluated first.  A checkpoint may be restored only for a
content-free continuation in the same authoritative environment.  An explicit
new task always wins, which prevents a stale repository-deployment workflow
from displacing an active Incident Scene or Grasshopper investigation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_MAX_TEXT = 256
_CONTINUATION_RE = re.compile(
    r"^\s*(?:please\s+)?(?:continue|proceed|go\s+on|keep\s+going|resume|"
    r"move\s+forward|proceed\s+with\s+steps?\s+[0-9]+(?:\s*[-–]\s*[0-9]+)?|"
    r"continue\s+with\s+steps?\s+[0-9]+(?:\s*[-–]\s*[0-9]+)?)\s*[.!]?\s*$",
    re.IGNORECASE,
)
_INCIDENT_SCENE_RE = re.compile(
    r"\bincident\s+scene\b|\b(?:build|validate|query|render|expand|get)_incident_scene\b|\bscene-[0-9a-f]{24}\b",
    re.IGNORECASE,
)
_GRASSHOPPER_RE = re.compile(
    r"\bgrasshopper\b.*\b(?:plan|upload|request|fresh|nal)\b|"
    r"\b(?:request|upload|plan)\b.*\b(?:fresh\s+)?(?:nal\s+)?logs?\b",
    re.IGNORECASE,
)
_S3_HISTORICAL_RE = re.compile(
    r"\b(?:list|read|search|find|historical|already\s+uploaded)\b.*\b(?:s3|files?|logs?)\b|"
    r"\bs3\b.*\b(?:list|read|search|historical)\b",
    re.IGNORECASE,
)
_REPO_RE = re.compile(
    r"\b(?:git\s+)?clone\b|\bcheckout\b|\bworktree\b|"
    r"\bdeploy\b.*\b(?:repo|repository|locally|local)\b",
    re.IGNORECASE,
)
_NIGHTLY_RE = re.compile(r"\bnightly\s+rca\b|\bcron\b.*\b(?:rca|grasshopper|logs?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ContinuityCheckpoint:
    task_scope: str = ""
    methodology: str = ""
    authoritative_environment: str = ""
    revision: int = 0


@dataclass(frozen=True)
class ContinuityResolution:
    task_scope: str
    methodology: str
    authoritative_environment: str
    restored_from_checkpoint: bool
    stale_checkpoint_rejected: bool
    environment_match: bool
    next_revision: int


def is_content_free_continuation(prompt: str | None) -> bool:
    return bool(_CONTINUATION_RE.fullmatch(str(prompt or "")[:_MAX_TEXT]))


def task_scope_from_prompt(prompt: str | None) -> str:
    text = str(prompt or "")[:4000]
    if not text or is_content_free_continuation(text):
        return ""
    if _INCIDENT_SCENE_RE.search(text):
        return "incident_scene"
    if _GRASSHOPPER_RE.search(text):
        return "grasshopper_log_acquisition"
    if _S3_HISTORICAL_RE.search(text):
        return "s3_historical_log_read"
    if _NIGHTLY_RE.search(text):
        return "nightly_rca"
    if _REPO_RE.search(text):
        return "repo_checkout_local_deploy"
    return "generic_engineering"


def resolve_continuity(
    *,
    current_prompt: str | None,
    selected_methodology: str,
    checkpoint: ContinuityCheckpoint | None = None,
    current_environment: str = "",
) -> ContinuityResolution:
    prior = checkpoint or ContinuityCheckpoint()
    current_scope = task_scope_from_prompt(current_prompt)
    current_environment = str(current_environment or "")[:256]
    prior_environment = str(prior.authoritative_environment or "")[:256]
    environment_match = bool(
        current_environment and prior_environment and current_environment == prior_environment
    )
    prior_revision = max(0, int(prior.revision or 0))

    # The current turn is authoritative.  Explicit domain/task language rejects
    # a stale checkpoint before any methodology/profile construction occurs.
    if current_scope:
        stale_rejected = bool(prior.task_scope and current_scope != prior.task_scope)
        changed = current_scope != prior.task_scope or selected_methodology != prior.methodology
        return ContinuityResolution(
            task_scope=current_scope,
            methodology=str(selected_methodology or "generic_engineering"),
            authoritative_environment=current_environment,
            restored_from_checkpoint=False,
            stale_checkpoint_rejected=stale_rejected,
            environment_match=environment_match,
            next_revision=prior_revision + (1 if changed else 0),
        )

    if is_content_free_continuation(current_prompt) and prior.task_scope and environment_match:
        return ContinuityResolution(
            task_scope=str(prior.task_scope),
            methodology=str(prior.methodology or selected_methodology or "generic_engineering"),
            authoritative_environment=current_environment,
            restored_from_checkpoint=True,
            stale_checkpoint_rejected=False,
            environment_match=True,
            next_revision=prior_revision,
        )

    return ContinuityResolution(
        task_scope="generic_engineering",
        methodology=str(selected_methodology or "generic_engineering"),
        authoritative_environment=current_environment,
        restored_from_checkpoint=False,
        stale_checkpoint_rejected=bool(prior.task_scope and not environment_match),
        environment_match=environment_match,
        next_revision=prior_revision + (1 if prior.task_scope != "generic_engineering" else 0),
    )
