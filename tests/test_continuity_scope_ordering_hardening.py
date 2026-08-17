from __future__ import annotations

from app.agent.continuity_policy import (
    ContinuityCheckpoint,
    resolve_continuity,
    task_scope_from_prompt,
)


def _checkpoint(**overrides):
    values = {
        "task_scope": "incident_scene",
        "methodology": "tool_systematic_validation",
        "authoritative_environment": "10.79.85.35|dsgpu3090-Lambda-Vector",
        "revision": 7,
    }
    values.update(overrides)
    return ContinuityCheckpoint(**values)


def test_explicit_incident_scene_task_overrides_stale_repo_workflow():
    stale = _checkpoint(task_scope="repo_checkout_local_deploy", methodology="repo_checkout_local_deploy")
    resolution = resolve_continuity(
        current_prompt="S3 Incident Scene IAM post-merge verification",
        selected_methodology="generic_engineering",
        checkpoint=stale,
        current_environment="10.79.85.35|dsgpu3090-Lambda-Vector",
    )
    assert resolution.task_scope == "incident_scene"
    assert resolution.methodology == "generic_engineering"
    assert resolution.restored_from_checkpoint is False
    assert resolution.stale_checkpoint_rejected is True


def test_content_free_steps_followup_restores_active_plan_in_same_environment():
    resolution = resolve_continuity(
        current_prompt="proceed with Steps 3-5",
        selected_methodology="generic_engineering",
        checkpoint=_checkpoint(),
        current_environment="10.79.85.35|dsgpu3090-Lambda-Vector",
    )
    assert resolution.task_scope == "incident_scene"
    assert resolution.methodology == "tool_systematic_validation"
    assert resolution.restored_from_checkpoint is True


def test_environment_mismatch_prevents_checkpoint_restore():
    resolution = resolve_continuity(
        current_prompt="continue",
        selected_methodology="generic_engineering",
        checkpoint=_checkpoint(),
        current_environment="10.79.85.47|dsgpu3080-Lambda-Vector",
    )
    assert resolution.restored_from_checkpoint is False
    assert resolution.environment_match is False
    assert resolution.task_scope == "generic_engineering"


def test_checkpoint_survives_message_compression_because_no_history_scan_is_used():
    checkpoint = _checkpoint()
    # No message window is supplied. The transition depends only on the current
    # user turn and the typed checkpoint envelope.
    resolution = resolve_continuity(
        current_prompt="proceed with Steps 3-5",
        selected_methodology="generic_engineering",
        checkpoint=checkpoint,
        current_environment=checkpoint.authoritative_environment,
    )
    assert resolution.restored_from_checkpoint
    assert resolution.next_revision == checkpoint.revision


def test_current_user_domain_switch_wins_before_checkpoint_restore():
    checkpoint = _checkpoint()
    resolution = resolve_continuity(
        current_prompt="Now clone the private repository and deploy it locally",
        selected_methodology="repo_code_review",
        checkpoint=checkpoint,
        current_environment=checkpoint.authoritative_environment,
    )
    assert resolution.task_scope == "repo_checkout_local_deploy"
    assert resolution.restored_from_checkpoint is False
    assert resolution.next_revision == checkpoint.revision + 1


def test_scope_detector_distinguishes_grasshopper_from_historical_s3_read():
    assert task_scope_from_prompt("request fresh NAL logs through Grasshopper") == "grasshopper_log_acquisition"
    assert task_scope_from_prompt("list historical NAL files in S3") == "s3_historical_log_read"
