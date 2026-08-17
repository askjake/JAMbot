"""Phase D3B2A: task-driven operational tool activation.

These tests prove that a repository checkout / isolated local deployment
request *asks for* the exact privileged tools it needs, that the request is
never an authorization, and that every existing control (D3B0 code-execution
privilege, D1 checkpointed policy, negation parsing) still holds.
"""

from __future__ import annotations

import pytest

from app.agent import operational_workflows as ow
from app.agent.methodology import select_methodology
from app.agent.tool_execution_gate import (
    evaluate_tool_call,
    required_authorizations_for_tool,
    tool_is_permitted_by_flags,
)
from app.agent.tool_execution_policy import build_profile_for_prompt, get_scoped_tools_for_prompt
from app.agent.tool_policy_state import (
    merge_authorization_state,
    parse_authorization_delta,
)
from app.agent.tool_profiles import (
    canonical_extra_tool,
    extra_tool_eligibility,
    extract_requested_extra_tools,
    is_code_execution_tool,
)

ORIGINAL_PROMPT = """use your ssh key and clone it:
https://gitlab.com/dish-cloud/dt/sse/datasolutions/cas/ai-test

follow protocol
then deploy and run it locally
follow protocol
Show reasoning"""

FULLY_AUTHORIZED_PROMPT = """Operator authorized.
Heavy tools authorized.
Mutation authorized.
Persistence authorized.

Use the configured SSH identity to clone:
https://gitlab.com/dish-cloud/dt/sse/datasolutions/cas/ai-test

Follow repository and deployment protocol.
Inspect it first, then deploy and run it in an isolated local environment.
Do not modify production or shared services."""

ALL_FALSE: dict[str, bool] = {
    "operator_authorized": False,
    "heavy_tools_authorized": False,
    "mutation_authorized": False,
    "persistence_authorized": False,
}
ALL_TRUE: dict[str, bool] = {key: True for key in ALL_FALSE}

CLONE = "agent_mode:agent_git_clone"
SHELL = "agent_mode:agent_run_shell"
VENV = "agent_mode:agent_create_venv"
PYTHON = "agent_mode:agent_run_python"

OPERATIONAL_RAW = ("agent_git_clone", "agent_run_shell", "agent_create_venv", "agent_run_python")


def _names(tools) -> list[str]:
    return [str(getattr(tool, "name", "") or "") for tool in tools]


def _bound(prompt: str, flags: dict[str, bool], **kwargs) -> list[str]:
    tools, _plan = get_scoped_tools_for_prompt(prompt, authorization_flags=flags, **kwargs)
    return _names(tools)


# 1 -------------------------------------------------------------------------
def test_clone_intent_detected():
    workflow = ow.detect_operational_workflow(ORIGINAL_PROMPT)
    assert workflow is not None
    assert workflow.workflow == ow.REPO_CHECKOUT_LOCAL_DEPLOY
    assert "repository_checkout" in workflow.triggers
    assert "configured_ssh_identity" in workflow.triggers


# 2 -------------------------------------------------------------------------
def test_local_deploy_intent_detected():
    workflow = ow.detect_operational_workflow(ORIGINAL_PROMPT)
    assert workflow is not None
    assert "local_deployment" in workflow.triggers
    assert workflow.deployment_scope == "isolated_local"


# 3 -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prompt",
    [
        "How does git clone work?",
        "What is a virtual environment?",
        "Review this pasted Dockerfile.",
        "Explain how to deploy a service locally.",
    ],
)
def test_discussion_only_prompts_do_not_activate(prompt):
    assert ow.detect_operational_workflow(prompt) is None
    assert extract_requested_extra_tools(prompt) == ()
    assert not [n for n in _bound(prompt, ALL_FALSE) if n in OPERATIONAL_RAW]


# 4 -------------------------------------------------------------------------
def test_private_gitlab_url_recognized():
    original, host, path = ow.extract_repository_target(ORIGINAL_PROMPT)
    assert host == "gitlab.com"
    assert path == "dish-cloud/dt/sse/datasolutions/cas/ai-test"
    assert original.startswith("https://gitlab.com/")


# 5 -------------------------------------------------------------------------
def test_ssh_clone_url_derived_correctly():
    workflow = ow.detect_operational_workflow(ORIGINAL_PROMPT)
    assert workflow is not None
    assert workflow.ssh_clone_url == (
        "git@gitlab.com:dish-cloud/dt/sse/datasolutions/cas/ai-test.git"
    )
    assert workflow.url_status == ow.URL_SSH_DERIVED


# 6 -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "host,path",
    [
        ("gitlab.com", "single-segment"),
        ("gitlab.com", "../etc/passwd"),
        ("gitlab.com", ""),
        ("", "group/project"),
        ("gitlab.com", "group//project"),
    ],
)
def test_malformed_repository_url_rejected(host, path):
    url, status = ow.derive_ssh_clone_url(host, path)
    assert url == ""
    assert status == ow.URL_MALFORMED


def test_untrusted_host_is_not_silently_rewritten():
    url, status = ow.derive_ssh_clone_url("evil.example.net", "group/project")
    assert url == ""
    assert status == ow.URL_UNTRUSTED_HOST


# 7 -------------------------------------------------------------------------
def test_exact_operational_tools_requested():
    requested = extract_requested_extra_tools(ORIGINAL_PROMPT)
    assert CLONE in requested
    assert SHELL in requested
    assert VENV in requested
    # agent_run_python is not requested unless the task actually needs it.
    assert PYTHON not in requested


def test_agent_run_python_requested_only_when_named():
    requested = extract_requested_extra_tools(
        ORIGINAL_PROMPT + "\nAlso run a python script to verify the output."
    )
    assert PYTHON in requested


# 8 -------------------------------------------------------------------------
def test_owner_family_activated():
    assert canonical_extra_tool("agent_git_clone") == CLONE
    profile = build_profile_for_prompt(ORIGINAL_PROMPT, authorization_flags=ALL_FALSE)
    assert "agent_mode" in profile.active_toolsets
    assert profile.operational_workflow == ow.REPO_CHECKOUT_LOCAL_DEPLOY


# 9 -------------------------------------------------------------------------
def test_unrelated_family_excluded():
    profile = build_profile_for_prompt(ORIGINAL_PROMPT, authorization_flags=ALL_FALSE)
    for family in ("s3_stb_logs", "epg_mcp", "qos_mcp", "viewership", "grasshopper_mcp"):
        assert family not in profile.active_toolsets


# 10 ------------------------------------------------------------------------
def test_all_false_authorization_leaves_tools_pending():
    profile = build_profile_for_prompt(ORIGINAL_PROMPT, authorization_flags=ALL_FALSE)
    assert CLONE in profile.pending_authorization_extra_tools
    assert SHELL in profile.pending_authorization_extra_tools
    assert VENV in profile.pending_authorization_extra_tools
    assert profile.eligible_extra_tools == ()
    assert not [n for n in _bound(ORIGINAL_PROMPT, ALL_FALSE) if n in OPERATIONAL_RAW]


# 11 ------------------------------------------------------------------------
def test_operator_only_is_insufficient():
    flags = dict(ALL_FALSE, operator_authorized=True)
    profile = build_profile_for_prompt(ORIGINAL_PROMPT, authorization_flags=flags)
    assert CLONE in profile.pending_authorization_extra_tools
    assert SHELL in profile.pending_authorization_extra_tools
    assert profile.eligible_extra_tools == ()


# 12 ------------------------------------------------------------------------
def test_heavy_only_is_insufficient():
    flags = dict(ALL_FALSE, heavy_tools_authorized=True)
    profile = build_profile_for_prompt(ORIGINAL_PROMPT, authorization_flags=flags)
    assert SHELL in profile.pending_authorization_extra_tools
    assert profile.eligible_extra_tools == ()


# 13 ------------------------------------------------------------------------
def test_mutation_requirement_enforced():
    flags = dict(ALL_TRUE, mutation_authorized=False)
    profile = build_profile_for_prompt(ORIGINAL_PROMPT, authorization_flags=flags)
    assert CLONE in profile.pending_authorization_extra_tools
    assert "mutation_authorized" in profile.missing_authorizations[CLONE]


# 14 ------------------------------------------------------------------------
def test_persistence_requirement_enforced():
    flags = dict(ALL_TRUE, persistence_authorized=False)
    profile = build_profile_for_prompt(ORIGINAL_PROMPT, authorization_flags=flags)
    assert CLONE in profile.pending_authorization_extra_tools
    assert "persistence_authorized" in profile.missing_authorizations[CLONE]


# 15 ------------------------------------------------------------------------
def test_same_turn_full_grant_promotes_tools():
    delta = parse_authorization_delta(FULLY_AUTHORIZED_PROMPT)
    flags = merge_authorization_state(dict(ALL_FALSE), delta)
    assert all(flags[key] for key in ALL_FALSE)
    profile = build_profile_for_prompt(FULLY_AUTHORIZED_PROMPT, authorization_flags=flags)
    assert CLONE in profile.eligible_extra_tools
    assert SHELL in profile.eligible_extra_tools
    assert profile.pending_authorization_extra_tools == ()
    bound = _bound(FULLY_AUTHORIZED_PROMPT, flags)
    assert "agent_git_clone" in bound
    assert "agent_run_shell" in bound


# 16 ------------------------------------------------------------------------
def test_later_grant_promotes_without_task_repetition():
    first = build_profile_for_prompt(ORIGINAL_PROMPT, authorization_flags=ALL_FALSE)
    follow_up = """Operator authorized.
Heavy tools authorized.
Mutation authorized.
Persistence authorized.

Continue the previously requested repository checkout and isolated local deployment."""
    delta = parse_authorization_delta(follow_up)
    flags = merge_authorization_state(dict(ALL_FALSE), delta)
    second = build_profile_for_prompt(
        follow_up,
        prior_active_toolsets=first.active_toolsets,
        prior_extra_tools=first.requested_extra_tools,
        authorization_flags=flags,
    )
    assert "gitlab.com" not in follow_up
    assert CLONE in second.eligible_extra_tools
    assert SHELL in second.eligible_extra_tools
    assert second.operational_workflow == ow.REPO_CHECKOUT_LOCAL_DEPLOY


# 17 ------------------------------------------------------------------------
def test_explicit_revocation_removes_eligibility():
    granted = build_profile_for_prompt(FULLY_AUTHORIZED_PROMPT, authorization_flags=ALL_TRUE)
    assert CLONE in granted.eligible_extra_tools
    revoke = "Revoke operator, heavy-tool, mutation, and persistence authorization."
    flags = merge_authorization_state(dict(ALL_TRUE), parse_authorization_delta(revoke))
    assert not any(flags[key] for key in ALL_FALSE)
    after = build_profile_for_prompt(
        revoke,
        prior_active_toolsets=granted.active_toolsets,
        prior_extra_tools=granted.requested_extra_tools,
        authorization_flags=flags,
    )
    assert after.eligible_extra_tools == ()
    assert CLONE in after.pending_authorization_extra_tools
    assert not [n for n in _bound(revoke, flags, prior_extra_tools=granted.requested_extra_tools) if n in OPERATIONAL_RAW]


# 18 ------------------------------------------------------------------------
def test_negative_conjunction_never_grants():
    text = (
        "Operator authorization and heavy-tool authorization are not granted.\n"
        "Do not authorize mutation or persistence.\n"
        "Clone https://gitlab.com/dish-cloud/dt/sse/datasolutions/cas/ai-test and run it locally."
    )
    flags = merge_authorization_state(dict(ALL_FALSE), parse_authorization_delta(text))
    assert not any(flags[key] for key in ALL_FALSE)
    profile = build_profile_for_prompt(text, authorization_flags=flags)
    assert profile.eligible_extra_tools == ()
    assert CLONE in profile.pending_authorization_extra_tools
    assert not [n for n in _bound(text, flags) if n in OPERATIONAL_RAW]


# 19 ------------------------------------------------------------------------
def test_authorization_question_does_not_mutate_state():
    question = "Are operator and heavy-tool authorization granted?"
    flags = merge_authorization_state(dict(ALL_FALSE), parse_authorization_delta(question))
    assert not any(flags[key] for key in ALL_FALSE)
    granted = merge_authorization_state(dict(ALL_TRUE), parse_authorization_delta(question))
    assert all(granted[key] for key in ALL_FALSE)


# 20 ------------------------------------------------------------------------
def test_generic_chat_gains_no_operational_tools():
    for prompt in ("Summarize the release notes.", "Draft an email about the outage."):
        assert not [n for n in _bound(prompt, ALL_TRUE) if n in OPERATIONAL_RAW]


# 21 ------------------------------------------------------------------------
def test_repository_review_without_execution_stays_read_only():
    prompt = "Review the code quality of https://gitlab.com/dish-cloud/dt/sse/datasolutions/cas/ai-test"
    assert ow.detect_operational_workflow(prompt) is None
    assert extract_requested_extra_tools(prompt) == ()


# 22 ------------------------------------------------------------------------
def test_operational_workflow_persists_across_followup():
    first = build_profile_for_prompt(ORIGINAL_PROMPT, authorization_flags=ALL_FALSE)
    follow = build_profile_for_prompt(
        "Proceed with the next step.",
        prior_active_toolsets=first.active_toolsets,
        prior_extra_tools=first.requested_extra_tools,
        authorization_flags=ALL_FALSE,
    )
    assert follow.operational_workflow == ow.REPO_CHECKOUT_LOCAL_DEPLOY
    assert CLONE in follow.requested_extra_tools
    assert CLONE in follow.pending_authorization_extra_tools


# 23 ------------------------------------------------------------------------
def test_profile_does_not_oscillate_across_workflow_sequence():
    sequence = [
        ORIGINAL_PROMPT,
        "Inspect the repository protocol files before executing anything.",
        "Install the dependencies in an isolated environment.",
        "Start the service on localhost only.",
        "Verify the health endpoint.",
        "Continue with the deployment.",
    ]
    prior_toolsets: tuple[str, ...] = ()
    prior_extras: tuple[str, ...] = ()
    methodologies: list[str] = []
    for prompt in sequence:
        profile = build_profile_for_prompt(
            prompt,
            prior_active_toolsets=prior_toolsets,
            prior_extra_tools=prior_extras,
            authorization_flags=ALL_TRUE,
        )
        methodologies.append(profile.methodology)
        prior_toolsets = profile.active_toolsets
        prior_extras = profile.requested_extra_tools
        assert CLONE in profile.requested_extra_tools
        assert profile.operational_workflow == ow.REPO_CHECKOUT_LOCAL_DEPLOY
    assert set(methodologies) == {"repo_checkout_local_deploy"}


# 24 ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "follow_up", ["Continue with the deployment.", "Proceed with the next step."]
)
def test_generic_continuation_does_not_inject_irrelevant_family(follow_up):
    first = build_profile_for_prompt(ORIGINAL_PROMPT, authorization_flags=ALL_TRUE)
    assert "dish_internal" not in first.active_toolsets
    follow = build_profile_for_prompt(
        follow_up,
        prior_active_toolsets=first.active_toolsets,
        prior_extra_tools=first.requested_extra_tools,
        authorization_flags=ALL_TRUE,
    )
    assert "dish_internal" not in follow.active_toolsets
    assert set(follow.active_toolsets) == set(first.active_toolsets)


# 25 ------------------------------------------------------------------------
def test_exact_last_binding_matches_eligible_workflow_tools():
    profile = build_profile_for_prompt(ORIGINAL_PROMPT, authorization_flags=ALL_TRUE)
    bound = set(_bound(ORIGINAL_PROMPT, ALL_TRUE))
    eligible_raw = {value.split(":", 1)[1] for value in profile.eligible_extra_tools}
    assert eligible_raw <= bound
    # agent_run_python was never requested by this workflow, so full
    # authorization must not surface it.
    assert "agent_run_python" not in bound
    assert profile.code_execution_allowlist == tuple(sorted(eligible_raw))


# 26 ------------------------------------------------------------------------
def test_stale_prior_binding_cannot_execute_after_revocation():
    call = {"name": "agent_git_clone", "id": "call-1", "args": {"repo_url": "git@gitlab.com:g/p.git"}}
    # Bound last turn, but authorization has since been revoked.
    revoked = evaluate_tool_call(
        call, last_bound_tool_names=["agent_git_clone"], authorization_flags=ALL_FALSE
    )
    assert not revoked.allowed
    assert "mutation_authorized" in revoked.missing_authorizations
    # Authorized now, but the tool is not in the current binding.
    stale = evaluate_tool_call(call, last_bound_tool_names=[], authorization_flags=ALL_TRUE)
    assert not stale.allowed
    assert stale.result_code == "TOOL_NOT_IN_LAST_BINDING"


# 27 ------------------------------------------------------------------------
def test_management_facade_reports_actual_state():
    from app.agent.agents.tools.management import _operational_workflow_state

    unknown = _operational_workflow_state({"state_known": False, "reason": "NO_ACTIVE_PARENT_TURN_CONTEXT"})
    assert unknown["state_known"] is False
    assert unknown["workflow"] == ""

    known = _operational_workflow_state(
        {
            "state_known": True,
            "requested_extra_tools": [CLONE, SHELL, "s3_stb_logs:list_dates"],
            "pending_authorization_extra_tools": [CLONE, SHELL],
            "eligible_extra_tools": [],
            "unavailable_extra_tools": [],
            "authorization_flags": dict(ALL_FALSE),
        }
    )
    assert known["workflow"] == ow.REPO_CHECKOUT_LOCAL_DEPLOY
    assert known["activation_status"] == "ACTIVATION_REQUESTED_PENDING_AUTHORIZATION"
    assert known["requested_operational_tools"] == sorted([CLONE, SHELL])
    assert "s3_stb_logs:list_dates" not in known["requested_operational_tools"]
    assert "mutation_authorized" in known["missing_authorization_grants"]
    assert known["ssh_environment"]["key_material_exposed"] is False


# 28 ------------------------------------------------------------------------
def test_audit_records_no_argument_values():
    from app.agent.tool_execution_audit import build_event, sanitize_event

    decision = evaluate_tool_call(
        {
            "name": "agent_git_clone",
            "id": "call-2",
            "args": {"repo_url": "git@gitlab.com:secret-group/secret-project.git"},
        },
        last_bound_tool_names=["agent_git_clone"],
        authorization_flags=ALL_FALSE,
    )
    event = sanitize_event(
        build_event(
            tool_name=decision.tool_name,
            tool_call_id=decision.tool_call_id,
            tool_was_bound=True,
            binding_known=True,
            required_capabilities=decision.required_authorizations,
            authorization_flags=ALL_FALSE,
            result_code=decision.result_code,
            argument_field_names=decision.original_argument_names,
        )
    )
    blob = repr(event)
    # Argument *names* are legitimate audit metadata; argument *values*
    # (which carry the repository identity) must never be recorded.
    assert "repo_url" in blob
    assert "secret-group" not in blob
    assert "secret-project" not in blob
    assert "gitlab.com" not in blob


# 29 ------------------------------------------------------------------------
def test_no_ssh_key_material_in_workflow_state():
    status = ow.ssh_environment_status()
    assert set(status) == {
        "schema",
        "ssh_executable_present",
        "identity_file_count",
        "client_config_present",
        "status",
        "key_material_exposed",
    }
    assert status["key_material_exposed"] is False
    workflow = ow.detect_operational_workflow(ORIGINAL_PROMPT)
    blob = repr(workflow.to_dict())
    for marker in ("BEGIN OPENSSH PRIVATE KEY", "BEGIN RSA PRIVATE KEY", "id_rsa", "id_ed25519"):
        assert marker not in blob


# 30 ------------------------------------------------------------------------
@pytest.mark.parametrize("name", OPERATIONAL_RAW)
def test_code_and_shell_execution_remain_privileged(name):
    if name == "agent_git_clone":
        expected = {"operator_authorized", "mutation_authorized", "persistence_authorized"}
    else:
        expected = {
            "operator_authorized",
            "heavy_tools_authorized",
            "mutation_authorized",
            "persistence_authorized",
        }
    assert is_code_execution_tool(name)
    assert set(required_authorizations_for_tool(name)) == expected
    assert not tool_is_permitted_by_flags(name, ALL_FALSE)
    assert tool_is_permitted_by_flags(name, ALL_TRUE)
    ok, missing = extra_tool_eligibility(f"agent_mode:{name}", ALL_FALSE)
    assert ok is False
    assert set(missing) == expected


def test_methodology_selected_for_operational_request():
    assert select_methodology(ORIGINAL_PROMPT)["name"] == "repo_checkout_local_deploy"
    assert select_methodology("How does git clone work?")["name"] != "repo_checkout_local_deploy"
