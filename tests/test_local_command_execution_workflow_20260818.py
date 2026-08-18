"""Regression tests for the 2026-08-18 local_command_execution workflow.

Closes the code-execution dead-end under the two GENERIC_CODE_EXECUTION_
METHODOLOGIES ("generic_engineering", "no_tool_response").

Background. build_tool_profile() builds `code_execution_allowlist` from
`eligible INTERSECT operational_requested` whenever a workflow is active OR the
methodology is generic. detect_operational_workflow() previously returned None
for every prompt without a repository URL, so a plain-English execution request
produced an empty operational set -> an empty-but-not-None allowlist -> the
workflow gate in get_tools_for_toolsets() withheld every code-execution tool
BEFORE the authorization gate was consulted. Granting all four authorization
flags could not override it.

Reproduced live before the fix: a short "Proceed with the ... task as
previously specified." follow-up selects generic_engineering, parses 4/4
authorization flags true, and still binds zero operational tools.

The fix adds a no-repository-target workflow so a genuine local command or
filesystem-inspection request can declare the executor tools it needs. It is a
capability request, never an authorization decision -- the declared tools still
have to clear extra_tool_eligibility().

The D3B2A/D3B0 invariant is preserved and independently asserted below:
authorization must never, by itself, bind a privileged executor into a chat
that never asked for one.
"""

from __future__ import annotations

import pytest

from app.agent import operational_workflows as ow
from app.agent.operational_workflows import (
    LOCAL_COMMAND_EXECUTION,
    detect_operational_workflow,
)
from app.agent.tool_execution_policy import get_scoped_tools_for_prompt
from app.agent.tool_profiles import build_tool_profile, extract_requested_extra_tools

OPERATIONAL_RAW = ("agent_git_clone", "agent_run_shell", "agent_create_venv", "agent_run_python")
ALL_TRUE = {
    "operator_authorized": True,
    "heavy_tools_authorized": True,
    "mutation_authorized": True,
    "persistence_authorized": True,
}
ALL_FALSE: dict[str, bool] = {}

#: Prompts that must NEVER register an operational need.
BENIGN = [
    "Please summarise the attached document.",
    "Summarize the release notes.",
    "Draft an email about the outage.",
    "Proceed with the task as previously specified.",
    "How does git clone work?",
    "What is a virtual environment?",
]
#: Genuine local execution / inspection requests with no repository target.
EXECUTION = [
    "Please run these diagnostic bash commands on the host: aws s3api get-bucket-location.",
    "Read README.md and CLAUDE.md, then every module under src/orchestrator/, and produce a plan.",
    "Analyze the app in ~/ai-test and develop a plan to integrate a GUI.",
    "Execute the following shell command to check connectivity.",
]


def _bound(prompt, flags):
    tools, _plan = get_scoped_tools_for_prompt(prompt, authorization_flags=dict(flags))
    names = {str(getattr(t, "name", "")) for t in tools}
    return sorted(n for n in OPERATIONAL_RAW if n in names)


def _allowlist(prompt, flags, methodology="generic_engineering"):
    return build_tool_profile(
        methodology=methodology,
        methodology_toolsets=("agent_mode", "search"),
        prompt=prompt,
        authorization_flags=flags,
    ).code_execution_allowlist


# ─── The invariant (must hold exactly as before the fix) ──────────────────────

@pytest.mark.parametrize("prompt", BENIGN)
def test_benign_prompts_register_no_operational_workflow(prompt):
    assert detect_operational_workflow(prompt) is None


@pytest.mark.parametrize("prompt", BENIGN)
def test_benign_prompts_bind_nothing_even_fully_authorized(prompt):
    """D3B2A: authorization alone must not bind executors into ordinary chat."""
    assert _bound(prompt, ALL_TRUE) == []


def test_review_only_repository_prompt_stays_read_only():
    """A prompt WITH a repo URL must not reach the new no-repo branch."""
    prompt = (
        "Review the code quality of "
        "https://gitlab.com/dish-cloud/dt/sse/datasolutions/cas/ai-test"
    )
    assert detect_operational_workflow(prompt) is None
    assert extract_requested_extra_tools(prompt) == ()
    assert _bound(prompt, ALL_TRUE) == []


# ─── The fix ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("prompt", EXECUTION)
def test_execution_requests_register_the_local_workflow(prompt):
    workflow = detect_operational_workflow(prompt)
    assert workflow is not None
    assert workflow.workflow == LOCAL_COMMAND_EXECUTION
    assert "local_command_execution" in workflow.triggers
    assert workflow.url_status == ow.URL_NO_TARGET


@pytest.mark.parametrize("prompt", EXECUTION)
def test_execution_requests_declare_shell_only_by_default(prompt):
    """Least privilege: shell, not arbitrary model-supplied python."""
    workflow = detect_operational_workflow(prompt)
    assert workflow.required_exact_tools == ("agent_mode:agent_run_shell",)


def test_explicitly_named_python_adds_python():
    prompt = "Run a python script to read every file under src/orchestrator/."
    workflow = detect_operational_workflow(prompt)
    assert workflow is not None
    assert set(workflow.required_exact_tools) == {
        "agent_mode:agent_run_shell",
        "agent_mode:agent_run_python",
    }


@pytest.mark.parametrize("prompt", EXECUTION)
def test_allowlist_is_populated_once_authorized(prompt):
    """The dead-end: allowlist was () regardless of authorization."""
    assert _allowlist(prompt, ALL_TRUE) == ("agent_run_shell",)


@pytest.mark.parametrize("prompt", EXECUTION)
def test_authorization_is_still_mandatory(prompt):
    """A capability request is not a grant: no flags -> still withheld."""
    assert _allowlist(prompt, ALL_FALSE) == ()
    assert _bound(prompt, ALL_FALSE) == []


@pytest.mark.parametrize("methodology", ["generic_engineering", "no_tool_response"])
def test_dead_end_methodologies_are_reachable_now(methodology):
    prompt = "Please run these diagnostic bash commands on the host."
    assert _allowlist(prompt, ALL_TRUE, methodology=methodology) == ("agent_run_shell",)
    assert _allowlist(prompt, ALL_FALSE, methodology=methodology) == ()


def test_repository_workflow_path_is_unchanged():
    """The pre-existing repo checkout/deploy detection must not regress."""
    prompt = (
        "Clone https://github.com/example/thing.git and deploy it locally to run the tests."
    )
    workflow = detect_operational_workflow(prompt)
    assert workflow is not None
    assert workflow.workflow == ow.REPO_CHECKOUT_LOCAL_DEPLOY
    assert "agent_mode:agent_git_clone" in workflow.required_exact_tools
