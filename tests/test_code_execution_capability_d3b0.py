"""Phase D3B0 tests: arbitrary-code-execution capability hardening.

A tool that runs model-supplied Python or shell is not read-only merely because
it is sandboxed by working directory. These tests prove that a parent or MCOP
child holding every authorization flag false can neither bind nor execute such a
tool, that parent and child consume one shared classifier, and that the audit
records capability and argument *names* only.

No model-supplied code is executed. Allowed-path assertions use a synthetic
no-op executor.
"""

from __future__ import annotations

import json

import pytest

from app.agent.tool_execution_audit import AUTHORIZATION_FLAG_KEYS
from app.agent.tool_execution_gate import (
    evaluate_tool_call,
    required_authorizations_for_tool,
    tool_is_permitted_by_flags,
)
from app.agent.tool_execution_policy import get_scoped_tools_for_prompt
from app.agent.tool_profiles import (
    CODE_EXECUTION_FALLBACK_REQUIREMENTS,
    CODE_EXECUTION_RISK,
    CODE_EXECUTION_TOOL_CAPABILITIES,
    code_execution_capability,
    is_code_execution_tool,
)
from app.agent_mode.child_tool_policy import (
    ChildToolPolicy,
    child_safe_tools,
    narrow_policy_for_task,
)
from app.agent.tool_execution_audit import SNAPSHOT_ACCEPTED

CODE_EXEC = "agent_run_python"
SHELL_EXEC = "agent_run_shell"
READ_ONLY_AGENT = "agent_list_artifacts"

ALL_FALSE = dict.fromkeys(AUTHORIZATION_FLAG_KEYS, False)
ALL_TRUE = dict.fromkeys(AUTHORIZATION_FLAG_KEYS, True)


class FakeTool:
    """Synthetic no-op executor. Never runs model-supplied code."""

    def __init__(self, name: str):
        self.name = name
        self.description = "synthetic"
        self.args_schema = None
        self.invocations = 0

    def invoke(self, args):  # pragma: no cover - must never run in these tests
        self.invocations += 1
        return "synthetic-no-op"


def _policy(**kw) -> ChildToolPolicy:
    flags = kw.pop("flags", ALL_FALSE)
    return ChildToolPolicy(
        authorization_flags=tuple(sorted(dict(flags).items())),
        eligible_toolsets=tuple(kw.pop("toolsets", ())),
        eligible_extra_tools=tuple(kw.pop("extras", ())),
        registry_generation=kw.pop("generation", "gen-d3b0"),
        profile_signature=kw.pop("signature", "sha256:sig"),
        parent_request_id=kw.pop("parent_request_id", "req-parent"),
        parent_thread_digest=kw.pop("digest", "thr-d3b0"),
        child_run_id=kw.pop("child_run_id", "child-d3b0"),
        snapshot_status=kw.pop("status", SNAPSHOT_ACCEPTED),
    )


def _bound_names(flags, prompt="Please summarise the attached document."):
    tools, _plan = get_scoped_tools_for_prompt(prompt, authorization_flags=dict(flags))
    return {str(getattr(t, "name", "")) for t in tools}


# ------------------------------------------------------- 1. not eligible
def test_all_false_code_execution_is_not_eligible():
    assert required_authorizations_for_tool(CODE_EXEC)
    assert not tool_is_permitted_by_flags(CODE_EXEC, ALL_FALSE)
    assert not tool_is_permitted_by_flags(SHELL_EXEC, ALL_FALSE)


# ------------------------------------------------------- 2. not model-bound
def test_all_false_code_execution_is_not_model_bound():
    names = _bound_names(ALL_FALSE)
    assert CODE_EXEC not in names
    assert SHELL_EXEC not in names
    assert "agent_create_venv" not in names
    assert "agent_git_clone" not in names


def test_code_execution_returns_when_fully_authorized():
    # D3B2A narrows this D3B0 contract deliberately.  Authorization *permits*
    # a requested privileged tool; it must never, by itself, bind every
    # privileged executor into a chat that never asked for one.  The
    # capability must still return for a task that genuinely requires it.
    generic = _bound_names(ALL_TRUE)
    assert CODE_EXEC not in generic
    assert SHELL_EXEC not in generic

    operational = _bound_names(
        ALL_TRUE,
        prompt=(
            "Use the configured SSH identity to clone "
            "https://gitlab.com/example-group/example-app then deploy and "
            "run it locally in an isolated environment, and run a python "
            "script to verify it."
        ),
    )
    assert CODE_EXEC in operational
    assert SHELL_EXEC in operational


# ------------------------------------------------------- 3. gate blocks
def test_gate_blocks_synthetic_code_execution_call_when_all_false():
    decision = evaluate_tool_call(
        {"id": "c1", "name": CODE_EXEC, "args": {"chat_id": "x", "code": "print(1)"}},
        last_bound_tool_names=[CODE_EXEC],
        authorization_flags=dict(ALL_FALSE),
    )
    assert decision.allowed is False
    assert decision.result_code == "TOOL_AUTHORIZATION_REQUIRED"
    assert set(decision.missing_authorizations) == set(decision.required_authorizations)
    assert decision.audit["risk"] == CODE_EXECUTION_RISK


def test_gate_blocks_even_when_not_in_binding():
    decision = evaluate_tool_call(
        {"id": "c2", "name": CODE_EXEC, "args": {"code": "print(1)"}},
        last_bound_tool_names=[],
        authorization_flags=dict(ALL_FALSE),
    )
    assert decision.allowed is False
    assert decision.result_code == "TOOL_NOT_IN_LAST_BINDING"


# ------------------------------------------------------- 4. child not bound
def test_child_all_false_cannot_bind_code_execution():
    policy = _policy(flags=ALL_FALSE, extras=(f"agent_mode:{CODE_EXEC}", "s3_stb_logs:list_dates"))
    kept = [t.name for t in child_safe_tools([FakeTool(CODE_EXEC), FakeTool("list_dates")], policy)]
    assert CODE_EXEC not in kept
    assert "list_dates" in kept


def test_child_fully_authorized_may_bind_code_execution():
    policy = _policy(flags=ALL_TRUE, extras=(f"agent_mode:{CODE_EXEC}",))
    kept = [t.name for t in child_safe_tools([FakeTool(CODE_EXEC)], policy)]
    assert kept == [CODE_EXEC]


# ------------------------------------------------------- 5. no task elevation
def test_child_cannot_elevate_code_execution_through_task_wording():
    policy = _policy(flags=ALL_FALSE, extras=(f"agent_mode:{CODE_EXEC}",))
    narrowed = narrow_policy_for_task(
        policy,
        task_required_tools=[f"agent_mode:{CODE_EXEC}"],
    )
    assert narrowed.flags == ALL_FALSE
    kept = [
        t.name
        for t in child_safe_tools([FakeTool(CODE_EXEC)], narrowed, task_required_names=[CODE_EXEC])
    ]
    assert kept == []


# ------------------------------------------------------- 6/7/8. partial flags
@pytest.mark.parametrize(
    "flags",
    [
        {**ALL_FALSE, "heavy_tools_authorized": True},
        {**ALL_FALSE, "operator_authorized": True},
        {**ALL_FALSE, "operator_authorized": True, "heavy_tools_authorized": True},
        {
            **ALL_FALSE,
            "operator_authorized": True,
            "heavy_tools_authorized": True,
            "mutation_authorized": True,
        },
    ],
)
def test_partial_authorization_remains_blocked(flags):
    assert not tool_is_permitted_by_flags(CODE_EXEC, flags)
    assert CODE_EXEC not in _bound_names(flags)


def test_operator_plus_heavy_is_sufficient_only_when_nothing_else_required():
    required = set(required_authorizations_for_tool(CODE_EXEC))
    op_heavy = {"operator_authorized", "heavy_tools_authorized"}
    # agent_run_python needs strictly more than operator+heavy.
    assert required - op_heavy
    flags = {**ALL_FALSE, **{k: True for k in op_heavy}}
    assert not tool_is_permitted_by_flags(CODE_EXEC, flags)
    # A tool requiring exactly operator+heavy is permitted by exactly those.
    assert all(
        tool_is_permitted_by_flags(name, flags)
        for name, meta in CODE_EXECUTION_TOOL_CAPABILITIES.items()
        if set(meta["required_authorizations"]) <= op_heavy
    )


# ------------------------------------------------------- 9/10. mutation+persist
def test_mutation_is_required_because_implementation_writes_and_spawns():
    meta = code_execution_capability(CODE_EXEC)
    assert "mutation_authorized" in meta["required_authorizations"]
    assert {"filesystem_write", "process_spawn"} <= set(meta["capabilities"])


def test_persistence_is_required_because_implementation_persists_artifacts():
    meta = code_execution_capability(CODE_EXEC)
    assert "persistence_authorized" in meta["required_authorizations"]
    assert "persistent_artifacts" in meta["capabilities"]


def test_operator_and_heavy_are_the_minimum_floor():
    for name in (CODE_EXEC, SHELL_EXEC, "agent_create_venv"):
        required = set(required_authorizations_for_tool(name))
        assert {"operator_authorized", "heavy_tools_authorized"} <= required


# ------------------------------------------------------- 11. facade honesty
def test_management_facade_reports_accurate_requirements():
    from app.agent.agents.tools.management import diship_backend_activate_tool_binding

    payload = diship_backend_activate_tool_binding.invoke(
        {"tool_names": f"agent_mode:{CODE_EXEC}", "toolsets": ""}
    )
    assert payload["activation_performed"] is False
    assert payload["execution_performed"] is False
    by_tool = payload["required_authorizations_by_tool"]
    assert by_tool, "facade must report per-tool requirements"
    reported = next(iter(by_tool.values()))
    assert set(reported) == set(required_authorizations_for_tool(CODE_EXEC))
    assert payload["code_execution_tools_requested"]


# ------------------------------------------------------- 12/13/14. audit hygiene
def test_audit_reports_capability_names_only():
    secret_code = "import os; os.environ['LEAKED']='yes'"
    decision = evaluate_tool_call(
        {"id": "c3", "name": CODE_EXEC, "args": {"chat_id": "abc", "code": secret_code}},
        last_bound_tool_names=[CODE_EXEC],
        authorization_flags=dict(ALL_FALSE),
    )
    blob = json.dumps(decision.audit)
    assert "arbitrary_code_execution" in blob
    assert sorted(decision.audit["argument_names"]) == ["chat_id", "code"]
    # argument VALUES must never appear
    assert secret_code not in blob
    assert "LEAKED" not in blob
    assert "abc" not in blob


def test_audit_contains_no_result_body():
    decision = evaluate_tool_call(
        {"id": "c4", "name": CODE_EXEC, "args": {"code": "print('body')"}},
        last_bound_tool_names=[CODE_EXEC],
        authorization_flags=dict(ALL_FALSE),
    )
    blob = json.dumps(decision.audit)
    for forbidden in ("stdout", "STDOUT", "body", "return code"):
        assert forbidden not in blob


# ------------------------------------------------------- 15. one classifier
def test_parent_and_child_use_the_same_classifier():
    from app.agent import tool_execution_gate as gate
    from app.agent import tool_execution_policy as policy_mod
    import app.agent_mode.child_tool_policy as child_mod
    import inspect

    # the gate resolves code-execution capability from tool_profiles
    assert "code_execution_capability" in inspect.getsource(gate._capability_metadata)
    # the parent binding path consults the shared classifier
    assert "is_code_execution_tool" in inspect.getsource(policy_mod.get_tools_for_toolsets)
    # the child consults the same gate-exported predicate
    assert "tool_is_permitted_by_flags" in inspect.getsource(child_mod.child_safe_tools)
    # and both agree for every enumerated tool
    for name in CODE_EXECUTION_TOOL_CAPABILITIES:
        assert set(required_authorizations_for_tool(name)) == set(
            code_execution_capability(name)["required_authorizations"]
        )


# ------------------------------------------------------- 16. read-only intact
def test_other_read_only_tools_remain_unaffected():
    assert not is_code_execution_tool(READ_ONLY_AGENT)
    assert required_authorizations_for_tool(READ_ONLY_AGENT) == ()
    names = _bound_names(ALL_FALSE)
    assert READ_ONLY_AGENT in names
    for benign in ("search_logs", "list_dates", "get_summary", "run_complete_corpus_job_batch"):
        assert not is_code_execution_tool(benign), benign


def test_heavy_s3_tools_keep_their_existing_classification():
    assert required_authorizations_for_tool("build_log_capsule") == ("heavy_tools_authorized",)
    assert not is_code_execution_tool("build_log_capsule")


# ------------------------------------------------------- 17. fail closed
@pytest.mark.parametrize(
    "unknown",
    [
        "mystery_run_python_tool",
        "vendor_exec_code",
        "sandbox_exec_helper",
        "nb_notebook_exec",
        "svc_eval_code_v2",
    ],
)
def test_unknown_code_execution_tools_fail_closed(unknown):
    meta = code_execution_capability(unknown)
    assert meta is not None, unknown
    assert meta["classification"] == "fail_closed_pattern"
    assert set(meta["required_authorizations"]) == set(CODE_EXECUTION_FALLBACK_REQUIREMENTS)
    assert not tool_is_permitted_by_flags(unknown, ALL_FALSE)


def test_fail_closed_pattern_does_not_capture_benign_names():
    for benign in (
        "list_dates",
        "search_logs",
        "get_timeline",
        "agent_list_artifacts",
        "run_complete_corpus_job_batch",
        "read_parsed_logs",
        "diagnose_receiver_from_logs",
    ):
        assert code_execution_capability(benign) is None, benign


# ------------------------------------------------------- 18. D2 bounds hold
def test_d2_family_upper_bound_still_applies_to_code_execution():
    # parent never granted the agent_mode family -> child cannot get it even
    # when fully authorized
    policy = _policy(flags=ALL_TRUE, toolsets=("s3_stb_logs",), extras=("s3_stb_logs:list_dates",))
    kept = [t.name for t in child_safe_tools([FakeTool(CODE_EXEC), FakeTool("list_dates")], policy)]
    assert CODE_EXEC not in kept


def test_d2_exact_tool_upper_bound_still_applies_to_code_execution():
    policy = _policy(flags=ALL_TRUE, extras=("s3_stb_logs:list_dates",))
    kept = [t.name for t in child_safe_tools([FakeTool(CODE_EXEC), FakeTool("list_dates")], policy)]
    assert CODE_EXEC not in kept
    assert kept == ["list_dates"]


def test_no_synthetic_tool_was_ever_invoked():
    tool = FakeTool(CODE_EXEC)
    policy = _policy(flags=ALL_FALSE, extras=(f"agent_mode:{CODE_EXEC}",))
    child_safe_tools([tool], policy)
    assert tool.invocations == 0
