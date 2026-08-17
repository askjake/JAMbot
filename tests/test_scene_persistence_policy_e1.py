"""Phase E1 tests: Incident Scene persistence requires explicit intent.

Policy under test
-----------------
``build_incident_scene`` is nonpersistent end to end.  The upstream S3 tool
defaults to ``persist=False``, so Jake must never convert an omitted argument
into ``True``.  Persistence authorization is *permission*, not *intent*:

  * argument omitted  + persistence_authorized=False -> persist False
  * argument omitted  + persistence_authorized=True  -> persist False
  * explicit False                                   -> persist False
  * explicit True     + persistence_authorized=True  -> persist True
  * explicit True     + persistence_authorized=False -> blocked or rewritten False

``build_log_capsule`` keeps its documented upstream default of ``persist=True``
and must not regress.

Every tool here is synthetic.  No live, heavy, or persistent S3 tool is
invoked, no network call is made, and no artifact is written.
"""

from __future__ import annotations

import asyncio
import json

from app.agent.tool_execution_gate import (
    _ALWAYS_PERSIST_TOOLS,
    _PERSIST_DEFAULT_FALSE_TOOLS,
    _PERSIST_DEFAULT_TRUE_TOOLS,
    evaluate_tool_call,
    execute_gated_tool_calls,
)

SCENE = "build_incident_scene"
CAPSULE = "build_log_capsule"
BOUND = (SCENE, CAPSULE, "query_incident_scene", "get_summary")

ALL_FALSE = {
    "operator_authorized": False,
    "heavy_tools_authorized": False,
    "persistence_authorized": False,
    "mutation_authorized": False,
}


def flags(**kw):
    merged = dict(ALL_FALSE)
    merged.update(kw)
    return merged


def full_auth(**kw):
    return flags(
        operator_authorized=True,
        heavy_tools_authorized=True,
        persistence_authorized=True,
        **kw,
    )


def heavy_only(**kw):
    return flags(operator_authorized=True, heavy_tools_authorized=True, **kw)


def call(name, args=None, ident="call-e1"):
    return {"id": ident, "name": name, "args": dict(args or {})}


def decide(name, args=None, authorization_flags=None, bound=BOUND):
    return evaluate_tool_call(
        call(name, args),
        last_bound_tool_names=bound,
        authorization_flags=authorization_flags or ALL_FALSE,
    )


class RecordingTool:
    """Synthetic executor that records the arguments the gate actually sends."""

    def __init__(self, name):
        self.name = name
        self.seen = []

    async def ainvoke(self, args, config=None):
        self.seen.append(dict(args))
        return {"ok": True, "persist_received": args.get("persist", "<absent>")}


# --------------------------------------------------------------------------
# 1. omitted persistence remains false with all authorization false
# --------------------------------------------------------------------------
def test_scene_persist_omitted_all_authorization_false_is_never_true():
    decision = decide(SCENE, {"capsule_id": "cap-synth"}, ALL_FALSE)
    assert decision.effective_args.get("persist") is False
    # A heavy tool without heavy authorization must not execute at all.
    assert decision.allowed is False
    assert decision.result_code == "TOOL_AUTHORIZATION_REQUIRED"


# --------------------------------------------------------------------------
# 2. omitted persistence remains false with persistence authorization true
#    (the Phase E1 regression this suite exists to prevent)
# --------------------------------------------------------------------------
def test_scene_persist_omitted_with_persistence_authorized_stays_false():
    decision = decide(SCENE, {"capsule_id": "cap-synth"}, full_auth())
    assert decision.allowed is True
    assert decision.effective_args.get("persist") is False, (
        "persistence authorization alone must not request Scene persistence"
    )
    assert "persist" in decision.rewritten_arguments


def test_scene_persist_omitted_with_only_persistence_authorized_stays_false():
    decision = decide(SCENE, {"capsule_id": "cap-synth"}, flags(persistence_authorized=True))
    assert decision.effective_args.get("persist") is False


# --------------------------------------------------------------------------
# 3. explicit false remains false
# --------------------------------------------------------------------------
def test_scene_explicit_persist_false_remains_false():
    decision = decide(SCENE, {"capsule_id": "cap-synth", "persist": False}, full_auth())
    assert decision.allowed is True
    assert decision.effective_args.get("persist") is False


# --------------------------------------------------------------------------
# 4. explicit true succeeds only with required authorization
# --------------------------------------------------------------------------
def test_scene_explicit_persist_true_with_authorization_is_honored():
    decision = decide(SCENE, {"capsule_id": "cap-synth", "persist": True}, full_auth())
    assert decision.allowed is True
    assert decision.effective_args.get("persist") is True


# --------------------------------------------------------------------------
# 5. explicit true without persistence authorization is blocked or rewritten
# --------------------------------------------------------------------------
def test_scene_explicit_persist_true_without_persistence_authorization_is_safe():
    decision = decide(SCENE, {"capsule_id": "cap-synth", "persist": True}, heavy_only())
    assert decision.effective_args.get("persist") is False
    if decision.allowed:
        assert decision.result_code == "TOOL_EXECUTION_ALLOWED_WITH_REWRITE"
        assert "persist" in decision.rewritten_arguments


# --------------------------------------------------------------------------
# 6. heavy authorization remains required (gate not weakened)
# --------------------------------------------------------------------------
def test_scene_heavy_authorization_still_required():
    decision = decide(SCENE, {"capsule_id": "cap-synth", "persist": True}, flags(persistence_authorized=True))
    assert decision.allowed is False
    assert "heavy_tools_authorized" in decision.missing_authorizations
    # A blocked decision is never dispatched, so the safety property is that
    # the upstream tool is not invoked at all -- not that the rejected
    # argument bundle was rewritten.
    tool = RecordingTool(SCENE)
    asyncio.run(
        execute_gated_tool_calls(
            [call(SCENE, {"capsule_id": "cap-synth", "persist": True})],
            tools_by_name={SCENE: tool},
            last_bound_tool_names=BOUND,
            authorization_flags=flags(persistence_authorized=True),
        )
    )
    assert tool.seen == []


def test_scene_allow_heavy_only_injected_after_heavy_authorization():
    unauthorized = decide(SCENE, {"capsule_id": "cap-synth"}, ALL_FALSE)
    assert unauthorized.effective_args.get("allow_heavy") is not True
    authorized = decide(SCENE, {"capsule_id": "cap-synth"}, full_auth())
    assert authorized.effective_args.get("allow_heavy") is True


# --------------------------------------------------------------------------
# 7 / 8. same-turn and later-turn authorization behave identically:
#        the decision depends only on the flags in force for that call.
# --------------------------------------------------------------------------
def test_same_turn_authorization_does_not_imply_scene_persistence():
    same_turn = decide(SCENE, {"capsule_id": "cap-synth"}, full_auth())
    assert same_turn.allowed is True
    assert same_turn.effective_args.get("persist") is False


def test_later_turn_authorization_does_not_imply_scene_persistence():
    first = decide(SCENE, {"capsule_id": "cap-synth"}, ALL_FALSE)
    assert first.allowed is False
    later = decide(SCENE, {"capsule_id": "cap-synth"}, full_auth())
    assert later.allowed is True
    assert later.effective_args.get("persist") is False
    explicit_later = decide(SCENE, {"capsule_id": "cap-synth", "persist": True}, full_auth())
    assert explicit_later.effective_args.get("persist") is True


# --------------------------------------------------------------------------
# 9. revocation blocks later persistence
# --------------------------------------------------------------------------
def test_revocation_blocks_later_scene_persistence():
    granted = decide(SCENE, {"capsule_id": "cap-synth", "persist": True}, full_auth())
    assert granted.effective_args.get("persist") is True

    # After revocation the same explicit request is blocked outright, so no
    # durable write can occur even though the argument still says persist=True.
    revoked = decide(SCENE, {"capsule_id": "cap-synth", "persist": True}, ALL_FALSE)
    assert revoked.allowed is False
    assert revoked.result_code == "TOOL_AUTHORIZATION_REQUIRED"

    tool = RecordingTool(SCENE)
    asyncio.run(
        execute_gated_tool_calls(
            [call(SCENE, {"capsule_id": "cap-synth", "persist": True})],
            tools_by_name={SCENE: tool},
            last_bound_tool_names=BOUND,
            authorization_flags=ALL_FALSE,
        )
    )
    assert tool.seen == [], "a revoked binding must not reach the upstream tool"

    revoked_omitted = decide(SCENE, {"capsule_id": "cap-synth"}, ALL_FALSE)
    assert revoked_omitted.effective_args.get("persist") is False

    # Heavy-only authorization still executes, but persistence is downgraded.
    downgraded = decide(SCENE, {"capsule_id": "cap-synth", "persist": True}, heavy_only())
    assert downgraded.allowed is True
    assert downgraded.effective_args.get("persist") is False


# --------------------------------------------------------------------------
# 10. a stale binding cannot persist
# --------------------------------------------------------------------------
def test_stale_binding_cannot_persist_a_scene():
    decision = evaluate_tool_call(
        call(SCENE, {"capsule_id": "cap-synth", "persist": True}),
        last_bound_tool_names=("get_summary",),
        authorization_flags=full_auth(),
    )
    assert decision.allowed is False
    assert decision.result_code == "TOOL_NOT_IN_LAST_BINDING"


# --------------------------------------------------------------------------
# 11 / 12. parent behavior passes and a child cannot elevate persistence
# --------------------------------------------------------------------------
def test_child_clamped_flags_cannot_elevate_scene_persistence():
    from app.agent_mode.child_tool_policy import clamp_authorization_flags

    parent = full_auth()
    child = clamp_authorization_flags(parent)
    child_decision = evaluate_tool_call(
        call(SCENE, {"capsule_id": "cap-synth"}),
        last_bound_tool_names=BOUND,
        authorization_flags=child,
    )
    assert child_decision.effective_args.get("persist") is False

    child_explicit = evaluate_tool_call(
        call(SCENE, {"capsule_id": "cap-synth", "persist": True}),
        last_bound_tool_names=BOUND,
        authorization_flags=child,
    )
    if child.get("persistence_authorized"):
        assert child_explicit.effective_args.get("persist") is True
    else:
        assert child_explicit.effective_args.get("persist") is False


def test_child_cannot_exceed_parent_persistence_authorization():
    from app.agent_mode.child_tool_policy import clamp_authorization_flags

    child = clamp_authorization_flags(heavy_only())
    assert child.get("persistence_authorized", False) is False
    decision = evaluate_tool_call(
        call(SCENE, {"capsule_id": "cap-synth", "persist": True}),
        last_bound_tool_names=BOUND,
        authorization_flags=child,
    )
    assert decision.effective_args.get("persist") is False


# --------------------------------------------------------------------------
# 13. audit records argument field names but never values
# --------------------------------------------------------------------------
def test_audit_records_argument_names_but_not_values():
    decision = decide(
        SCENE,
        {"capsule_id": "cap-synth-secret", "persist": True},
        full_auth(),
    )
    audit = decision.audit
    assert audit["argument_values_recorded"] is False
    assert "capsule_id" in audit["argument_names"]
    blob = json.dumps(audit)
    assert "cap-synth-secret" not in blob
    assert "persist" in audit["argument_names"]


# --------------------------------------------------------------------------
# 14 / 15. existing persistent tools retain documented behavior
# --------------------------------------------------------------------------
def test_build_log_capsule_persistence_default_does_not_regress():
    assert CAPSULE in _PERSIST_DEFAULT_TRUE_TOOLS
    assert CAPSULE not in _PERSIST_DEFAULT_FALSE_TOOLS

    authorized = decide(CAPSULE, {"receiver_id": "SYNTH"}, full_auth())
    assert authorized.effective_args.get("persist") is True

    unauthorized = decide(CAPSULE, {"receiver_id": "SYNTH"}, heavy_only())
    assert unauthorized.effective_args.get("persist") is False

    explicit_false = decide(CAPSULE, {"receiver_id": "SYNTH", "persist": False}, full_auth())
    assert explicit_false.effective_args.get("persist") is False


def test_policy_sets_are_disjoint_and_scene_is_default_false():
    assert SCENE in _PERSIST_DEFAULT_FALSE_TOOLS
    assert SCENE not in _PERSIST_DEFAULT_TRUE_TOOLS
    assert SCENE not in _ALWAYS_PERSIST_TOOLS
    assert not (_PERSIST_DEFAULT_TRUE_TOOLS & _PERSIST_DEFAULT_FALSE_TOOLS)


# --------------------------------------------------------------------------
# 16. Scene canary: end-to-end executed call receives persist=False
# --------------------------------------------------------------------------
def test_executed_scene_call_sends_persist_false_to_upstream_tool():
    tool = RecordingTool(SCENE)
    audits = []
    messages = asyncio.run(
        execute_gated_tool_calls(
            [call(SCENE, {"capsule_id": "cap-synth"})],
            tools_by_name={SCENE: tool},
            last_bound_tool_names=BOUND,
            authorization_flags=full_auth(),
            audit_sink=audits.append,
        )
    )
    assert len(messages) == 1
    assert len(tool.seen) == 1
    assert tool.seen[0].get("persist") is False
    assert len(audits) == 1
    assert audits[0]["argument_values_recorded"] is False


def test_executed_scene_call_honors_explicit_persist_true():
    tool = RecordingTool(SCENE)
    messages = asyncio.run(
        execute_gated_tool_calls(
            [call(SCENE, {"capsule_id": "cap-synth", "persist": True})],
            tools_by_name={SCENE: tool},
            last_bound_tool_names=BOUND,
            authorization_flags=full_auth(),
        )
    )
    assert len(messages) == 1
    assert tool.seen[0].get("persist") is True


def test_blocked_scene_call_never_reaches_the_upstream_tool():
    tool = RecordingTool(SCENE)
    messages = asyncio.run(
        execute_gated_tool_calls(
            [call(SCENE, {"capsule_id": "cap-synth", "persist": True})],
            tools_by_name={SCENE: tool},
            last_bound_tool_names=BOUND,
            authorization_flags=ALL_FALSE,
        )
    )
    assert len(messages) == 1
    assert tool.seen == []


# --------------------------------------------------------------------------
# 17. compression / ToolMessage pairing remains exact
# --------------------------------------------------------------------------
def test_tool_message_pairing_is_exactly_one_per_call():
    scene_tool = RecordingTool(SCENE)
    capsule_tool = RecordingTool(CAPSULE)
    calls = [
        call(SCENE, {"capsule_id": "cap-a"}, ident="c1"),
        call(SCENE, {"capsule_id": "cap-b", "persist": True}, ident="c2"),
        call(CAPSULE, {"receiver_id": "SYNTH"}, ident="c3"),
        call("unbound_tool", {}, ident="c4"),
    ]
    messages = asyncio.run(
        execute_gated_tool_calls(
            calls,
            tools_by_name={SCENE: scene_tool, CAPSULE: capsule_tool},
            last_bound_tool_names=BOUND,
            authorization_flags=full_auth(),
        )
    )
    assert len(messages) == len(calls)
    ids = [getattr(m, "tool_call_id", None) for m in messages]
    assert ids == ["c1", "c2", "c3", "c4"]
    assert scene_tool.seen[0].get("persist") is False
    assert scene_tool.seen[1].get("persist") is True
