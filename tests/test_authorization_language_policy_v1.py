"""Authorization-language policy tests.

These tests enforce section 12 of the Phase C contract: the assistant must not
fabricate authorization-derived material, and must not conflate an activation
*request* with activation or execution.

The server-side audit remains the primary control.  These tests guard the
secondary control (prompt + facade semantics).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = REPO_ROOT / "app/agent/agents/prompts/chat_system_prompt.txt"


@pytest.fixture(scope="module")
def prompt_text() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


REQUIRED_PHRASES = (
    "tool_execution_authorization_language",
    "Your prose is NOT execution evidence",
    "Never invent, guess, echo, reconstruct, or display an authorization token",
    "token hash, digest, checksum, prefix, suffix",
    "ACTIVATION_REQUESTED",
    "ACTIVATION_PENDING",
    "ACTIVATION_ELIGIBLE",
    "EXECUTION_BLOCKED",
    "EXECUTION_ALLOWED",
    "EXECUTION_COMPLETED",
    "Requesting activation is not activation",
    "server-provided booleans",
)


def test_prompt_declares_required_authorization_rules(prompt_text):
    missing = [phrase for phrase in REQUIRED_PHRASES if phrase not in prompt_text]
    assert not missing, f"system prompt is missing required rules: {missing}"


def test_prompt_states_assistant_prose_is_not_authoritative(prompt_text):
    assert "audit" in prompt_text.lower()
    assert "not execution evidence" in prompt_text.lower()


def test_prompt_forbids_claiming_unverified_activation_progress(prompt_text):
    for word in ("submitted", "in flight", "executing"):
        assert word in prompt_text, f"prompt must name the forbidden claim {word!r}"
    assert "unless a server-provided status value" in prompt_text


# --------------------------------------------------------------------------
# Fabricated authorization-derived material must be absent
# --------------------------------------------------------------------------
_FABRICATED_MATERIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("long_hex_digest", re.compile(r"\b[0-9a-fA-F]{32,}\b")),
    ("bearer_literal", re.compile(r"Bearer\s+[A-Za-z0-9._\-]{12,}")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9]{16,}")),
    ("gitlab_token", re.compile("gl" + r"pat-[A-Za-z0-9_.-]{15,}")),
    ("google_api_key", re.compile("AI" + r"za[0-9A-Za-z_-]{30,}")),
    ("token_assignment", re.compile(r"(?:auth[_-]?token|heavy_auth_token|authorization_token)\s*[:=]\s*[\"']?[A-Za-z0-9._\-]{8,}")),
    ("hash_assignment", re.compile(r"(?:token_hash|auth_hash|token_digest)\s*[:=]\s*[\"']?[A-Za-z0-9]{8,}")),
    ("token_prefix_claim", re.compile(r"token\s+(?:prefix|suffix)\s+is\s+[A-Za-z0-9]{4,}", re.IGNORECASE)),
)


def test_prompt_contains_no_fabricated_authorization_material(prompt_text):
    findings = [name for name, pattern in _FABRICATED_MATERIAL_PATTERNS if pattern.search(prompt_text)]
    assert not findings, f"prompt contains fabricated authorization-derived material: {findings}"


def test_authorization_language_policy_document_has_no_fabricated_material():
    doc = REPO_ROOT / "docs/AUTHORIZATION_LANGUAGE_POLICY.md"
    assert doc.is_file(), "AUTHORIZATION_LANGUAGE_POLICY.md must ship with the policy"
    text = doc.read_text(encoding="utf-8")
    findings = [name for name, pattern in _FABRICATED_MATERIAL_PATTERNS if pattern.search(text)]
    assert not findings, f"policy document contains fabricated authorization material: {findings}"
    assert "Requesting activation is not activation" in text or "request is not" in text.lower()


# --------------------------------------------------------------------------
# Facade semantics
# --------------------------------------------------------------------------
def _facade_registry(*, include_scene_tool: bool):
    from app.agent.tool_activation_intent import RegistryToolIndex

    inventory = []
    family_status = {}
    if include_scene_tool:
        inventory = [
            {
                "toolset": "s3_stb_logs",
                "tool_name": "build_incident_scene",
                "enabled": True,
            }
        ]
        family_status = {"s3_stb_logs": "HEALTHY"}
    return RegistryToolIndex.from_material(
        inventory=inventory,
        family_status=family_status,
    )


def test_activation_facade_reports_request_not_activation(monkeypatch):
    import app.agent.agents.tools.management as management

    registry = _facade_registry(include_scene_tool=True)
    monkeypatch.setattr(
        management.RegistryToolIndex,
        "live",
        classmethod(lambda cls: registry),
    )

    payload = management.diship_backend_activate_tool_binding.invoke(
        {"tool_names": "build_incident_scene", "toolsets": "s3_stb_logs"}
    )
    assert payload["activation_status"] == "REQUESTED"
    assert payload["request_recorded"] is False
    assert payload["recording_deferred_to_orchestration_hook"] is True
    assert payload["activation_performed"] is False
    assert payload["execution_performed"] is False
    assert payload["authorization_material_accepted"] is False
    assert payload["write_performed"] is False

    serialized = json.dumps(payload)
    findings = [name for name, pattern in _FABRICATED_MATERIAL_PATTERNS if pattern.search(serialized)]
    assert not findings, f"facade payload contains authorization-derived material: {findings}"


def test_activation_facade_pending_registry_is_still_request_only(monkeypatch):
    import app.agent.agents.tools.management as management

    registry = _facade_registry(include_scene_tool=False)
    monkeypatch.setattr(
        management.RegistryToolIndex,
        "live",
        classmethod(lambda cls: registry),
    )

    payload = management.diship_backend_activate_tool_binding.invoke(
        {"tool_names": "build_incident_scene", "toolsets": "s3_stb_logs"}
    )
    assert payload["activation_status"] == "PENDING_REGISTRY"
    assert payload["requested_toolsets"] == ["s3_stb_logs"]
    assert "s3_stb_logs" in payload["pending_registry_requests"]
    assert payload["request_recorded"] is False
    assert payload["activation_performed"] is False
    assert payload["execution_performed"] is False
    assert payload["write_performed"] is False


def test_activation_facade_docstring_denies_activation():
    from app.agent.agents.tools.management import diship_backend_activate_tool_binding

    description = str(diship_backend_activate_tool_binding.description or "")
    assert "does not activate" in description.lower()
    assert "token" in description.lower()


def test_management_facades_never_expose_authorization_arguments():
    from app.agent.agents.tools.management import BACKEND_MANAGEMENT_FACADES

    forbidden = {"heavy_auth_token", "operator_auth_token", "authorization_token", "mutation_auth_token"}
    for facade in BACKEND_MANAGEMENT_FACADES:
        schema = getattr(facade, "args_schema", None)
        fields = set(getattr(schema, "model_fields", {}) or {})
        assert not (fields & forbidden), f"{facade.name} exposes authorization arguments"


# --------------------------------------------------------------------------
# Parser never retains material
# --------------------------------------------------------------------------
def test_authorization_parser_records_booleans_only():
    from app.agent.tool_authorization import authorization_state_for_turn

    state = authorization_state_for_turn(
        None,
        "Heavy tools authorized. Persistence authorized. My token is CANARY-TOKEN-VALUE-77.",
    )
    assert state["authorization_flags"]["heavy_tools_authorized"] is True
    assert state["authorization_flags"]["persistence_authorized"] is True
    assert state["authorization_material_stored"] is False
    serialized = json.dumps(state)
    assert "CANARY-TOKEN-VALUE-77" not in serialized

    negative = authorization_state_for_turn(
        state["authorization_flags"],
        "Heavy tools are not authorized. Persistence is not authorized.",
    )
    assert negative["authorization_flags"]["heavy_tools_authorized"] is False
    assert negative["authorization_flags"]["persistence_authorized"] is False
