"""D3B1: health/selftest must separate registry content, health, and lifecycle."""

from __future__ import annotations

import asyncio
import json

from app.agent import mcp_registry_health as rh
from app.health.router import tool_registry_identity_payload


def test_identity_payload_separates_content_health_and_lifecycle():
    payload = tool_registry_identity_payload()
    assert payload["status"] == "pass"
    assert payload["schema"] == "diship_tool_registry_identity.v1"
    assert payload["canonicalization_version"] == "registry_signature.v2"
    assert payload["lkg_schema_version"] == rh.LKG_SCHEMA_VERSION
    assert str(payload["registry_content_signature"]).startswith("sha256:")
    assert isinstance(payload["registry_refresh_epoch"], int)
    assert str(payload["fixed_policy_profile_signature"]).startswith("sha256:")
    assert str(payload["model_facing_schema_digest"]).startswith("sha256:")
    # Content identity is not the epoch, and not the health signature.
    assert payload["registry_content_signature"] != str(payload["registry_refresh_epoch"])
    assert payload["registry_content_signature"] != payload["registry_health_signature"]


def test_identity_payload_reports_per_family_health_and_source():
    payload = tool_registry_identity_payload()
    for name, state in (payload["families"] or {}).items():
        assert state["health"] in rh.FAMILY_HEALTH_STATES
        assert state["error_class"] in rh.SAFE_ERROR_CLASSES
        assert "url" not in json.dumps(state).lower()


def test_identity_payload_contains_no_transport_material():
    blob = json.dumps(tool_registry_identity_payload())
    assert "https://" not in blob
    assert "Authorization" not in blob
    assert "Bearer " not in blob


def test_identity_payload_is_read_only():
    payload = tool_registry_identity_payload()
    assert payload["write_performed"] is False


def test_identity_route_is_registered():
    from app.health.router import router

    paths = {getattr(route, "path", "") for route in router.routes}
    assert "/health/tool-registry-identity" in paths


def test_identity_payload_is_stable_within_one_process():
    first = tool_registry_identity_payload()
    second = tool_registry_identity_payload()
    assert first["registry_content_signature"] == second["registry_content_signature"]
    assert first["fixed_policy_profile_signature"] == second["fixed_policy_profile_signature"]
    assert first["model_facing_schema_digest"] == second["model_facing_schema_digest"]
