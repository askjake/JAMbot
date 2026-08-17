from __future__ import annotations

from app.agent.operational_identity import (
    OperationalIdentity,
    typed_host_attribution,
)
from app.agent.tool_execution_audit import build_event


def test_remote_tool_endpoint_cannot_establish_or_override_self_location():
    identity = OperationalIdentity(
        user_pinned_target="10.79.85.35",
        runtime_hostname="dsgpu3090-Lambda-Vector",
    )
    identity = identity.observe_endpoint("mcp_server_host", "http://10.79.85.47:8765/mcp")
    assert identity.self_location_established is True
    assert identity.authoritative_target == "10.79.85.35"
    assert identity.endpoint_observations["mcp_server_host"] == "10.79.85.47:8765"
    assert identity.endpoint_can_establish_self_location is False


def test_pinned_target_without_successful_runtime_identity_is_not_self_location():
    identity = OperationalIdentity(user_pinned_target="10.79.85.35")
    assert identity.self_location_established is False
    assert identity.self_location_basis == "PINNED_TARGET_RUNTIME_IDENTITY_PENDING"


def test_runtime_identity_on_pinned_target_establishes_self_location():
    identity = OperationalIdentity(
        user_pinned_target="10.79.85.35",
        runtime_hostname="dsgpu3090-Lambda-Vector",
    )
    assert identity.self_location_established is True
    assert identity.self_location_basis == "USER_PINNED_TARGET_PLUS_RUNTIME_HOSTNAME"


def test_repository_claim_requires_typed_checkout_identity():
    identity = OperationalIdentity(
        user_pinned_target="10.79.85.35",
        runtime_hostname="dsgpu3090-Lambda-Vector",
        repository_checkout_host="dsgpu3090-Lambda-Vector",
        repository_path="/home/jakebot/Jakes-agent",
        repository_origin_host="git.dtc.dish.corp",
        repository_branch="montjac",
        repository_head="ab12c0bc983e156c38a4c0497509299d93ecd9b7",
    )
    assert identity.repository_identity_established is True


def test_http_target_is_host_only_and_drops_path_query_and_credentials():
    attrs = typed_host_attribution(
        {
            "http_target": "https://user:password@example.internal:8443/private?token=secret",
            "remote_service_host": "service.internal",
        }
    )
    assert attrs["http_target"] == "example.internal:8443"
    assert "password" not in str(attrs)
    assert "token" not in str(attrs)


def test_tool_audit_contains_all_typed_host_fields():
    attrs = typed_host_attribution(
        {
            "agent_execution_host": "dsgpu3090-Lambda-Vector",
            "tool_executor_host": "lambda-executor",
            "remote_service_host": "grasshopper-autoupload.dishanywhere.com:8443",
            "http_target": "https://example.internal/mcp",
            "source_client_address": "165.225.10.217",
            "mcp_server_host": "s3-stb-logs-mcp",
            "repository_checkout_host": "dsgpu3090-Lambda-Vector",
            "repository_path": "/home/jakebot/Jakes-agent",
            "repository_origin_host": "git.dtc.dish.corp",
            "self_location_basis": "USER_PINNED_TARGET_PLUS_RUNTIME_HOSTNAME",
        }
    )
    event = build_event(**attrs)
    for field in (
        "agent_execution_host",
        "tool_executor_host",
        "remote_service_host",
        "http_target",
        "source_client_address",
        "mcp_server_host",
        "repository_checkout_host",
        "repository_path",
        "repository_origin_host",
        "self_location_basis",
        "endpoint_can_establish_self_location",
    ):
        assert field in event
    assert event["endpoint_can_establish_self_location"] is False
