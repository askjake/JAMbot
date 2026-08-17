"""
Bedrock-DishIP Parity Tests
Tests added 2026-07-22 as part of the methodology parity enhancement.
Covers: explicit directive parsing, remote_host_network_usb_triage routing,
log_acquisition_preview routing, ollama_backend_readiness routing,
ssh_host_key_audit, ssh_remote_service_inspection, and safety guards.
"""

import pytest
from app.agent.methodology import select_methodology, METHODOLOGY_TEMPLATES, methodology_selftest


# ---------------------------------------------------------------------------
# Test Case A: Explicit methodology directive parsing
# ---------------------------------------------------------------------------

class TestExplicitDirectiveParsing:
    """Requirement: 'Required methodology: <name>' overrides all keyword routing."""

    def test_directive_overrides_keyword_match(self):
        prompt = "Required methodology: remote_host_network_usb_triage; target: montjac@10.79.85.47"
        result = select_methodology(prompt)
        assert result["name"] == "remote_host_network_usb_triage"
        assert result["selection_reason"] == "explicit_directive"

    def test_directive_case_insensitive(self):
        prompt = "required methodology: log_acquisition_preview. Please check receiver R1955706171."
        result = select_methodology(prompt)
        assert result["name"] == "log_acquisition_preview"

    def test_directive_with_receiver_log_keywords_preserves_directive(self):
        """Even if 'logs' + receiver ID present, explicit directive takes priority."""
        prompt = (
            "Required methodology: log_acquisition_preview. "
            "Acquisition preview for R1955706171 fresh NAL logs. Do not upload."
        )
        result = select_methodology(prompt)
        assert result["name"] == "log_acquisition_preview"

    def test_directive_for_unknown_name_falls_through(self):
        """Unknown methodology names in directive should fall through to keyword routing."""
        prompt = "Required methodology: nonexistent_methodology_xyz; check R1234567890 logs"
        result = select_methodology(prompt)
        assert result["name"] != "nonexistent_methodology_xyz"

    def test_directive_backend_readiness(self):
        prompt = "Required methodology: ollama_backend_readiness. Check backend status."
        result = select_methodology(prompt)
        assert result["name"] == "ollama_backend_readiness"

    def test_directive_generic_engineering(self):
        prompt = "Required methodology: generic_engineering. Just help with an engineering question."
        result = select_methodology(prompt)
        assert result["name"] == "generic_engineering"


# ---------------------------------------------------------------------------
# Test Case F: remote_host_network_usb_triage
# ---------------------------------------------------------------------------

class TestRemoteHostNetworkUsbTriage:
    """New methodology not present in diship-test reference."""

    def test_network_interface_down_with_ssh_target_routes_to_triage(self):
        prompt = (
            "Investigate host-level network triage for montjac@10.79.85.47; "
            "interface enp68s0 state DOWN, USB mouse not working, investigate"
        )
        result = select_methodology(prompt)
        assert result["name"] == "remote_host_network_usb_triage"

    def test_bridge_stp_port_security_routes_to_triage(self):
        prompt = (
            "Investigate 10.79.85.47 for suspected spanning tree / port-security shutdown "
            "on the switch port. NIC appears down. Investigate and triage."
        )
        result = select_methodology(prompt)
        assert result["name"] == "remote_host_network_usb_triage"

    def test_usb_keyboard_issue_with_ssh_target(self):
        prompt = (
            "SSH to montjac@10.79.85.47 and diagnose why the USB keyboard is not working "
            "in the Ubuntu GUI."
        )
        result = select_methodology(prompt)
        assert result["name"] == "remote_host_network_usb_triage"

    def test_network_triage_does_not_match_port_service_inspection(self):
        """Single port/service inspection must not trigger triage."""
        prompt = "Use SSH to inspect the process listening on port 8510 at montjac@10.79.85.47"
        result = select_methodology(prompt)
        assert result["name"] == "ssh_remote_service_inspection"
        assert result["name"] != "remote_host_network_usb_triage"

    def test_network_triage_not_triggered_without_ssh_target(self):
        """Must have SSH target to route to triage."""
        prompt = "investigate network interface down state and USB not working"
        result = select_methodology(prompt)
        # Without a valid SSH target it should NOT route to remote_host_network_usb_triage
        assert result["name"] != "remote_host_network_usb_triage"

    def test_explicit_directive_for_triage(self):
        prompt = (
            "Required methodology: remote_host_network_usb_triage; "
            "ssh_target: montjac@10.79.85.47; "
            "interface: enp68s0; "
            "observed_state: state DOWN; "
            "suspected_cause: router/switch port shutdown (bridge/loop/port-security); "
            "usb_state: USB mouse and keyboard do not work; "
            "Read-only only. Do not mutate. Do not reboot."
        )
        result = select_methodology(prompt)
        assert result["name"] == "remote_host_network_usb_triage"
        assert result["selection_reason"] == "explicit_directive"

    def test_triage_methodology_template_has_required_fields(self):
        tmpl = METHODOLOGY_TEMPLATES["remote_host_network_usb_triage"]
        assert "agent_mode" in tmpl.required_tool_families
        assert "ssh_target" in tmpl.required_evidence_fields
        assert "interface" in tmpl.required_evidence_fields
        assert "observed_state" in tmpl.required_evidence_fields
        # Must classify result
        assert "classification" in tmpl.required_final_classification_fields
        assert "operator_actions" in tmpl.required_final_classification_fields


# ---------------------------------------------------------------------------
# Test Case B: log_acquisition_preview
# ---------------------------------------------------------------------------

class TestLogAcquisitionPreview:
    """log_acquisition_preview must not mutate and must be distinct from RCA."""

    def test_acquisition_preview_keyword_routes_correctly(self):
        prompt = "Acquisition preview for receiver R1955706171 fresh NAL logs; do not upload."
        result = select_methodology(prompt)
        assert result["name"] == "log_acquisition_preview"

    def test_log_preview_explicit_directive(self):
        prompt = "Required methodology: log_acquisition_preview. Receiver R1234567890. Preview only."
        result = select_methodology(prompt)
        assert result["name"] == "log_acquisition_preview"

    def test_log_acquisition_does_not_trigger_for_popup_prompt(self):
        """Popup investigation must not be overridden by receiver+logs keywords."""
        prompt = (
            "Joey R2200001234 signal lost popup during live TV around 2:30pm yesterday. "
            "Hopper R1100005678. Pull S3 STB logs and identify exactly which popup was displayed."
        )
        result = select_methodology(prompt)
        assert result["name"] == "popup_signal_loss_investigation"
        assert result["name"] != "log_acquisition_preview"

    def test_log_acquisition_does_not_trigger_for_reboot_prompt(self):
        """DVR reboot RCA must not be overridden by log keyword."""
        prompt = (
            "Receiver R1911746693 is repeatedly rebooting during DVR playback around 9pm on U820; "
            "check S3 logs and RTR alerts"
        )
        result = select_methodology(prompt)
        assert result["name"] == "receiver_reboot_dvr_playback"
        assert result["name"] != "log_acquisition_preview"

    def test_log_acquisition_execute_directive(self):
        prompt = "Required methodology: log_acquisition_execute. Execute log acquisition for R1234567890."
        result = select_methodology(prompt)
        assert result["name"] == "log_acquisition_execute"

    def test_log_acquisition_preview_template_has_mutation_fields(self):
        tmpl = METHODOLOGY_TEMPLATES["log_acquisition_preview"]
        assert "s3_stb_logs" in tmpl.required_tool_families
        assert "receiver_id" in tmpl.required_evidence_fields
        assert "requested_log_family" in tmpl.required_evidence_fields
        assert "mutation_performed" in tmpl.required_final_classification_fields


# ---------------------------------------------------------------------------
# Test Case A: Ollama/backend readiness
# ---------------------------------------------------------------------------

class TestOllamaBackendReadiness:

    def test_ollama_readiness_keyword_routes_correctly(self):
        for prompt in [
            "Check ollama backend readiness and model roles status",
            "Inspect local Ollama Agent Mode readiness and ollama-agent-selftest",
            "Check model roles and tool-worker model status",
            "ollama agent mode readiness check",
        ]:
            result = select_methodology(prompt)
            assert result["name"] == "ollama_backend_readiness", (
                f"Failed for: {prompt!r} -> got {result['name']!r}"
            )

    def test_ollama_readiness_template_has_required_fields(self):
        tmpl = METHODOLOGY_TEMPLATES["ollama_backend_readiness"]
        assert "agent_mode" in tmpl.required_tool_families
        assert "readiness_status" in tmpl.required_final_classification_fields
        assert "model_roles" in tmpl.required_final_classification_fields
        assert "tool_worker_status" in tmpl.required_final_classification_fields


# ---------------------------------------------------------------------------
# No-tool response
# ---------------------------------------------------------------------------

class TestNoToolResponse:

    def test_plan_only_routes_to_no_tool(self):
        prompt = "Plan only. Do not call any tool. Describe how you would investigate."
        result = select_methodology(prompt)
        assert result["name"] == "no_tool_response"

    def test_no_tool_request(self):
        result = select_methodology("No-tool explanation of QoS session switching")
        assert result["name"] == "no_tool_response"


# ---------------------------------------------------------------------------
# SSH methodology tests
# ---------------------------------------------------------------------------

class TestSshMethodologies:

    def test_ssh_host_key_audit_routes(self):
        prompt = "Inspect the stored SSH host key fingerprint for montjac@10.79.85.47 and compare"
        result = select_methodology(prompt)
        assert result["name"] == "ssh_host_key_audit"

    def test_ssh_artifact_retrieval_routes(self):
        prompt = "Retrieve an agent artifact from montjac@10.79.85.35 using its artifact URL"
        result = select_methodology(prompt)
        assert result["name"] == "ssh_artifact_retrieval"

    def test_ssh_connectivity_check_routes(self):
        prompt = "test if you can ssh to montjac@10.79.85.35"
        result = select_methodology(prompt)
        assert result["name"] == "remote_ssh_connectivity_check"

    def test_ssh_remote_service_inspection_routes(self):
        prompt = "Use SSH to inspect the process listening on port 8510 at montjac@10.79.85.47"
        result = select_methodology(prompt)
        assert result["name"] == "ssh_remote_service_inspection"


# ---------------------------------------------------------------------------
# Existing methodology regression checks
# ---------------------------------------------------------------------------

class TestExistingMethodologyRegressions:
    """Ensure original 12 core methodologies still route correctly after parity patch."""

    def test_receiver_reboot_dvr_regression(self):
        result = select_methodology(
            "Receiver R1911746693 is repeatedly rebooting during DVR playback around 9pm on U820"
        )
        assert result["name"] == "receiver_reboot_dvr_playback"

    def test_popup_signal_loss_regression(self):
        result = select_methodology(
            "Joey signal lost popup during live TV; Hopper attached; pull STB logs"
        )
        assert result["name"] == "popup_signal_loss_investigation"

    def test_qos_ota_switchback_regression(self):
        result = select_methodology(
            "investigate QoS OTA switchback throughput stall followed by ABR session switch"
        )
        assert result["name"] == "qos_ota_switchback_investigation"

    def test_viewership_rtr_regression(self):
        result = select_methodology(
            "content partner viewership watch hours dropped; compare top services and RTR anomalies"
        )
        assert result["name"] == "viewership_rtr_investigation"

    def test_backend_runtime_debug_regression(self):
        result = select_methodology(
            "debug backend startup traceback and health endpoint logs"
        )
        assert result["name"] == "backend_runtime_debug"

    def test_repo_code_review_regression(self):
        result = select_methodology("patch the FastAPI LangGraph repo and run pytest")
        assert result["name"] == "repo_code_review"

    def test_generic_engineering_fallback(self):
        result = select_methodology("help me with a Python function")
        assert result["name"] == "generic_engineering"


# ---------------------------------------------------------------------------
# Methodology selftest
# ---------------------------------------------------------------------------

def test_methodology_selftest_passes():
    result = methodology_selftest()
    assert result["status"] == "pass", (
        f"methodology_selftest failed. Mismatches: "
        + str({k: v for k, v in result["observed"].items() if v != k and k not in result.get("known_aliases", [])})
    )
    assert result["directive_parsing_test"] is True, "Directive parsing selftest failed"
    assert len(result["expected"]) >= 29, f"Expected at least 29 templates, got {len(result['expected'])}"
