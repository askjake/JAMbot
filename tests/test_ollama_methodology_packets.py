from app.agent.methodology import METHODOLOGY_TEMPLATES, methodology_selftest, select_methodology
from app.agent_mode.orchestration_packets import ToolEvidencePacket, init_run_state, load_packets, orchestration_packet_selftest, read_packet, verify_final_answer_against_packets, write_tool_evidence_packet
from app.agent_mode.orchestrator import ROLE_MODEL_MAPPING, orchestration_smoke_test


def test_methodology_selector_required_templates():
    # The original core set must always be present.
    # New methodologies added during parity enhancement are permitted in the superset.
    required_minimum = {
        "receiver_reboot_dvr_playback",
        "popup_signal_loss_investigation",
        "qos_ota_switchback_investigation",
        "viewership_rtr_investigation",
        "repo_code_review",
        "backend_runtime_debug",
        "qos_switchback_investigation",
        "epg_schedule_metadata_check",
        "dva_stb_firmware_workflow",
        "web_internal_research",
        "artifact_generation",
        "generic_engineering",
    }
    # Verify original set is a subset of (possibly expanded) templates
    assert required_minimum.issubset(set(METHODOLOGY_TEMPLATES)), (
        f"Missing required core templates: {required_minimum - set(METHODOLOGY_TEMPLATES)}"
    )
    # Additional parity-enhanced templates that must also be present
    required_parity = {
        "no_tool_response",
        "ssh_host_key_audit",
        "ssh_remote_service_inspection",
        "ssh_artifact_retrieval",
        "remote_ssh_connectivity_check",
        "remote_host_network_usb_triage",
        "log_acquisition_preview",
        "log_acquisition_execute",
        "ollama_backend_readiness",
        "tool_binding_audit",
        "tool_execution_proof",
        "tool_systematic_validation",
        "agent_efficiency_dashboard_audit",
        "local_repository_audit",
        "dvr_playback_instability_investigation",
        "performance_scalability_review",
        "timezone_window_resolution",
    }
    missing_parity = required_parity - set(METHODOLOGY_TEMPLATES)
    assert not missing_parity, f"Missing parity-required templates: {missing_parity}"
    assert select_methodology("investigate QoS OTA switchback")["name"] == "qos_ota_switchback_investigation"
    assert methodology_selftest()["status"] == "pass"


def test_packet_roundtrip_and_verifier(tmp_path):
    run_dir = init_run_state(tmp_path, chat_id="c", goal="g")
    packet_path = write_tool_evidence_packet(run_dir, ToolEvidencePacket(task_id="t1", status="complete", facts=[{"claim": "packet works", "source": "unit", "reference": "test", "confidence": "high"}]))
    assert read_packet(packet_path)["packet_type"] == "tool_evidence"
    assert load_packets(run_dir, "tool_evidence")[0]["task_id"] == "t1"
    assert verify_final_answer_against_packets("verified all and no remaining risks", []).verdict == "FAIL"
    assert orchestration_packet_selftest(tmp_path)["status"] == "pass"
    smoke = orchestration_smoke_test(tmp_path)
    assert smoke["status"] == "pass"
    assert ROLE_MODEL_MAPPING["verifier"] == "verifier"
