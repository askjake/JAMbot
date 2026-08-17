from types import SimpleNamespace

from app.agent.methodology import select_methodology
from app.agent.tool_execution_policy import (
    build_compact_tool_execution_system_prompt,
    build_tool_choice_plan,
    contains_pseudo_tool_response,
    rank_tools_for_retry,
    selected_toolsets_for_prompt,
)


PROMPT_A = (
    "Follow protocol. Receiver R1911746693 is experiencing repeated reboots. "
    "The customer reports it happens during DVR playback around 9pm nightly. "
    "Software version is U820. Check RTR alerts, pull S3 logs, and determine root cause."
)
PROMPT_B = (
    "Follow protocol. Customer reports a popup appeared on their Joey R2200001234 "
    "saying something about signal lost during live TV around 2:30pm yesterday. "
    "Their Hopper is R1100005678. Pull S3 STB logs and identify exactly which popup was displayed."
)
PROMPT_C = (
    "Investigate the QoS OTA switchback event on receiver R3300009876. There was a "
    "throughput stall followed by an ABR session switch around 11:45pm last Tuesday. "
    "Check QoS session data and RTR alerts for that time window."
)
PROMPT_D = (
    "A content partner is claiming their channel lost significant viewership between "
    "July 1-7, 2026. Query top 20 services by watch hours, get daily trend for top 5, "
    "check RTR anomalies, pull hourly breakdowns for >20% day-over-day drops."
)


def test_methodology_scoped_toolsets_for_regression_prompts():
    assert select_methodology(PROMPT_A)["name"] == "receiver_reboot_dvr_playback"
    assert selected_toolsets_for_prompt(PROMPT_A) == ("s3_stb_logs", "rtr_alerts_mcp")

    assert select_methodology(PROMPT_B)["name"] == "popup_signal_loss_investigation"
    assert selected_toolsets_for_prompt(PROMPT_B) == ("s3_stb_logs", "rtr_alerts_mcp")
    assert selected_toolsets_for_prompt(PROMPT_B, has_prior_tool_results=True) == (
        "s3_stb_logs",
        "rtr_alerts_mcp",
        "stbhealth_popups_mcp",
    )

    assert select_methodology(PROMPT_C)["name"] == "qos_ota_switchback_investigation"
    assert selected_toolsets_for_prompt(PROMPT_C) == ("qos_mcp", "rtr_alerts_mcp")

    assert select_methodology(PROMPT_D)["name"] == "viewership_rtr_investigation"
    assert selected_toolsets_for_prompt(PROMPT_D) == ("viewership", "rtr_alerts_mcp")


def test_tool_choice_plan_preserves_inputs_and_first_tool_targets():
    qos = build_tool_choice_plan(PROMPT_C, candidate_toolsets=("qos_mcp", "rtr_alerts_mcp"), candidate_tools=("qos_lookup_devices", "rtr_alert_lookup"))
    assert qos.methodology == "qos_ota_switchback_investigation"
    assert qos.first_tool == "qos_get_coverage"
    assert qos.required_inputs["receivers"] == ["R3300009876"]
    assert "last Tuesday" in qos.required_inputs["time_windows"]

    viewership = build_tool_choice_plan(PROMPT_D, candidate_toolsets=("viewership", "rtr_alerts_mcp"))
    assert viewership.required_inputs["date_window"] == "July 1-7, 2026"


def test_policy_prompt_is_compact_not_schema_catalog():
    plan = build_tool_choice_plan(PROMPT_A, candidate_toolsets=("s3_stb_logs", "rtr_alerts_mcp"))
    prompt = build_compact_tool_execution_system_prompt(plan)
    assert "Use bound tools when data is required" in prompt
    assert "schema" in prompt.lower()
    assert "properties" not in prompt
    assert "function" not in prompt.lower()
    assert "s3_stb_logs" in prompt and "rtr_alerts_mcp" in prompt


def test_pseudocode_and_schema_answers_are_blocked_patterns():
    assert contains_pseudo_tool_response('Here is the function definition: def query_popups(args): pass')
    assert contains_pseudo_tool_response('{"tool_calls": [{"name": "query_popups", "args": {}}]}')
    assert not contains_pseudo_tool_response('Confirmed facts: S3 log search returned no matching popup code.')


def test_retry_ranking_prefers_methodology_tools():
    tools = [
        SimpleNamespace(name="random_tool"),
        SimpleNamespace(name="qos_lookup_devices"),
        SimpleNamespace(name="qos_get_coverage"),
        SimpleNamespace(name="rtr_alert_lookup"),
    ]
    ranked = [t.name for t in rank_tools_for_retry("qos_ota_switchback_investigation", tools, limit=3)]
    assert ranked[0] == "qos_get_coverage"
    assert "qos_lookup_devices" in ranked
