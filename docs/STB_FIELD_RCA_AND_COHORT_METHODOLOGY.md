---
name: STB Field RCA and Cohort Methodology
protocol_id: stb-field-rca
version: "1.1"
status: active
priority: primary
date: 2026-06-17
owner: montjac
triggers:
  - receiver issue
  - RXID investigation
  - STB crash
  - reboot
  - watchdog
  - popup
  - guide failure
  - DVR failure
  - playback failure
  - network instability
  - customer impact
  - fleet cohort
  - Grasshopper logs
  - S3 logs
  - QoS investigation
  - RTR investigation
  - field-to-code RCA
  - lab reproduction
  - follow protocol (when STB/receiver/field context present)
---

# STB Field RCA, Cohort, Customer-Impact, and Lab-Validation Methodology

**Protocol ID:** `stb-field-rca`  
**Version:** 1.1  
**Date:** 2026-06-17  
**Owner:** montjac  
**Status:** ACTIVE

---

## 1. PURPOSE AND SCOPE

This protocol provides a disciplined, repeatable methodology for investigating DISH set-top-box field issues. It answers five increasingly difficult questions:

1. **What happened?** — Observable failure event with timeline
2. **Which physical device and subsystem failed?** — Device attribution and topology
3. **What did the customer experience?** — Customer-facing impact
4. **How widespread is the same verified signature?** — Fleet prevalence and cohort
5. **What source-code mechanism and laboratory trigger explain it?** — Root cause and reproduction

### Semantic Separation

Every investigation must separate these concepts and never collapse them into one statement:

- **Field signature** — The specific log/alert pattern observed (e.g., SgsAv.js:112 mismatch loop)
- **Customer symptom** — What the viewer experienced (e.g., spontaneous reboot, lost recording)
- **Failure mechanism** — The proximate technical failure (e.g., ASSERT on IPC delivery failure)
- **Causal trigger** — What initiated the failure chain (e.g., stale reply-service race condition)
- **Design defect** — The underlying code/architecture flaw (e.g., fatal handling of recoverable error)
- **Root cause** — The complete chain from trigger through defect to customer impact
- **Recovery mechanism** — How the system recovered or failed to recover

---

## 2. PROTOCOL SELECTION RULES

### Primary Match — Select This Protocol When:

- An RXID, receiver, Hopper, Joey, set-top, or household is involved
- Crash, reboot, watchdog, SEGV, SIGABRT, tombstone, core, or stack dump
- Guide, popup, DVR, tuner, playback, video, audio, or channel issue
- SGS, RCA, Gandalf, zip1018D, QML, Rigel, or stbCtrl
- RTR alerts, Grasshopper uploads, or S3 STB logs
- QoS sessions or customer impact analysis
- Fleet prevalence, receiver cohorts, or beta reports
- Lab reproduction or field-to-code investigation
- User says "follow protocol" with STB/field context present

### Secondary Match — Use as Supporting Protocol When:

- A JIRA defect needs field evidence
- A code fix must be validated against receiver behavior
- A software regression must be correlated with builds
- A lab test must reproduce a field issue

### Do NOT Select as Primary For:

- Generic coding tasks without field symptoms
- Deployment-only work (use deployment_best_practices protocol)
- Kubernetes/AWS infrastructure issues
- Document formatting or business reporting without receiver evidence
- General source-code review with no field symptom

### Mixed-Protocol Handling

If multiple protocols apply, select this as primary for field evidence and RCA, then apply source-code, deployment, or documentation protocols as secondary procedures. State the chosen protocol briefly.

### Invocation Message

> Protocol selected: STB Field RCA and Cohort Methodology. Beginning with device attribution, evidence inventory, and lightweight triage; escalating only where evidence requires it.

---

## 3. CORE INVESTIGATION PRINCIPLES

### A. Device Attribution

Always distinguish:

- **Requested RXID** — what the user asked about
- **S3 anchor RXID** — whose S3 path the logs are stored under
- **Source RXID** — embedded in filenames (may differ from anchor)
- **Host/Hopper** — the primary STB
- **Joey/client** — the satellite client device
- **Household member** — any device in the same household
- **Upload proxy** — Grasshopper may route through a different device
- **Process owner** — which device owns the failing process

Never attribute an event to the anchor merely because logs are stored beneath its S3 path.

### B. Time Normalization

For every significant event preserve: raw timestamp, timezone, normalized UTC, timestamp source, precision, and category (event time, filename time, upload time, boot-relative time, uptime, or retained historical content).

Never compare timestamps until their time bases are reconciled.

### C. Evidence Classifications

Every important conclusion must be classified:

- **RUNTIME_FACT** — Directly observed in logs with reliable timestamp
- **SOURCE_CODE_FACT** — Verified in source code at a specific commit
- **BUILD_APPLICABILITY_FACT** — Confirmed present in the incident build
- **CUSTOMER_IMPACT_FACT** — Confirmed via QoS/viewership/session data
- **STRONG_INFERENCE** — Supported by multiple independent evidence sources
- **HYPOTHESIS** — Plausible but not yet confirmed
- **CONTRADICTED** — Actively contradicted by evidence
- **EVIDENCE_UNAVAILABLE** — Cannot be determined from available data
- **TOOL_OR_DATA_DEFECT** — Evidence missing due to tool/pipeline limitation

A conclusion must not silently move from HYPOTHESIS to RUNTIME_FACT.

### D. Semantic Safety Rules

Enforced:

- Numeric service UID is not a channel name until resolved via EPG
- DVR ID is an opaque identifier, not a playback position
- Only videoPos fields represent playback position
- No video stall conclusion without repeated identical videoPos values
- Numeric result codes remain opaque until mapped via source/documentation
- Trick-mode state is not visual motion detection
- Normal polling at designed intervals is not a retry storm
- Normal Android lifecycle kills are not automatically memory exhaustion
- Stale crash files do not prove a current crash
- A post-crash event cannot be promoted into a pre-crash trigger without clock evidence
- Absence from bounded search results is not proof of absence from full log corpus
- Parser failure must not masquerade as absence of evidence

---

## 4. LIGHTWEIGHT TRIAGE (Mandatory First Pass)

Before launching heavy tools, perform bounded triage:

### Information to Gather

1. Receiver identity and model
2. Household topology (host/client relationships)
3. Software version
4. Available dates (list_dates)
5. Log inventory (list_files for relevant dates)
6. Source RXIDs in filenames
7. RTR alerts (count_alerts with receiver filter)
8. STB Health counters (get_detected_errors)
9. QoS availability (qos_get_coverage)
10. Existing artifacts (list_log_capsules, list_upload_trackers, search_investigation_cases)
11. Likely issue family
12. Missing critical log families

### Triage Output

- Investigation anchor (primary RXID, date, event window)
- Suspected signature and issue family
- Initial topology map
- Evidence inventory (what exists, what is missing)
- Initial hypotheses (ranked)
- Heavy-tool escalation decision

Use search_logs for bounded raw-text searches during triage. Do not immediately launch complete-corpus jobs for every issue.

---

## 5. HEAVY-TOOL ESCALATION

### Default: Restricted Mode

Appropriate for: initial triage, inventory, bounded search, existing capsule queries, preliminary customer impact, hypothesis generation.

### Escalation Criteria

Escalate to heavy tools (allow_heavy=true) when:

- Complete log coverage across all rotations is required
- A new persisted capsule must be built
- Multiple log families must be correlated across time
- Format discovery is needed
- Parser drift is suspected
- A negative conclusion is needed
- A full timeline must be reconstructed
- Cohort comparison requires parallel capsules
- Engineering handoff requires evidence packaging
- Complete-corpus processing needed for high-volume logs

Document the specific heavy tools needed.

---

## 6. GRASSHOPPER ACQUISITION AND RECEIPT GATES

### Workflow

1. Inspect available files: grasshopper_list_files
2. Choose profile: grasshopper_get_log_type_catalog
3. Plan upload: grasshopper_plan_profile_upload (preview)
4. Execute: grasshopper_upload_profile_logs (dry_run=true first)
5. Record tracker: record_upload_tracker
6. Monitor receipt: update_upload_tracker_from_s3
7. Verify landing: verify_profile_upload_receipt

### Receipt Gate Rule

Set final_root_cause_allowed=false when a required log type has been requested but has not landed AND that log type contains a material missing causal link.

Positive findings remain valid. Negative conclusions may not be drawn past an unsatisfied receipt gate.

---

## 7. LOG FORMAT AND PARSER CALIBRATION

Before broad analysis:

1. Use read_log for bounded inspection, then run discover_log_formats_and_templates where available
2. Identify parser selected for each physical log type
3. Report parse success rate
4. Detect unknown or changed formats
5. Check for: parser errors, read errors, compressed-file errors, zero-event files, out-of-window files

Parser failure must not be reported as "no evidence found."

---

## 8. CAPSULE METHODOLOGY

### When to Build

Issue-focused investigation (use build_log_capsule), engineering handoff, cross-receiver comparison, timeline reconstruction.

### Capsule Record

Each capsule must track: capsule ID, receiver, date, source scope, profile, time window (requested and actual), selected files, contributing files, unavailable core types, zero-event types, parser errors, event count, template count, truncation state, persistence location.

### Separation Principle

Use different capsules for different hypotheses: reboot/crash, DVR desync, guide 1031, playback/PTS, network, process manager, advertising/splicer.

---

## 9. COMPLETE-CORPUS METHODOLOGY

### When Required

All rotations must be scanned, high-volume logs involved, negative conclusion needed, template prevalence matters, issue crosses multiple log types.

### Job Lifecycle

1. start_complete_corpus_job (dry_run first)
2. run_complete_corpus_job_autopilot
3. get_complete_corpus_job_status
4. finalize_complete_corpus_job
5. Apply deterministic diagnosis
6. Export evidence report
7. Link to casebook case

An incomplete corpus job must not be treated as exhaustive.

---

## 9A. S3 STB LOG ANALYSIS SUB-PROTOCOL

When logs are already in S3 and require receiver-level root-cause analysis, the
**S3 STB Log Analysis Methodology** sub-protocol activates automatically.

**Sub-Protocol Documentation:** `/home/jakebot/Jakes-agent/docs/S3_STB_LOG_ANALYSIS_METHODOLOGY.md`
**Protocol ID:** `s3-stb-logs-analysis`

This sub-protocol enforces additional discipline specific to S3-resident evidence:

1. **Symptom Contract** — Frozen before analysis; repeated at report top; invalidated on user correction.
2. **Device-Attribution Gate** — Events enter the causal chain only from ANCHOR_DEVICE unless topology rules explicitly permit.
3. **Event-Window Gate** — Files classified as IN_EVENT_WINDOW, NEAR, HISTORICAL, or UNRESOLVED; only IN_EVENT_WINDOW establishes the primary failure sequence.
4. **Receipt and Coverage Gate** — Required vs observed log families inventoried; partial capsule coverage blocks negative conclusions.
5. **Search Order** — Symptom-first: exact reported symptom → UI lifecycle → input/remote state → recovery behavior → then broader anomalies.
6. **Popup Identity Gate** — A popup may be identified only with direct lifecycle evidence from anchor device and event window.
7. **Warning vs Trigger Distinction** — Warnings classified by temporal relation; not promoted to trigger without five-point proof.
8. **Competing-Hypothesis Ledger** — Minimum three hypotheses maintained with status tracking.
9. **Background Anomaly Quarantine** — Unrelated significant findings kept separate from causal chain.
10. **Causal-Claim Grading** — RUNTIME_FACT / CORRELATED_FACT / STRONG_INFERENCE / WEAK_INFERENCE / UNRESOLVED / REJECTED.
11. **Final-Root-Cause Gate** — Allowed only when anchor device known, event window established, required logs present, processing complete, competing hypotheses tested.

**Activation:** Automatic when S3 STB logs are the primary evidence source for crash,
reboot, guide, DVR, playback, network, remote-control, pairing, popup, or UI investigations.

---

## 10. TIMELINE RECONSTRUCTION

Each row: timestamp UTC, original timestamp, source RXID, process, log type, physical file, line/template ID, raw excerpt, structured fields, interpretation, evidence classification.

Nested windows: broad context, 30 min before, 10 min before, 5 min before, 1 min before, failure boundary, recovery period.

Temporal categories: precursor, trigger, failure, recovery, post-failure cleanup, historical/stale evidence.

---

## 11. CROSS-SOURCE CORRELATION

### RTR Alerts
Fleet counts, model/software prevalence, recurrence, household linkage. Tools: count_alerts, alert_time_series, compare_alert_volume, lookup_alert_definitions, list_anomalies.

### STB Health
Boot count, SEGV count, restart count, watchdog count, last stack dump, uptime. Tools: get_detected_errors, get_ml_analysis, get_change_points, get_matrix_profile.

### QoS
Sessions, failures, customer-facing behavior, error states, timing, maturity. Tools: qos_get_sessions, qos_get_error_summary, qos_channel_failure_profile, qos_baseline_failure_rate.

### Viewership
Active viewing, service exposure, live vs DVR, customer count, duration. Tools: query_viewership, get_service_trends, get_hourly_breakdown.

### EPG
Service identity, schedule, arc, transponder, LLOTT. Tools: get_service_info, get_transponder_info, get_schedule_by_channel.

### Beta Reports
Software version, deployment timing, regression windows. Tools: search_issues, search_reports, get_daily_summary.

### JIRA/Confluence
Known defects, release history, prior reproduction. Tools: search_jira, get_jira_issue, search_confluence.

### RCA Pipeline
Automated analysis pipeline. Tools: tool_trigger_analysis, tool_get_analysis_result, tool_get_device_history.

Do not use one source as a substitute for another.

---

## 12. CUSTOMER-IMPACT METHODOLOGY

Report separately: receiver failure, household impact, active-viewing impact, session impact, fleet exposure.

Every number labeled: directly measured, sampled, estimated, or extrapolated.

Report: affected receivers, households, active customers, viewers, repeated failures, duration, interruption type, recovery mode, recordings affected, confidence limits.

---

## 13. COHORT METHODOLOGY

### Anchor Signature
Define using: model, software, topology, exact runtime signature, event sequence, process, signal, alert combination, customer symptom, time relationship. Never build a cohort around one noisy alert.

### Cohort Classes
1. Exact-signature affected
2. Recurring/chronic
3. Single-event
4. Matched unaffected control
5. Symptom-without-signature
6. Signature-without-customer-impact

### Matching Dimensions
Model, software, host/client role, topology, tuner/DVR state, application state, network, time of day, build exposure, active viewing.

### Validation
Prevalence, recurrence, false-positive rate, build correlation, model correlation, customer impact, confidence limits.

### Limitations Disclosure
Missing logs, incomplete sampling, upload bias, alert noise, changing populations, data maturity, activity coverage.

---

## 14. CASEBOOK METHODOLOGY

### When to Create
Stable signature with evidence IDs, meaningful engineering conclusion (use record_investigation_case) (use record_investigation_case), escalation, reproducible condition, control comparison complete.

### Case Status Vocabulary
TRIAGE, FIELD_SIGNATURE_CONFIRMED, CUSTOMER_IMPACT_CONFIRMED, CAUSAL_CHAIN_PARTIAL, CODE_PATH_CONFIRMED, ROOT_CAUSE_CONFIRMED, HYPOTHESIS_FALSIFIED, MORE_LOGS_REQUIRED, TOOL_OR_DATA_DEFECT, CLOSED_FIXED, CLOSED_NOT_REPRODUCED.

### Required Fields
Case ID, anchor RXID, household, source devices, model/software, event window, symptoms, confirmed facts, hypotheses, contradictions, missing evidence, capsule IDs, corpus job IDs, Grasshopper requests, source commits, customer impact, cohort metrics, lab status, JIRA link, next action, final verdict.

---

## 15. SOURCE-CODE INVESTIGATION

### Dual Tool Strategy

**Qodo (semantic):** Architecture discovery, multi-file research, candidate call chains.

**dish-code-tools (deterministic):** Exact string match, file reads, symbols, references, git history, diffs.

### Required Workflow

1. Qodo to locate candidates
2. search_regex to verify exact strings
3. read_file for complete function
4. find_references for callers/callees
5. Map runtime values to source conditions
6. Record branch/commit from get_log
7. Determine build applicability
8. Search history for regressions via get_diff
9. Separate current-HEAD from incident-build behavior

No code-level claim is final without: exact source, exact condition, build applicability, runtime evidence linking the path.

---

## 16. CRASH AND TOMBSTONE METHODOLOGY

1. Identify physical device and crash type
2. Determine artifact type and build ID
3. Validate integrity (not truncated, not stale)
4. Identify signal, PC/LR/SP, faulting instruction, ASSERT condition
5. Distinguish full symbolication vs partial stack string
6. Preserve chain of custody

---

## 17. HYPOTHESIS AND FALSIFICATION

For every hypothesis: supporting evidence, contradicting evidence, expected mechanism, expected sequence, missing evidence, falsification test, confidence.

Actively test alternatives. Coherent narrative alone does not increase confidence.

---

## 18. ROOT-CAUSE PROMOTION GATES

### Verdicts
ROOT_CAUSE_CONFIRMED, EXACT_ASSERT_CONFIRMED_TRIGGER_PARTIAL, CODE_PATH_CONFIRMED_RUNTIME_TRIGGER_PARTIAL, FIELD_CAUSAL_CHAIN_PARTIALLY_CONFIRMED, FIELD_SIGNATURE_CONFIRMED_CAUSE_UNKNOWN, CUSTOMER_IMPACT_CONFIRMED_CAUSE_UNKNOWN, MULTIPLE_CAUSES_BEHIND_COMMON_ALERT, SOURCE_VERSION_APPLICABILITY_UNCONFIRMED, MORE_LOGS_REQUIRED, TOOL_OR_DATA_DEFECT_FOUND, HYPOTHESIS_FALSIFIED.

### ROOT_CAUSE_CONFIRMED Requirements (ALL must be met)

1. Correct physical-device attribution
2. Confirmed failure event with reliable timestamp
3. Reliable timeline connecting precursors to failure
4. Exact source mechanism identified
5. Incident-build applicability confirmed
6. Required evidence receipt gate satisfied
7. Material competing hypotheses tested
8. Customer impact independently assessed
9. Direct crash evidence OR repeatable lab reproduction
10. No unresolved contradiction in causal chain

---

## 19. LAB REPRODUCTION

### Prerequisites
Earliest abnormal state, required device/model/build, topology, customer action, background load, exact source path, expected logs, expected symptom.

### Experiment Types
Minimal trigger, field-equivalent, stress/acceleration, negative controls, last-known-good comparison, candidate-fix comparison.

### Confirmation Criteria
Same field signature occurs, same code path exercised, same failure manifests, customer symptom matches, repeats consistently, negative controls clean.

---

## 20. FIX VALIDATION

Old build reproduces, fixed build receives same trigger, fatal condition gone, normal functionality preserved, no replacement failure mode, no performance regression, multiple passes, source diff matches tested binary.

---

## 21. TOKEN DISCIPLINE

Use: bounded searches, persisted capsules, template IDs, targeted queries, artifact links, concise excerpts. Do not: paste full logs, repeat evidence in every phase, confuse token reduction with heavy-tool restriction.

---

## 22. UNAVAILABLE TOOLS

The following are NOT currently available as registered MCP tools:

- **Net Detective / Netra** — Use S3 log evidence (sddp_log, wjap) and QoS session data instead
- **Live STB Health popups** — Use query_popups for definitions; no live snapshot available
- **Full remote symbolication** — Partial stack strings from logs only

Note unavailability explicitly when these are referenced.

---

## 23. CHANGE HISTORY

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-06-17 | Initial protocol creation |
| 1.1 | 2026-06-23 | Added S3 STB Log Analysis sub-protocol reference (Section 9A) |

---

## 24. REVIEW AND MAINTENANCE

- Next review: 2026-09-17 (quarterly)
- Tool assumptions: Based on MCP tool registry as of 2026-06-17
- Deprecated methods: None (initial version)
