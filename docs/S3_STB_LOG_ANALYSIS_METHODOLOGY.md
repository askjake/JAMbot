---
name: S3 STB Log Analysis Methodology
protocol_id: s3-stb-logs-analysis
version: "1.0"
status: active
priority: sub-protocol of stb-field-rca
date: 2026-06-23
owner: montjac
parent_protocol: stb-field-rca
triggers:
  - S3 STB logs available
  - receiver-level root-cause analysis
  - crash investigation from S3 logs
  - reboot analysis from S3 logs
  - guide failure from S3 logs
  - DVR failure from S3 logs
  - playback failure from S3 logs
  - network instability from S3 logs
  - remote-control or pairing from S3 logs
  - popup investigation from S3 logs
  - UI investigation from S3 logs
  - symptom anchoring required
  - event-window control required
  - receiver and household-topology attribution
  - complete-corpus processing from S3
  - bounded evidence export
---

# S3 STB LOG ANALYSIS METHODOLOGY

**Protocol ID:** `s3-stb-logs-analysis`
**Version:** 1.0
**Date:** 2026-06-23
**Owner:** montjac
**Parent Protocol:** `stb-field-rca`
**Status:** ACTIVE

---

## 1. GOVERNING PRINCIPLES

This methodology governs analysis of STB diagnostic logs already stored in S3 for
receiver-level root-cause analysis. It uses strict symptom anchoring, event-window
control, receiver and household-topology attribution, complete-corpus processing,
competing hypotheses, persisted capsules, deterministic diagnoses, casebook provenance,
and bounded evidence export.

Use for crash, reboot, guide, DVR, playback, network, remote-control, pairing, popup,
or UI investigations where logs are in S3.

The method rejects stale rotated evidence, cross-device attribution, popup
identification without direct lifecycle evidence, and causal claims that exceed
receipt or coverage.

The investigation must remain anchored to:

1. The symptom the user actually reported
2. The exact device where the symptom was observed
3. The event date and bounded time window
4. Direct evidence from that device and window
5. A clearly stated coverage and receipt gate

A large or familiar anomaly is not automatically the investigated problem.

Do not replace the reported symptom with a more common issue merely because the
common issue has more log volume.

Do not infer popup identity from surrounding warnings, unrelated popup definitions,
or historic incidents.

Do not treat a host receiver's UI logs as direct proof of what appeared on a
client Joey.

---

## 2. SYMPTOM CONTRACT

Before broad searching, create a frozen symptom contract:

```text
reported_by
observation_timestamp or bounded time
device where symptom was visible
known receiver ID for that device
host receiver ID, when different
household/topology relationship
visible text or paraphrase
screen or application context
whether buttons responded
whether popup dismissed
recovery action
whether power cycle was required
```

Repeat the symptom contract at the top of the investigation.

When the user corrects the symptom, invalidate prior symptom-specific hypotheses
and restart from this phase.

---

## 3. DEVICE-ATTRIBUTION GATE

Classify every source as:

```text
ANCHOR_DEVICE
HOST_OF_ANCHOR
CLIENT_OF_ANCHOR
SAME_HOUSEHOLD_UNVERIFIED
FOREIGN_RECEIVER
UNKNOWN
```

An event may enter the direct causal chain only when it is from `ANCHOR_DEVICE`
or when a verified topology rule explicitly permits attribution.

Host evidence may establish:
- Household topology
- Client connectivity
- Pairing relationships
- Cross-device command routing
- Host-side responses to client requests

Host evidence may NOT by itself establish:
- The popup displayed on the Joey
- The Joey's local screen stack
- The Joey's popup create/destroy lifecycle
- The Joey's local input-routing state
- The exact recovery state of the Joey UI process

When the symptom occurred on a Joey and only Hopper logs are present:

```text
final_root_cause_allowed = false
exact_popup_identity_allowed = false
```

Request or locate the Joey receiver ID and its own logs.

---

## 4. EVENT-WINDOW GATE

Inventory log files before searching content.

For each file record:

```text
receiver_id
log family
source path
file timestamp
earliest internal timestamp
latest internal timestamp
rotation age
event-window overlap
```

Classify each file:

```text
IN_EVENT_WINDOW        — Direct evidence for this incident
NEAR_EVENT_WINDOW      — Context within ~30 min of incident
HISTORICAL_BACKGROUND  — Old rotated content
TIMESTAMP_UNRESOLVED   — Cannot determine temporal relevance
```

Only `IN_EVENT_WINDOW` evidence may establish the primary failure sequence.

Historical rotations may be reported as background behavior, but they must be
kept in a separate section and may not identify the current popup or root cause.

A log uploaded on June 23 that contains November 2022 events is historical
evidence, not evidence of the June 23 incident.

---

## 5. RECEIPT AND COVERAGE GATE

Inventory all available and required log families.

For popup and remote-mode investigations, consider:

```text
qt_gui
uitv2
input_mgr
rf4ce or remote-pairing logs
sg_server or SGS handler logs
procmgr
launcher
system or Android logs
STB health
host/client topology records
```

Record:

```text
required_log_families
observed_log_families
missing_log_families
receipt_complete
processing_complete
coverage_complete
negative_conclusions_allowed
```

If a capsule reaches an event limit:
- Shard by file or bounded time
- Process every shard
- Persist the shard manifest
- Merge deterministic findings
- Do not treat the first indexed files as the whole corpus

A capsule that indexed 2 of 11 relevant files is partial coverage.

---

## 6. SEARCH ORDER

Search in this order — do not skip to broad anomaly searches before completing
symptom-direct searches.

### A. Exact Reported Symptom

Search terms derived directly from the user's description:

```text
remote mode, remote current mode, mode, TV mode, SAT mode, AUX mode,
remote, paired, pairing, RF4CE, UHF, IR, popup, dialog, message
```

### B. Direct UI Lifecycle

```text
createScreen, destroyScreen, screenStackChanged, Popup, popup number,
popup title, popup text, dialog ID, buttonSelected, screenTimeout, standby, wake
```

### C. Input and Remote State

```text
keyPressed, keyReleased, remoteType, remote mode, input source, routing,
focus, button handler, paired remote, unpaired remote, remote address
```

### D. Recovery Behavior

```text
UI restart, process exit, watchdog, reboot, power cycle, standby resume,
screen-stack restoration, stale overlay, focus lock
```

Only after the symptom search should broader anomaly searches begin.

---

## 7. POPUP IDENTITY GATE

A popup may be identified by name or number only when there is direct evidence
from the anchor device and event window containing at least one of:

```text
explicit popup number
explicit popup name
visible message text
createScreen with typed popup identity
popup model payload
dialog resource ID linked to a definition
```

A generic `createScreen name=Popup` is NOT sufficient.

A screen-stack entry such as `PlaybackHaltedPopup` proves only that such an
object was on that device's stack at that timestamp. It does not prove it was
the popup described by the user unless device, time, text, and behavior match.

Do not select a popup from a catalog because its description sounds plausible.

---

## 8. WARNING AND TRIGGER DISTINCTION

Every warning must be classified as:

```text
PRECEDES_SYMPTOM
COINCIDES_WITH_SYMPTOM
FOLLOWS_SYMPTOM
BACKGROUND_REPEATING
TEARDOWN_ARTIFACT
UNKNOWN_RELATION
```

A warning is not a trigger unless evidence proves:

1. It occurred on the anchor device
2. It preceded popup creation
3. The same event window is involved
4. A code or runtime path links it to the popup
5. Competing explanations were tested

`Warning: unsupported remote type` must not be promoted to popup trigger merely
because it appears near a screen-stack transition.

`remoteType: "Unknown"` is a logged state value. It does not, by itself, prove:
- What message the popup displayed
- That the remote was unpaired
- That a keypress was misrouted
- That the popup concerned remote mode
- That the popup could not be dismissed

---

## 9. COMPETING-HYPOTHESIS LEDGER

Maintain at least three hypotheses when evidence permits.

For each hypothesis record:

```text
hypothesis
supporting evidence
contradicting evidence
device attribution
event-window attribution
missing proof
current status
```

Statuses:

```text
ACTIVE
WEAKENED
REJECTED
CONFIRMED_CLASS
UNRESOLVED
```

---

## 10. BACKGROUND ANOMALY QUARANTINE

Significant unrelated findings should be retained separately as `BACKGROUND_ANOMALY`.

Examples:
- SGS retry storms
- Guide/EIT failures
- Old process crashes
- Historical network failures
- Unrelated popup lifecycles

Report them, but do not place them in the primary causal chain unless exact
device and time correlation is proven.

When a prior conclusion is retracted, explicitly state:

```text
retracted hypothesis
reason for retraction
evidence that remains valid
evidence removed from primary chain
```

---

## 11. EVIDENCE LEDGER

Every load-bearing finding requires:

```text
receiver ID
source file
log family
timestamp
line or event locator
bounded raw excerpt
event-window classification
device-attribution classification
evidence hash or capsule reference
```

Separate counts into:

```text
direct symptom evidence
supporting context
normal activity
known false positives
unclassified events
```

Do not flatten all related events into one positive count.

---

## 12. CAUSAL-CLAIM GRADING

Use:

```text
RUNTIME_FACT
CORRELATED_FACT
STRONG_INFERENCE
WEAK_INFERENCE
UNRESOLVED
REJECTED
```

Examples:

```text
remoteType="Unknown" was logged → RUNTIME_FACT
the value caused popup dismissal failure → UNRESOLVED unless handler evidence proves it
a popup object remained on the Joey stack through standby → only RUNTIME_FACT with complete Joey lifecycle evidence
power cycle cleared the user-visible condition → USER_OBSERVED_FACT
power cycle was required because a specific popup object survived standby → inference until process and lifecycle evidence prove it
```

---

## 13. FINAL-ROOT-CAUSE GATE

Set `final_root_cause_allowed = true` only when:

- The anchor device is known
- The event window is known
- Relevant required logs are present
- Complete processing has finished
- The symptom is directly observed in logs or independently human-confirmed
- The causal chain uses same-device, same-window evidence
- Major competing hypotheses have been tested

Otherwise issue a `PROVISIONAL RCA` with exact next evidence requests.

---

## 14. FINAL REPORT STRUCTURE

Return:

1. Frozen symptom contract
2. Device and topology attribution
3. Event-window and log-receipt inventory
4. Direct symptom evidence
5. Competing-hypothesis ledger
6. Confirmed runtime facts
7. Background anomalies kept out of the causal chain
8. Provisional or final causal assessment
9. Evidence gaps
10. Exact next log requests
11. Casebook persistence recommendation
12. Confidence and final-root-cause gate

Never title a report with a popup number that has not been directly proven.

---

## 15. RELATIONSHIP TO PARENT PROTOCOL

This methodology is a sub-protocol of `stb-field-rca`. It activates automatically
when S3 STB logs are the primary evidence source. All parent protocol principles
(device attribution, time normalization, evidence classification, semantic safety,
receipt gates, root-cause promotion gates) remain in force. This protocol adds
additional specificity for:

- Symptom contracts (frozen before analysis)
- Event-window gating of evidence
- Popup identity verification
- Warning vs trigger separation
- S3-specific file inventory and rotation handling
- Complete-corpus methodology for bounded S3 evidence

---

## 16. CHANGE HISTORY

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-06-23 | Initial creation from SKILL.md specification |

---

## 17. REVIEW AND MAINTENANCE

- Next review: 2026-09-23 (quarterly)
- Source: gitlab dish-chat-owui skills/s3-stb-logs-analysis/SKILL.md
- Parent protocol: STB_FIELD_RCA_AND_COHORT_METHODOLOGY.md
