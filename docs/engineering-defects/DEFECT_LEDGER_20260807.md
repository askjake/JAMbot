# Engineering Defect Ledger — 2026-08-07

Canonical entries: **27**
Named IDs in supplied debug checklist: **79**
Alias/recurrence records: **93**

## Summary

| Disposition | Count |
|---|---:|
| Fixed and verified in Jake/source scope | 21 |
| External component patch required | 2 |
| Runtime evidence needed | 1 |
| Deferred with reason | 3 |
| Aliases/recurrences | 93 |

## Canonical entries

| ID | Severity | Component | Status | Title |
|---|---|---|---|---|
| `ACTIVATION-DIRECTIVE-PROVENANCE-CONTAMINATION` | P1 | Jake-Bot activation intent | **FIXED_AND_VERIFIED** | Examples, negations, URIs, and incidental owner:name text became sticky tool requests |
| `AUTH-CURRENT-TURN-DIRECTIVE-PROVENANCE` | P1 | Jake-Bot authorization parser | **FIXED_AND_VERIFIED** | Canonical flags were not runtime-wired and authorization examples mutated live state |
| `BIND-DYNAMIC-MCP-EXACT-TOOL-NOT-PROMOTED` | P1 | Jake-Bot tool binding | **FIXED_AND_VERIFIED** | Dynamic MCP activation requests were not promoted or persisted |
| `CONFIG-EFFECTIVE-MODE-SOURCE-HIDDEN` | P2 | Jake Nightly RCA config | **FIXED_AND_VERIFIED** | Effective commit mode and cron timezone provenance were hidden or overstated |
| `EVIDENCE-TOOL-ERROR-CONVERTED-TO-ABSENCE` | P1 | Jake-Bot evidence semantics | **FIXED_AND_VERIFIED** | Tool execution errors were converted into negative evidence |
| `GH-UNKNOWN-PROFILE-RETURNS-SUCCESS-ZERO` | P1 | Grasshopper MCP | **FIXED_AND_VERIFIED** | Grasshopper treated unknown profile names as successful zero-file plans |
| `GIT-AUTHORITATIVE-WORKTREE-CONTEXT-LOST` | P1 | Jake-Bot repository execution planning | **DEFERRED_WITH_REASON** | Known repositories, worktrees, branches, and push targets were not durable across turns |
| `GRASSHOPPER-SMP-WIRE-CONTRACT-UNRESOLVED` | P1 | Grasshopper MCP SMP upload client | **EXTERNAL_COMPONENT_PATCH_REQUIRED** | The upstream v2 upload DTO/request contract remains unresolved in the supplied Jake source |
| `GRASSHOPPER-STATUS-HISTORY-CAPABILITY-GAP` | P1 | Grasshopper MCP status/history client | **EXTERNAL_COMPONENT_PATCH_REQUIRED** | Advertised upload status/history tools have no proven upstream client capability |
| `GRASSHOPPER-UPLOAD-SAFETY-CONTRACT` | P1 | Jake Nightly RCA Grasshopper adapter | **FIXED_AND_VERIFIED** | Live-write confirmation, duplicate safety, protocol errors, and plan identity were incomplete |
| `INCIDENT-SCENE-CONTRACT-RENDER-BOUNDS` | P1 | S3 STB Logs Incident Scene | **FIXED_AND_VERIFIED** | Incident Scene query, expansion, and rendering contracts were incomplete and unbounded |
| `INFRA-S3-ARTIFACT-FETCH-NONRESILIENT` | P2 | S3 STB Logs transfer | **FIXED_AND_VERIFIED** | S3 artifact fetches were single-shot and transport failures were misclassified |
| `NAL-CURRENT-PLAN-TEST-INCOMPLETE` | P2 | Nightly RCA / Grasshopper runtime | **RUNTIME_EVIDENCE_NEEDED** | Current Grasshopper NAL plan/upload canary has not been executed |
| `ORCH-CAPABILITY-PREFLIGHT-MISSING` | P1 | Jake-Bot MCOP | **FIXED_AND_VERIFIED** | Child tasks could start without a sufficient usable capability binding |
| `ORCH-COMPLETION-CONTRACT-ABANDONMENT` | P1 | Jake-Bot parent orchestration | **FIXED_AND_VERIFIED** | Structured multi-phase tasks could terminate before required report fields were produced |
| `ORCH-CROSS-DOMAIN-CONTEXT-CONTAMINATION` | P1 | Jake-Bot agent continuity | **FIXED_AND_VERIFIED** | Current-turn task and methodology displaced by stale checkpoint state |
| `ORCH-ENVIRONMENT-IDENTITY-DRIFT` | P1 | Jake-Bot operational identity | **FIXED_AND_VERIFIED** | Remote endpoints could be mistaken for the agent or repository host |
| `ORCH-EVIDENCELESS-PROGRESS-NARRATION` | P1 | Jake-Bot progress/UI contract | **DEFERRED_WITH_REASON** | Progress claims were not structurally tied to load-bearing evidence |
| `ORCH-NO-PROGRESS-TOOL-THRASH` | P1 | Jake-Bot orchestration | **FIXED_AND_VERIFIED** | Missing required capability caused phase bypass and unrelated fallback loops |
| `ORCH-PARENT-ONLY-VIOLATION-RECURRENCE` | P1 | Jake-Bot orchestration | **FIXED_AND_VERIFIED** | Explicit parent-only execution constraint was not represented as durable policy |
| `ORCH-RESULT-ORPHANING` | P1 | Jake-Bot MCOP/audit | **FIXED_AND_VERIFIED** | Blocked/partial child packets and executed-tool evidence lost status or provenance |
| `PROFILE-METADATA-DROPPED-BEFORE-ACQUISITION` | P1 | Jake Nightly RCA / S3 profile catalog | **FIXED_AND_VERIFIED** | Nightly RCA dropped S3 profile metadata and misread nested Grasshopper contracts |
| `SEC-CREDENTIALS-PASTED-IN-CONVERSATION` | P0 | Operational security | **DEFERRED_WITH_REASON** | Raw credentials were pasted into the conversation transcript |
| `SEC-GRASSHOPPER-SOURCE-TLS-CONTAINMENT` | P0 | Grasshopper MCP security | **FIXED_AND_VERIFIED** | Legacy Grasshopper source exposed a credential and disabled TLS verification |
| `TEST-PASS-CLAIM-LACKS-CALL-PATH-COVERAGE` | P1 | Cross-repository engineering process | **FIXED_AND_VERIFIED** | Green unit tests and baseline failures were overstated without environment/call-path proof |
| `TRACKER-REQUEST-ID-PROVENANCE-MISSING` | P1 | Jake Nightly RCA | **FIXED_AND_VERIFIED** | Tracker, local correlation, external request IDs, currentness, and count scope were conflated |
| `TYPED-ID-ARGUMENT-PROVENANCE` | P1 | Jake-Bot execution gate | **FIXED_AND_VERIFIED** | Context identifiers could populate domain identifier arguments |

## Canonical details

### ACTIVATION-DIRECTIVE-PROVENANCE-CONTAMINATION — Examples, negations, URIs, and incidental owner:name text became sticky tool requests

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Phantom pending-registry entries polluted binding state and no-progress decisions.
- **Runtime evidence:** mailto:git, mailto:test, key:value, owner:name, toolset:tool, and foo_mcp:some_tool persisted despite not being operator activation directives.
- **Source evidence:** Activation extraction now consumes only positive current-turn activation directives after code/quote/example sanitization; unknown pending families require an explicit plausible *_mcp request.
- **Reproduction:** Mention qualified tokens in URLs, reports, examples, fenced code, negated instructions, and explicit nonsense bind commands; none may enter checkpoint state. Explicit newserver_mcp:some_tool remains pending registry.
- **Candidate files:** app/agent/directive_text.py, app/agent/tool_activation_intent.py, app/agent/tool_policy_transition.py
- **Acceptance tests:** tests/test_post_canary_hardening_20260807.py
- **Deployment:** Jake local/dev first
- **Rollback:** Revert activation provenance changes; legacy phantom checkpoint rows remain dropped fail-closed.
- **Aliases/recurrences:** BARE-EXACT-TOOL-LIST-NOT-EXTRACTED, JAKE-02, PHANTOM-QUALIFIED-TOOL-REQUESTS

### AUTH-CURRENT-TURN-DIRECTIVE-PROVENANCE — Canonical flags were not runtime-wired and authorization examples mutated live state

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Legitimate grants failed while quoted/example revocations could remove operational tools.
- **Runtime evidence:** operator_authorized=true was ignored, while a later example containing revoke operator authorization changed only the operator flag to false.
- **Source evidence:** The single current-Human parser now accepts only exact key=true|false declarations, retains natural-language grants/revocations, strips non-authoritative example/code/quoted regions, and applies revocation precedence.
- **Reproduction:** Compare canonical and natural-language grants; include quoted/fenced/example authorization text and contradictory live revocation.
- **Candidate files:** app/agent/directive_text.py, app/agent/tool_authorization.py, app/agent/tool_policy_state.py
- **Acceptance tests:** tests/test_post_canary_hardening_20260807.py, tests/test_tool_authorization_v1.py
- **Deployment:** Jake local/dev first
- **Rollback:** Revert parser v2; stored authorization booleans remain backward compatible.
- **Aliases/recurrences:** AUTH-KEY-VALUE-GRANTS-TESTED-BUT-NOT-RUNTIME-WIRED, AUTH-PARSER-CANONICAL-FLAGS-DIVERGENCE, AUTH-PROMPT-CONTRACT-MISMATCH, JAKE-03, JAKE-17

### BIND-DYNAMIC-MCP-EXACT-TOOL-NOT-PROMOTED — Dynamic MCP activation requests were not promoted or persisted

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Authorized tools disappear, forcing false capability blocks and unrelated fallbacks.
- **Runtime evidence:** Healthy live-discovered Grasshopper family and true authorization flags still produced no model-facing plan/upload tools after a turn or restart.
- **Source evidence:** Request-only management facade output was not merged into checkpoint policy; exact dynamic names were not resolved early enough.
- **Reproduction:** Synthetic healthy dynamic family, exact qualified tools, same-turn bind, management-result merge, two-turn persistence, revoke/restore.
- **Candidate files:** app/agent/tool_activation_intent.py, app/agent/tool_policy_transition.py, app/agent/tool_policy_state.py, app/agent/tool_profiles.py, app/agent/agents/agentic_rag.py, app/agent/agents/tools/management.py
- **Acceptance tests:** tests/test_dynamic_activation_intent_hardening.py
- **Deployment:** Jake local/dev first
- **Rollback:** Revert activation commits; facade remains truthful/read-only.
- **Aliases/recurrences:** AUTH-BINDING-LOSS-DURING-IMPLEMENTATION, AUTH-PRIVILEGED-TOOL-BINDING-NOT-STICKY, OBS-MCP-FAMILY-HEALTH-CONFLATED-WITH-TOOL-BINDING, ORCH-UPLOAD-CAPABILITY-NOT-DISCOVERED

### CONFIG-EFFECTIVE-MODE-SOURCE-HIDDEN — Effective commit mode and cron timezone provenance were hidden or overstated

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Operators cannot know what mode ran or when cron actually fires.
- **Runtime evidence:** Base config showed writes false while launcher injected true; 02:30 was labeled UTC without direct timezone evidence.
- **Source evidence:** Effective configuration records CLI/env/bootstrap source without values; cron timezone is DIRECT only from CRON_TZ/TZ, otherwise UNKNOWN.
- **Reproduction:** CLI commit, env bootstrap, direct CRON_TZ, absent timezone, safe report/notification.
- **Candidate files:** apps/nightly_rca/config.py, apps/nightly_rca/run.py, apps/nightly_rca/run_nightly.sh, apps/nightly_rca/report.py, apps/nightly_rca/notify.py
- **Acceptance tests:** apps/nightly_rca/tests/test_effective_config_and_tracker_provenance_hardening.py
- **Deployment:** Jake local/dev first
- **Rollback:** Revert config-provenance commit.
- **Aliases/recurrences:** CRON-TIMEZONE-INFERENCE-OVERSTATED, CRON-TIMEZONE-MISINTERPRETED

### EVIDENCE-TOOL-ERROR-CONVERTED-TO-ABSENCE — Tool execution errors were converted into negative evidence

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** False RCA conclusions and incorrect upload decisions.
- **Runtime evidence:** S3/tool transport failures were described as proving no logs or no artifacts.
- **Source evidence:** Normalized tool outcomes separate execution result, coverage, and negative-conclusion permission.
- **Reproduction:** Executor error, not-found, empty successful result, and positive result fixtures.
- **Candidate files:** app/agent/evidence_status.py, app/agent/tool_execution_gate.py
- **Acceptance tests:** tests/test_no_progress_and_evidence_status_hardening.py, tests/test_tool_execution_gate_v1.py
- **Deployment:** Jake local/dev first
- **Rollback:** Revert outcome normalization.
- **Aliases/recurrences:** EVIDENCE-FROZEN-REPLAY-DESCRIBED-AS-LIVE, EVIDENCE-NO-CAPSULES-AS-NO-ANALYSIS, EVIDENCE-S3-COVERAGE-CONFLATED-WITH-GRASSHOPPER-PLAN, EVIDENCE-TOOL-ERROR-CONVERTED-TO-ABSENCE-RECURRENCE, NIGHTLY-CURRENT-COVERAGE-NOT-VERIFIED, OBS-GRASSHOPPER-MCP-MISIDENTIFIED-AS-S3-TOOLCHAIN

### GH-UNKNOWN-PROFILE-RETURNS-SUCCESS-ZERO — Grasshopper treated unknown profile names as successful zero-file plans

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Submission skipped without a diagnostic; upstream reports false absence.
- **Runtime evidence:** Unknown atv_reboot_instability value was treated as a physical log type and returned success with zero selected files.
- **Source evidence:** Canonical catalog/alias resolution validates before network and returns INVALID_PROFILE with valid vocabulary.
- **Reproduction:** Known canonical, known alias, and unknown profile pure-function/server fixtures.
- **Candidate files:** app/server.py
- **Acceptance tests:** tests/test_profile_contract_hardening.py
- **Deployment:** Grasshopper dev first; production manual
- **Rollback:** Revert profile-validation commit.

### GIT-AUTHORITATIVE-WORKTREE-CONTEXT-LOST — Known repositories, worktrees, branches, and push targets were not durable across turns

- **Status:** `DEFERRED_WITH_REASON`
- **Impact:** Edits wrong checkout, loses test lineage, or conflates repositories.
- **Runtime evidence:** The agent ignored established /tmp worktrees, searched a generic workspace, and re-asked which repository/branch to push.
- **Source evidence:** Task scope and authoritative environment are now durable, but the uploaded backend has no typed multi-repository execution-plan checkpoint.
- **Reproduction:** Requires a two-turn multi-repository push workflow with exact worktrees/branches and no repeated details.
- **Candidate files:** app/agent/continuity_policy.py, app/agent/tool_policy_state.py
- **Acceptance tests:** test_multi_repository_execution_plan_persists_across_push_turn
- **Deployment:** Future focused Jake patch
- **Rollback:** N/A until implemented.
- **Aliases/recurrences:** GIT-KNOWN-PUSH-TARGET-REASKED, GIT-PUSH-PROTOCOL-STATE-NOT-STICKY, ORCH-KNOWN-RECEIVER-ID-REASKED
- **Deferred reason:** A typed multi-repository plan must be designed with the unavailable full runtime/checkpoint suite; no fake generic-workspace memory was added.

### GRASSHOPPER-SMP-WIRE-CONTRACT-UNRESOLVED — The upstream v2 upload DTO/request contract remains unresolved in the supplied Jake source

- **Status:** `EXTERNAL_COMPONENT_PATCH_REQUIRED`
- **Impact:** The one-call NAL canary may continue to receive HTTP 406/upstream 4002, and Jake cannot safely repair a DTO it does not own.
- **Runtime evidence:** A prior bounded call returned HTTP 406, upstream code 4002, JSON Not readable, with no accepted request ID.
- **Source evidence:** This Jake patch classifies the failure and disables automatic retry, but the uploaded artifact contains no authoritative Grasshopper SMP client source or verified upstream DTO.
- **Reproduction:** Run the separately authorized one-call dev canary after external wire-contract tests and credential rotation.
- **Candidate files:** external grasshopper-mcp repository: client/server upload adapter
- **Acceptance tests:** Grasshopper v2 outbound DTO integration test, bounded dev NAL upload returns accepted external request ID
- **Deployment:** Grasshopper MCP repository/dev deployment; production manual only
- **Rollback:** No Jake rollback can repair this external component; retain Jake fail-closed behavior until the external patch is deployed.
- **Aliases/recurrences:** GH-FRESH-LOG-REQUEST-CONFLATED-WITH-FILE-UPLOAD, GH-OUTBOUND-PAYLOAD-OBSERVABILITY-MISSING, GH-UPLOAD-V2-PAYLOAD-CONTRACT-MISMATCH

### GRASSHOPPER-STATUS-HISTORY-CAPABILITY-GAP — Advertised upload status/history tools have no proven upstream client capability

- **Status:** `EXTERNAL_COMPONENT_PATCH_REQUIRED`
- **Impact:** Legacy tracker request IDs cannot be reconciled authoritatively through Grasshopper, so duplicate preflight remains incomplete unless durable receipt evidence resolves them.
- **Runtime evidence:** Both bound read-only tools returned status=unavailable with configured status/history paths empty and no compatible client methods.
- **Source evidence:** The uploaded Jake source can fail closed and preserve UNKNOWN coverage, but does not contain the Grasshopper client implementation needed to add a proven upstream method.
- **Reproduction:** Call the two status/history tools for request 14054596881153157; expect capability unavailable until the external component is patched.
- **Candidate files:** external grasshopper-mcp repository: grasshopper_client/status-history tools
- **Acceptance tests:** proven read-only status client contract, transport failure remains UNKNOWN, no upload endpoint invocation
- **Deployment:** Grasshopper MCP repository/dev deployment; production manual only
- **Rollback:** No Jake rollback can repair this external component; retain Jake fail-closed behavior until the external patch is deployed.
- **Aliases/recurrences:** GH-UPLOAD-STATUS-HISTORY-CLIENT-CAPABILITY-MISSING

### GRASSHOPPER-UPLOAD-SAFETY-CONTRACT — Live-write confirmation, duplicate safety, protocol errors, and plan identity were incomplete

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** A duplicate request could be submitted, transport/protocol errors could be mislabeled as rejection, and accepted requests could lose provenance.
- **Runtime evidence:** The canary exposed explicit allow_live_upload requirements, two unresolved NAL trackers, and HTTP 406/upstream 4002 JSON Not readable.
- **Source evidence:** Both upload paths now require trustworthy remote and local tracker inventory, carry allow_live_upload, retain bounded plan identity/fingerprint, classify 406/4002 as non-retryable protocol error, and treat executor failure as acceptance unknown.
- **Reproduction:** Unavailable/corrupt tracker inventory, equivalent pending tracker, dry-run/live arguments, nested/executor 406/4002, transport failure, and profile mismatch fixtures.
- **Candidate files:** apps/nightly_rca/grasshopper_contract.py, apps/nightly_rca/state.py, apps/nightly_rca/phases.py
- **Acceptance tests:** apps/nightly_rca/tests/test_grasshopper_call_path_hardening.py, tests/test_post_canary_hardening_20260807.py
- **Deployment:** Jake local/dev first; actual Grasshopper wire repair remains external
- **Rollback:** Revert Nightly adapter changes; do not retry unresolved external requests automatically.
- **Aliases/recurrences:** GH-HTTP-406-MISCLASSIFIED-AS-BUSINESS-REJECTION, JAKE-06, JAKE-07, JAKE-08, JAKE-09, JAKE-10, JAKE-11, JAKE-12, JAKE-13, NIGHTLY-LOCAL-TRACKER-LEDGER-ERROR-AS-EMPTY, STATUS-DISTINCT-BLOCKERS-COLLAPSED

### INCIDENT-SCENE-CONTRACT-RENDER-BOUNDS — Incident Scene query, expansion, and rendering contracts were incomplete and unbounded

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Turn timeouts, invalid inline SVG, misleading zero results, and lost artifacts.
- **Runtime evidence:** ~47.8 MB SVG, 53,278 default relationships, mixed 2024/2026 axis, silent invalid classifications, missing totals/cursors/counts, null expansion windows.
- **Source evidence:** Query and renderer now expose validation, pagination, canonical windows, deterministic slices, caps, omissions, hashes, and artifact-only delivery.
- **Reproduction:** Existing/new Scene unit fixtures; no live persisted Scene mutation.
- **Candidate files:** app/scenes/query.py, app/scenes/render_svg.py, app/scenes/tools.py, app/scenes/persistence.py
- **Acceptance tests:** tests/test_incident_scene_contract_hardening.py, tests/test_incident_scene_persistence_tools_v1.py
- **Deployment:** S3 dev first; production manual
- **Rollback:** Revert Scene commit; persisted canonical Scene files remain unchanged.
- **Aliases/recurrences:** SCENE-CLASSIFICATION-SILENT-ZERO, SCENE-EPHEMERAL-ARTIFACT-LOSS, SCENE-EXPANSION-WINDOW-NULL, SCENE-GLOBAL-TIME-AXIS-COLLAPSE, SCENE-LARGE-INLINE-SVG, SCENE-QUERY-PAGINATION-MISSING, SCENE-RELATIONSHIP-GROWTH, SCENE-RENDER-COUNTS-MISSING

### INFRA-S3-ARTIFACT-FETCH-NONRESILIENT — S3 artifact fetches were single-shot and transport failures were misclassified

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Lost artifacts and incorrect missing-object conclusions.
- **Runtime evidence:** head-object/cp and session closure interrupted artifact reads.
- **Source evidence:** Bounded ranged download supports retry, resume, temporary integrity, final length, checksum/ETag validation, and typed errors.
- **Reproduction:** Interrupted synthetic range download resumes without corruption; transport closure distinct from not found.
- **Candidate files:** app/s3_resumable_fetch.py, app/server.py
- **Acceptance tests:** tests/test_s3_resumable_fetch_hardening.py
- **Deployment:** S3 dev first; production manual
- **Rollback:** Revert transfer helper commit.

### NAL-CURRENT-PLAN-TEST-INCOMPLETE — Current Grasshopper NAL plan/upload canary has not been executed

- **Status:** `RUNTIME_EVIDENCE_NEEDED`
- **Impact:** End-to-end production behavior remains unproven.
- **Runtime evidence:** Historical S3 NAL files exist, but no current Grasshopper plan, duplicate preflight, upload request, tracker write, or receipt canary was completed.
- **Source evidence:** Source and clean-room contract tests pass; runtime mutation was intentionally not attempted.
- **Reproduction:** After credential rotation and dev deployment: bounded plan for R1955706171, duplicate preflight, at most one authorized upload, one tracker write, one receipt check.
- **Candidate files:** none
- **Acceptance tests:** bounded dev NAL functional canary
- **Deployment:** Dev runtime only after security clearance
- **Rollback:** No source rollback; cancel/close test tracker per runbook.
- **Aliases/recurrences:** GH-PROFILE-UPLOAD-LIVE-CONTRACT-NOT-INTEGRATION_TESTED

### ORCH-CAPABILITY-PREFLIGHT-MISSING — Child tasks could start without a sufficient usable capability binding

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Invented execution, wasted turns, orphaned results, and policy confusion.
- **Runtime evidence:** File-writing and repository-inspection children were launched without writer or shell/filesystem tools.
- **Source evidence:** Child narrowing was called without declared required tools/toolsets/capabilities and prompts named agent_run_python unconditionally.
- **Reproduction:** Declare write/repository/shell requirements against insufficient and sufficient synthetic bindings.
- **Candidate files:** app/agent_mode/task_capability_plan.py, app/agent_mode/child_conversation.py, app/agent_mode/mcop_tools.py
- **Acceptance tests:** tests/test_child_capability_preflight_hardening.py
- **Deployment:** Jake local/dev first
- **Rollback:** Revert MCOP capability-plan commit.
- **Aliases/recurrences:** ORCH-CHILD-CAPABILITY-MISMATCH-REPRO

### ORCH-COMPLETION-CONTRACT-ABANDONMENT — Structured multi-phase tasks could terminate before required report fields were produced

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** The model switched to branch review or explanation mode and falsely implied completion.
- **Runtime evidence:** A repair task with explicit downstream phases stopped after early source inspection/eight old tests and asked what to tackle next.
- **Source evidence:** Explicit FINAL REQUIRED REPORT / ABSOLUTE COMPLETION GATE contracts are parsed deterministically; a tool-less terminal response gets one bounded repair invocation and then an honest INCOMPLETE_EXECUTION_CONTRACT.
- **Reproduction:** Provide a safe structured task with required FIELD= entries and a model response that stops early or merely echoes placeholders.
- **Candidate files:** app/agent/completion_contract.py, app/agent/agents/agentic_rag.py
- **Acceptance tests:** tests/test_post_canary_hardening_20260807.py
- **Deployment:** Jake local/dev first
- **Rollback:** Revert completion guard; no checkpoint schema data depends on it.
- **Aliases/recurrences:** ORCH-EXECUTION-TASK-STOPPED-FOR-SCOPE-CHOICE, ORCH-IMPLEMENTATION-TASK-CONVERTED-TO-EXPLANATION, ORCH-REQUIRED-PHASE-ABANDONMENT, PATCH-INCOMPLETE-BRANCH-PRESENTED-FOR-PUSH

### ORCH-CROSS-DOMAIN-CONTEXT-CONTAMINATION — Current-turn task and methodology displaced by stale checkpoint state

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Wrong methodology, lost plan, repeated questions, and cross-domain actions.
- **Runtime evidence:** Stale repo_checkout_local_deploy state displaced Incident Scene work and a later turn was described as a new conversation.
- **Source evidence:** Typed continuity state is now checkpointed and restored only for content-free continuations in the same authoritative environment.
- **Reproduction:** Explicit Incident Scene task with stale repo checkpoint; then same-chat Steps 3-5 continuation after compression.
- **Candidate files:** app/agent/continuity_policy.py, app/agent/tool_execution_policy.py, app/agent/tool_policy_state.py, app/agent/agents/agentic_rag.py
- **Acceptance tests:** tests/test_continuity_scope_ordering_hardening.py
- **Deployment:** Jake local/dev first
- **Rollback:** Revert continuity commit; checkpoint fields are backward-safe defaults.

### ORCH-ENVIRONMENT-IDENTITY-DRIFT — Remote endpoints could be mistaken for the agent or repository host

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Wrong repository, logs, config, and deployment target.
- **Runtime evidence:** The agent moved among .35, .47, EKS, ArgoCD, dashboards, and MCP/HTTP endpoints despite an explicit pinned target.
- **Source evidence:** Operational identity separates pinned target/runtime hostname from typed remote endpoint fields.
- **Reproduction:** Pinned .35 plus verified hostname; inject MCP, HTTP, client, and repository-origin hosts and assert no self-location override.
- **Candidate files:** app/agent/operational_identity.py, app/agent/audited_tool_node.py, app/agent/tool_execution_audit.py
- **Acceptance tests:** tests/test_operational_identity_and_host_attribution_hardening.py
- **Deployment:** Jake local/dev first
- **Rollback:** Revert host-attribution integration.
- **Aliases/recurrences:** OBSERVABILITY-TOOL-ENDPOINT-HOST-UNATTRIBUTED, ORCH-AUTHORITATIVE-TARGET-OVERRIDDEN-BY-INFERENCE, ORCH-SELF-LOCATION-INFERRED-FROM-TOOL-ENDPOINT

### ORCH-EVIDENCELESS-PROGRESS-NARRATION — Progress claims were not structurally tied to load-bearing evidence

- **Status:** `DEFERRED_WITH_REASON`
- **Impact:** Operators cannot distinguish real progress from narration.
- **Runtime evidence:** Found/confirmed/verified language appeared without a tool result, source line, request ID, or artifact.
- **Source evidence:** Tool audit and typed evidence records now exist, but no end-to-end progress-event schema consumer is present in the uploaded backend.
- **Reproduction:** Requires the production streaming/UI progress path, absent from the clean-room runtime.
- **Candidate files:** app/agent/tool_execution_audit.py, app/message/*
- **Acceptance tests:** test_progress_event_requires_evidence_reference
- **Deployment:** Future full-stack Jake patch
- **Rollback:** N/A until implemented.
- **Aliases/recurrences:** REPORT-INTERMEDIATE-HYPOTHESIS-NOT-CLEARED
- **Deferred reason:** A safe fix requires the missing streaming progress-event consumer; a brittle lexical ban was intentionally not added.

### ORCH-NO-PROGRESS-TOOL-THRASH — Missing required capability caused phase bypass and unrelated fallback loops

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Long stalls, wrong environment, fabricated progress, and turn timeouts.
- **Runtime evidence:** Grasshopper failure led to S3 reads, children, web, EKS, ArgoCD, dashboards, ports, and other unrelated surfaces.
- **Source evidence:** A semantic no-progress controller now consumes only authoritative tool-result envelopes and is called before further orchestration. Current-parent requirements are now derived only from current prompt/management/operational intent; child-scope records and phantom qualified suffixes are excluded.
- **Reproduction:** Required tool blocked; one exact activation allowed at most once; two no-progress results terminate.
- **Candidate files:** app/agent/no_progress_controller.py, app/agent/agents/agentic_rag.py
- **Acceptance tests:** tests/test_no_progress_and_evidence_status_hardening.py, tests/test_post_canary_hardening_20260807.py
- **Deployment:** Jake local/dev first
- **Rollback:** Revert no-progress controller integration.
- **Aliases/recurrences:** JAKE-16, ORCH-BLOCKED-PHASE-CONTINUES-WITH-SUBSTITUTES, ORCH-EXPLICIT-STOP-CONDITION-IGNORED, ORCH-GRASSHOPPER-WRITE-REQUEST-SUBSTITUTED-BY-S3-READ, ORCH-HEALTH-VS-FUNCTIONAL-TEST-CONFLATION, ORCH-MISSING-TERMINAL-BLOCKED-STATE, ORCH-PHASE-GATE-BYPASS, ORCH-PHASE0-IDENTITY-GATE-BYPASS, ORCH-PRIVATE-SOURCE-PUBLIC-SEARCH-FALLBACK, ORCH-UNBOUNDED-PARALLEL-SEARCH, ORCH-UNRELATED-FALLBACK-SUBSTITUTION, ORCH-USER-ORDERING-CONSTRAINT-IGNORED, PHANTOM-NO-PROGRESS-CAPABILITY-CONTAMINATION

### ORCH-PARENT-ONLY-VIOLATION-RECURRENCE — Explicit parent-only execution constraint was not represented as durable policy

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Violates user execution topology and can reproduce the defects under repair.
- **Runtime evidence:** The debug corpus records child spawning after repeated parent-only instructions.
- **Source evidence:** A checkpointed parent-only execution constraint now removes MCOP spawn tools from the model-facing tray, is copied into the server-owned audit context, and is independently enforced by the execution gate and both spawn entrypoints.
- **Reproduction:** Current turn says parent thread only; assert agent_spawn_task/agent_spawn_parallel are not model-facing and a stale direct call returns BLOCKED_PARENT_ONLY_TURN with zero writes.
- **Candidate files:** app/agent/execution_constraints.py, app/agent/tool_profiles.py, app/agent/tool_policy_state.py, app/agent/tool_execution_policy.py, app/agent/tool_execution_gate.py, app/agent/tool_execution_audit.py, app/agent_mode/mcop_tools.py, app/agent/agents/agentic_rag.py
- **Acceptance tests:** tests/test_post_canary_hardening_20260807.py
- **Deployment:** Jake local/dev first
- **Rollback:** Revert the post-canary hardening commit; checkpoint migration defaults the flag false.
- **Aliases/recurrences:** AUDIT-PARENT-ONLY-BLOCK-MISCLASSIFIED

### ORCH-RESULT-ORPHANING — Blocked/partial child packets and executed-tool evidence lost status or provenance

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Parent conclusions detach from actual execution and turns time out.
- **Runtime evidence:** Children were reported completed despite blocked packets; assistant-authored tool-like JSON was mixed with execution evidence; large repeated reports amplified.
- **Source evidence:** Child results preserve blocked/partial status, include audit IDs, and bound aggregation; no-progress recognizes only typed tool records.
- **Reproduction:** Blocked child packet, partial packet, assistant-authored JSON, and fallback ToolMessage pairing.
- **Candidate files:** app/agent_mode/child_conversation.py, app/agent/no_progress_controller.py, app/agent/tool_execution_gate.py, app/agent/tool_execution_audit.py
- **Acceptance tests:** tests/test_child_capability_preflight_hardening.py, tests/test_scene_persistence_policy_e1.py
- **Deployment:** Jake local/dev first
- **Rollback:** Revert child packet/audit changes.

### PROFILE-METADATA-DROPPED-BEFORE-ACQUISITION — Nightly RCA dropped S3 profile metadata and misread nested Grasshopper contracts

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Zero-file plans, incorrect retry/safety state, missing external IDs, and misleading notifications.
- **Runtime evidence:** atv_reboot_instability was sent literally; nested selected counts and request IDs were lost; zero files and unattempted upload collapsed into generic coverage messages.
- **Source evidence:** S3 maps issue taxonomy to atv_core; Jake propagates canonical profile, file families, caps, and parses nested plan/upload layers in the production Phase 5/6 path.
- **Reproduction:** Real-name synthetic catalog and nested response fixtures exercise production call path.
- **Candidate files:** apps/nightly_rca/grasshopper_contract.py, apps/nightly_rca/phases.py, app/issue_profile_registry.py
- **Acceptance tests:** apps/nightly_rca/tests/test_grasshopper_call_path_hardening.py, tests/test_grasshopper_profile_contract_hardening.py
- **Deployment:** Jake local/dev and S3 dev first
- **Rollback:** Revert Nightly/S3 integration commits together.
- **Aliases/recurrences:** GH-PROFILE-NAMESPACE-MISMATCH, NIGHTLY-NOT-SUBMITTED-REPORTED-NOT-ACCEPTED, NIGHTLY-SUBMIT-SKIPPED-REASON-NOT-EXPOSED, PROFILE-ALIAS-PROPOSED-AS-CANONICAL-FIX, ROOT-CAUSE-FALSE-STATUS-NORMALIZATION, STATUS-DISTINCT-BLOCKERS-COLLAPSED, TEST-FIXTURE-GRASSHOPPER-CONTRACT-DRIFT

### SEC-CREDENTIALS-PASTED-IN-CONVERSATION — Raw credentials were pasted into the conversation transcript

- **Status:** `DEFERRED_WITH_REASON`
- **Impact:** Credential must be assumed exposed.
- **Runtime evidence:** The supplied transcript contains credential material.
- **Source evidence:** No value was copied into patches, reports, or archives; deliverables use [REDACTED_GRASSHOPPER_AUTH_SECRET].
- **Reproduction:** Not repeated.
- **Candidate files:** none
- **Acceptance tests:** post-rotation authentication canary, secret scan
- **Deployment:** Approved credential rotation
- **Rollback:** Rotation rollback follows secret-management procedure, never source history.
- **Deferred reason:** Rotation/revocation requires operator approval and was explicitly outside this environment.

### SEC-GRASSHOPPER-SOURCE-TLS-CONTAINMENT — Legacy Grasshopper source exposed a credential and disabled TLS verification

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Credential compromise and man-in-the-middle exposure.
- **Runtime evidence:** Uploaded source contained one raw legacy auth literal and verify=False calls; deployment script targeted a stale model.
- **Source evidence:** Final source retrieves secrets server-side, verifies TLS by default, supports CA bundle and explicit audited emergency override, and quarantines legacy deploy script.
- **Reproduction:** Secret/TLS source scan and pure-function security tests; no credential value emitted.
- **Candidate files:** app/grasshopper_client.py, app/server.py, deploy.sh
- **Acceptance tests:** tests/test_security_containment_hardening.py
- **Deployment:** Grasshopper dev first; rotation separately approved
- **Rollback:** Rollback source only after restoring secure secret retrieval/TLS; never restore the legacy literal.
- **Aliases/recurrences:** GRASSHOPPER-HARDCODED-AUTH-SECRET, GRASSHOPPER-LEGACY-DEPLOYMENT-DRIFT, GRASSHOPPER-TLS-VERIFY-DISABLED

### TEST-PASS-CLAIM-LACKS-CALL-PATH-COVERAGE — Green unit tests and baseline failures were overstated without environment/call-path proof

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Incomplete patches are pushed or deployed as complete.
- **Runtime evidence:** 119 passing tests were presented as completion while production Phase 5 was unwired; contradictory 101/1 vs 69/33 baselines were not reconciled; failures were called pre-existing without proof.
- **Source evidence:** This program added failing-before production-path tests, recorded dependency-blocked suites honestly, ran complete relevant suites, and validated hash-gated clean-room application.
- **Reproduction:** Feature adapter unit test passes while Phase 5 metadata is absent; collection failure with missing langchain_core.
- **Candidate files:** apps/nightly_rca/tests/test_grasshopper_call_path_hardening.py, docs/engineering-defects/*
- **Acceptance tests:** clean-room test evidence, baseline/post-patch evidence comparison
- **Deployment:** Process and CI
- **Rollback:** Revert source commits; evidence artifacts remain immutable.
- **Aliases/recurrences:** ORCH-EXECUTION-TASK-STOPPED-FOR-SCOPE-CHOICE, ORCH-IMPLEMENTATION-TASK-CONVERTED-TO-EXPLANATION, PATCH-CROSS-REPO-WORKTREE-CONFLATION, PATCH-INCOMPLETE-BRANCH-PRESENTED-FOR-PUSH, PATCH-SCOPE-DROPS-VERIFIED-DEFECTS, PATCH-UNIT-TESTED-BUT-NOT-WIRED, TEST-BASELINE-CONTRADICTION-UNRESOLVED, TEST-FAILURES-LABELED-PREEXISTING-WITHOUT-PROOF

### TRACKER-REQUEST-ID-PROVENANCE-MISSING — Tracker, local correlation, external request IDs, currentness, and count scope were conflated

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Duplicate submissions, bad receipt reconciliation, and contradictory reports.
- **Runtime evidence:** Legacy July rows appeared current; request-like IDs lacked source; counts mixed direct/cross-reference/logical/physical scopes.
- **Source evidence:** Pending ledger normalizes typed identifiers, origin run/currentness, physical identity, and explicit count scope; reports expose provenance. Legacy requested_log_types are boundedly normalized without rewriting raw values; corrupt local ledgers are preserved and reported unavailable; both Phase 5 and Phase 6 fail closed before upload.
- **Reproduction:** Typed current row, historical row, legacy opaque request ID, duplicate physical tracker rows.
- **Candidate files:** apps/nightly_rca/notify.py, apps/nightly_rca/phases.py, apps/nightly_rca/report.py, apps/nightly_rca/state.py
- **Acceptance tests:** apps/nightly_rca/tests/test_effective_config_and_tracker_provenance_hardening.py, apps/nightly_rca/tests/test_grasshopper_call_path_hardening.py, tests/test_post_canary_hardening_20260807.py
- **Deployment:** Jake local/dev first
- **Rollback:** Revert tracker provenance changes; retain backup of pending ledger.
- **Aliases/recurrences:** LEGACY-TRACKER-ID-PROVENANCE-ABSENT, NIGHTLY-STALE-TRACKER-STATE-PRESENTED-AS-CURRENT, REPORT-EVIDENCE-COUNT-INCONSISTENT, REPORT-RECORD-SCOPE-COUNT-AMBIGUITY, TRACKER-PROLIFERATION-POSSIBLE-DUPLICATES

### TYPED-ID-ARGUMENT-PROVENANCE — Context identifiers could populate domain identifier arguments

- **Status:** `FIXED_AND_VERIFIED`
- **Impact:** Calls target the wrong object and audit trails become misleading.
- **Runtime evidence:** A chat/workspace UUID was supplied as scene_id and UUID shape alone passed.
- **Source evidence:** Domain identifier validators and source/type audit records now run before executor invocation.
- **Reproduction:** Cross-type chat/workspace UUID to scene_id; valid scene/receiver IDs; invalid semantic shapes.
- **Candidate files:** app/agent/identifier_validation.py, app/agent/tool_execution_gate.py, app/agent/audited_tool_node.py
- **Acceptance tests:** tests/test_typed_identifier_provenance_hardening.py
- **Deployment:** Jake local/dev first
- **Rollback:** Revert identifier gate integration.
- **Aliases/recurrences:** ARGUMENT-CONTEXT-ID-SUBSTITUTED-INTO-SCENE-ID, IDENTIFIER-SEMANTIC-VALIDATION-MISSING


## Post-restart live canary addendum — 2026-08-07 v2.2

- **LIVE PASS — canonical authorization and privileged binding persistence:** all four canonical flags were true; `agent_run_shell` and `agent_run_python` remained requested, eligible, `BOUND`, and in `last_bound_tool_names` on the next turn without a new activation.
- **LIVE PASS — incidental/example activation filtering:** URI/Git/placeholder/example/negated qualified tokens produced empty requested/pending/eligible extra-tool state.
- **LIVE PASS — parent-only MCOP enforcement:** authoritative `tool_policy_state` reported `mcop_children_forbidden=true`; both spawn tools were absent from `last_bound_tool_names`, with backend withholding logs for both.
- **LIVE PASS — Grasshopper status/history truthfulness:** exact read-only tools bound; upstream client exposed no authoritative status/history method (`attempted_methods=[]`); output was `status=unavailable`, coverage `UNKNOWN`, negative conclusion false, uploads zero.
- **LIVE FAIL / v2.2 SOURCE FIX — sequential no-progress:** two sequential `internal_search` `TOOL_NOT_IN_LAST_BINDING` failures separated by AI turns did not terminate. Root cause: `_tail_tool_records` stopped at the first non-tool AI message and unrelated parent fallback failures were skipped when a nominal required set existed. v2.2 collects across AI separators and counts real parent fallback failures globally while preserving child/phantom isolation.
- **LIVE FAIL / v2.2 SOURCE FIX — forced completion contract:** the model called the contract fictional and terminated without required fields. v2.2 injects a server-trusted completion-contract SystemMessage before the initial invocation, retains deterministic post-response enforcement, and logs DETECTED/REPAIR/TERMINAL events.
- **LIVE INCONCLUSIVE — JAKE-17 example authorization contamination:** the attempted canary did not call the binding-status facade, so no live state proof was captured. Source/unit tests remain PASS; rerun after v2.2 deployment.
- **Operational-only fixed defect:** the v2.1 deployment Step 5 runner incorrectly applied whole-worktree `git diff --check` to intentionally preserved dirt. The package runner was hotfixed to check the committed feature delta after exact dirty path/fingerprint/non-overlap gates.
- **Security containment:** embedded dirty `run.py` webhook was removed; source security test and all Nightly tests passed. The scheduled wrapper uses a different external secret. Provider-side revocation/rotation of any exposed legacy webhook remains required.
- **Runtime evidence needed:** an existing chat emitted `Database branch reconstruction yielded 16 messages` vs `checkpoint_messages has 20`; track as `CHAT-HISTORY-CHECKPOINT-COUNT-DIVERGENCE` without mixing it into v2.2.
