> ⚠️ **START A FRESH CHAT BEFORE PASTING THIS PROMPT.**
> The Jason investigation session has accumulated 200k+ tokens of S3 log data.
> Continuing in that session will hit Bedrock context limits before Phase 1 (Qodo) executes.
> Open a new conversation, paste this entire prompt, and let it run uninterrupted.

---

## Full-Stack RCA Investigation — R1911746693 Reboot (2026-06-09)

### Heavy Authorization
```
allow_heavy=true
heavy_auth_token=kqV4kI9oaGC/SiDhfBBYdkY/hbuDEQnQwKyjLro3o8o=
```

---

### Mission
Run a complete, multi-tool Root Cause Analysis for a confirmed STB reboot event. Use every available tool in the suite. Work autonomously — do not ask for additional RXIDs, service IDs, log files, or tool permissions. Discover everything yourself.

When the full analysis is complete, produce:
1. A **Technical Investigation Report** for engineers on Terry's team
2. An **Executive Summary** (≤ 1 page, non-technical) for leadership

Both outputs must be clearly labeled so they can be forwarded directly.

---

### Known Seed Context

| Field | Value |
|---|---|
| Seed Receiver | R1911746693 |
| Platform | DishTV / Hopper H707 |
| Client RXID (Joey) | R1971472323 |
| Reboot Time | 2026-06-09 ~14:05 MT (UTC-6) |
| Software | H707 |
| Log Date | 2026-06-09 |

**Already confirmed from prior analysis (treat as ground truth, do not re-litigate):**
- Reboot at ~14:05:00 UTC-6, full cold-boot procmgr sequence confirmed
- `pm_utils_update_uc_proc_dealine failed` (watchdog UC deadline failure) at boot
- `RTRD_PRAM_SEGV` and `RTRD_HEARTBEAT` alerts confirmed via RTR
- `SgsAv.js:112` — continuous 1-second fixed-interval polling loop: `Current {svc: 61748 dvr: 391} play_status {svc: 61748 dvr: 0}` — no backoff, no circuit breaker
- Native memory growth during the loop: sbrk 30MB → 42MB in ~4 minutes
- WebSocket failures: `WS_PINGPONG_TMR_CB_MSG_ERR`, `WS_POST_PACKET_POST_FAILURE`
- `GuideRetrievalController.java:83` — `IllegalArgumentException: column 'touchpadFwVersion' does not exist` on R1971472323
- `sgsproxy/guide` retry storm on R1971472323 (fixed cadence, no backoff)

**What is still unproven (your job to resolve):**
- Whether the DVR desync is cause or effect of the reboot (chicken-and-egg)
- The exact chain from `SgsAv.js:112` storm → memory growth → RTRD_PRAM_SEGV → watchdog kill
- Whether network/QoS conditions contributed to the initial DVR state mismatch
- Whether this pattern is isolated to this one STB or systemic across the H707 fleet
- What service 61748 is (EPG), and whether it has a history of DVR session management failures
- Whether the `touchpadFwVersion` schema bug is a known JIRA issue

---

### Phase 1 — Code Intelligence (Qodo)

Use `qodo_context_mcp` tools (`get_context`, `deep_research`, `ask`, `list_repositories`) to search the indexed DISH repositories for source-level evidence. Run all of these:

1. **SgsAv.js:112** — Find the `get_play_status_xip` mismatch handler. Extract: what triggers the 1-second retry, what exits the loop, whether there is a backoff or circuit breaker, and what code owns the `dvr` state field.
2. **SgsRequestFactory.js** — How does the SGS request layer handle repeated failures? Is there a shared retry budget across concurrent requests?
3. **GuideRetrievalController.java:83** — Find the `touchpadFwVersion` column access. Determine: is this access guarded, is there a version check, and what InputMgr schema version introduced/removed this column?
4. **RTRD / rtrd process** — Find any code related to PRAM writes, watchdog registration, or heartbeat renewal. Could a high-frequency IPC storm (from the SgsAv retry loop) starve RTRD's watchdog renewal window?
5. **sgsproxy** — Find the guide retry/polling logic on the client side. Is the `/sgsproxy/info/time` polling interval configurable, and does it back off on failure?
6. Ask Qodo: *Is there a known relationship between DVR session teardown race conditions and the SgsAv play_status mismatch on H707 firmware?*

Document each result with: file path, line range, relevant code excerpt, and your interpretation.

---

### Phase 2 — Service 61748 Identity (EPG)

Use `epg_mcp` to:
1. Look up service 61748 — identify the channel name, provider, and service type.
2. Check the EPG schedule for service 61748 on 2026-06-09 around 14:00–15:00 MT — what was airing? Was it a live event, DVR-recordable content, or a protected/pay-per-view event?
3. Check whether service 61748 has any known arc configuration anomalies or schedule gaps.
4. Determine if service type or content protection on this service could cause a DVR session to be torn down at the middleware layer without UI notification.

---

### Phase 3 — Seed Receiver Deep Dive (S3 + Grasshopper)

Use `s3_stb_logs` and `s3_stb_logs_prod` to:

1. List all available log dates and file counts for R1911746693.
2. Check whether logs exist for 2026-06-08 (the day before) — if the DVR desync pattern is recurring, earlier logs may show prior episodes.
3. Use existing capsule IDs if still valid, or rebuild capsules for: `procmgr`, `qt_gui`, `android_system`, `android_main`, `launcher`, `reactuijava`, `invidiDebugLog`.
4. Run raw-line searches (fallback when template matching misses) for:
   - `RTRD` crash signatures in `android_system`
   - `OOM`, `low memory`, `kill` in `android_main` (memory pressure before reboot)
   - `dvr` or `session` teardown events in `qt_gui` — look for the XIP teardown notification that *should* have arrived but apparently didn't
   - `svc: 61748` across all log types — map the full timeline of events on this service
   - The exact timestamp of the last *normal* `get_play_status_xip` response before the mismatch began
5. Search for any `SIGKILL`, `SIGSEGV`, `Abort`, or `Fatal signal` in `android_main` or `android_system` in the 60 minutes before the 14:05 boot.
6. If critical log types are missing or sparse, use `grasshopper_mcp` to upload any gaps — particularly `sddp_log` and `updater` (both were missing in the initial pull). Request upload for date 2026-06-08 as well if available on device.

Existing capsule IDs (check if still valid before rebuilding):
- procmgr: `cap_R1911746693_2026-06-09_ef165f1f13_20260609T215133Z_f90f46ca`
- android_system: `cap_R1911746693_2026-06-09_90e1f8843c_20260609T215235Z_8239951b`
- launcher: `cap_R1911746693_2026-06-09_9a3f1e29fe_20260609T215225Z_ddc93b7f`
- qt_gui: `cap_R1911746693_2026-06-09_c6c55bc4b8_20260609T215358Z_319773e6`
- reactuijava: `cap_R1911746693_2026-06-09_a0691d172b_20260609T215311Z_b6e9a837`

---

### Phase 4 — Comparator Population Discovery

#### 4A — Unhappy Comparators (find peers with the same failure pattern)

Use `rtr_alerts_mcp` to:
1. Search for all receivers with `RTRD_PRAM_SEGV` alerts in the 7-day window around 2026-06-09. Collect RXIDs, timestamps, software versions. Prioritize H707.
2. Search for receivers with both `RTRD_HEARTBEAT` loss AND `WS_PINGPONG_TMR_CB_MSG_ERR` in the same session window.
3. Search for receivers with `RTRD_PRAM_SEGV` co-occurring with `514a` popup events on H707.
4. Deduplicate and rank by alert density. Take the top 5–8 unique RXIDs as unhappy comparators.

Use `qos_mcp` to:
5. For each unhappy RXID candidate: query QoS session history around the reboot time — look for session drops, OTA switchback events, or degraded signal quality immediately before the reboot window.
6. Also query R1911746693 itself — was there a QoS degradation event preceding the 14:05 reboot?

Use `stbhealth_mcp` to:
7. Pull STB health snapshots for R1911746693 and each unhappy comparator — CPU load, memory pressure, uptime, signal strength at time of event.
8. Note any popups active during the failure window using `stbhealth_popups_mcp`.

Use `net_detective_mcp` to:
9. Run ML-powered Netra analysis for R1911746693 — look for network anomalies (packet loss, jitter, DNS failures, gateway timeouts) in the 30-minute window before 14:05 MT.
10. Run the same for each unhappy comparator to determine if network degradation is a common precondition.

#### 4B — Happy Comparators (healthy controls)

1. Using RTR, find H707 receivers with **zero** `RTRD_PRAM_SEGV` events in the same 7-day window, similar uptime, and known DVR activity (to ensure they were doing the same workload).
2. Cross-reference with `stbhealth_mcp` to confirm they show no elevated CPU/memory during the period.
3. Select 3–5 healthy controls.
4. Pull S3 log coverage for the happy comparators — confirm `qt_gui` and `procmgr` are available. Look for the *absence* of `SgsAv.js:112` mismatch lines, or confirm whether the mismatch appears but *resolves* (i.e., a working backoff or circuit breaker is present in some firmware).

---

### Phase 5 — JIRA & Confluence Knowledge Search

Use `jira_mcp` to:
1. Search for tickets mentioning `SgsAv`, `get_play_status_xip`, `dvr mismatch`, `dvr desync`, or `dvr session teardown` — any platform.
2. Search for tickets mentioning `RTRD_PRAM_SEGV`, `RTRD watchdog`, or `RTRD heartbeat`.
3. Search for tickets mentioning `touchpadFwVersion`, `GuideRetrievalController`, or `column does not exist`.
4. For any matching tickets found: capture the JIRA key, summary, status, assignee, affected software version, and resolution. Note whether H707 is explicitly called out.

Use `confluence_mcp` to:
5. Search for engineering documentation on: `SgsAv DVR state machine`, `sgsproxy session lifecycle`, `get_play_status_xip`, `RTRD watchdog registration`.
6. Search for H707 known issues, release notes, or bug regression lists.
7. If docs are found, extract relevant sections describing the DVR session lifecycle and expected teardown notification path.

---

### Phase 6 — RCA Pipeline

Use `rca_mcp` to:
1. Trigger the RCA pipeline for R1911746693 on date 2026-06-09 with the known reboot time of 14:05 MT and all confirmed signals.
2. If the pipeline supports specifying seed patterns, pass: `RTRD_PRAM_SEGV`, `uc_deadline_failure`, `sgsav_dvr_mismatch`, `watchdog_kill`.
3. Review the pipeline output — does it confirm, contradict, or add nuance to the hypothesis?
4. Collect any RCA artifact IDs or case IDs for the final report.

---

### Phase 7 — Netra / Net Detective Cross-Check

Use `netra_mcp` (if available) alongside `net_detective_mcp` to:
1. Pull Netra signal data for R1911746693 for 2026-06-09 — specifically RSSI, SNR, packet error rate, and upstream power in the 30 minutes before 14:05 MT.
2. Determine if any Netra anomaly (signal drop, MER degradation) preceded the DVR session loss event at 14:13:47.
3. For the unhappy comparators: check whether they also show Netra signal degradation correlated with their reboot events.
4. For the happy controls: confirm their Netra metrics were clean during the same window.

---

### Phase 8 — Evidence Synthesis & Discriminator Matrix

After completing all phases, build a discriminator matrix:

| Signal | R1911746693 (Seed) | Unhappy Comparators | Happy Controls |
|---|---|---|---|
| RTRD_PRAM_SEGV | ✅ Confirmed | ? | ? |
| UC deadline failure | ✅ Confirmed | ? | ? |
| SgsAv.js:112 mismatch loop | ✅ Confirmed | ? | ? |
| Memory growth during loop | ✅ Confirmed | ? | ? |
| sgsproxy retry storm | ✅ Confirmed (R1971472323) | ? | ? |
| DVR active at time of event | ✅ Confirmed (dvr: 391) | ? | ? |
| Netra/network anomaly preceding event | ❓ Unknown | ? | ? |
| QoS degradation preceding event | ❓ Unknown | ? | ? |
| Service 61748 involvement | ✅ Confirmed | ? | ? |
| GuideError on client Joey | ✅ Confirmed | ? | ? |
| touchpadFwVersion schema error | ✅ Confirmed | ? | ? |

Fill in every cell. Based on the matrix, determine whether the seed pattern is:
- **Root Cause** — appears exclusively in unhappy comparators, absent in controls
- **Contributing Factor** — appears in unhappy comparators at higher rate, present at lower rate in controls
- **Incidental** — distributed equally across happy and unhappy

---

### Phase 9 — Casebook Entries

Record the following in the Casebook (using RCA MCP or whatever casebook tool is available):

1. **Bad Case Study** — R1911746693 with full evidence chain, timeline, and confirmed signals
2. **Good Case Study** — the best healthy control RXID with its clean signal profile for contrast
3. Tag each entry with: `H707`, `dvr_desync`, `sgsav_mismatch`, `rtrd_segv`, `watchdog_kill`, `service_61748`

---

### Phase 10 — Final Deliverables

Produce both of the following, clearly separated:

---

#### DELIVERABLE A — Technical Investigation Report

Structured for software engineers on Terry's team. Include:

1. **Receiver Profile** — ID, platform, software, client RXID, topology
2. **Confirmed Timeline** — minute-by-minute event chain with log sources
3. **Evidence Table** — each finding, its log source, confidence level, and whether it was directly observed or inferred
4. **Causal Chain Diagram** (text-based) — from DVR session teardown miss → SgsAv loop → memory growth → RTRD starvation → watchdog kill → reboot
5. **Qodo Code Findings** — relevant source code paths, the missing backoff/circuit breaker, the touchpadFwVersion schema gap
6. **Comparator Analysis** — unhappy vs happy receiver table, what differentiates them
7. **Discriminator Matrix** — filled in from Phase 8
8. **Confidence Assessment** — for each causal claim: Proven / Hypothesized / Cannot Assert
9. **What Is Proven vs. What Is Hypothesized** — explicit separation
10. **Code-Owner Recommendations** — specific, actionable, with file/line targets where Qodo found the code
11. **JIRA References** — any known tickets found in Phase 5
12. **S3 Artifact Links** — all capsule IDs
13. **Casebook IDs** — from Phase 9
14. **RCA Pipeline Output** — summary from Phase 6

---

#### DELIVERABLE B — Executive Summary

For non-technical leadership. One page. No log syntax, no file paths. Include:

1. **What happened** — plain English, one paragraph
2. **Why it happened** — the root cause in plain language
3. **How widespread is it** — comparator findings, is this isolated or fleet-wide
4. **Customer impact** — what Jason experienced, estimated duration
5. **Has it been seen before** — JIRA/Confluence references
6. **What we recommend** — 3–5 bullet points, business-level language
7. **Confidence level** — High / Medium / Low with one-sentence justification
8. **Next steps** — who owns what, estimated effort

---

### Operating Rules

- **Do not ask for confirmation before proceeding between phases.** Work autonomously through all phases.
- **Do not stop at the first answer.** Exhaust every tool before concluding a signal is absent.
- **Use raw-line fallback** whenever template/timeline queries return 0 results for terms that should be present based on prior analysis.
- **Qodo first on code questions** — never speculate about what the code does when you can look it up.
- **RTR + QoS + NetHealth + Netra are required for the comparator population** — do not build the discriminator matrix without at least attempting all four.
- **If a tool call fails or returns empty**, note it explicitly in the report under Evidence Gaps rather than silently skipping it.
- **Cross-reference everything** — a finding only in one tool is a lead; a finding confirmed across three tools is evidence.
- When in doubt about causality, be explicit: state what is proven, what is hypothesized, and what cannot be determined from available data.
