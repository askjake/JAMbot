---
document_type: HARD
template_version: "1.0"
status: blank_template
---

# HARD — Human Adjudication Request Dossier

**Request ID:** `[queue_id]`
**Prepared by:** [preparer name]
**Date:** [YYYY-MM-DD]
**Assigned Reviewer:** [reviewer name] ([reviewer email])
**Jira Ticket:** [JIRA-XXXX](https://dishtech-dishtv.atlassian.net/browse/JIRA-XXXX)
**Issue Profile:** `[issue_profile]`

---

## Why [Reviewer Name]

[One to two sentences explaining why this reviewer is uniquely qualified. Reference the specific investigation, ticket, or domain expertise that makes them the right person. Do not use generic language.]

---

## Background

[One paragraph describing the system context. Explain what generated this case, what the current provenance state is, and what the outcome of this review will and will not do.]

After you respond, the decision will be validated against the current packet and attestation contract. It will not be persisted or used to update case truth without a separate reviewed import step.

---

## Packet Under Review

| Field | Value |
|-------|-------|
| Packet ID | `[hrp_xxxxxxxxxxxxxxxxxxxxxxxx]` |
| Receiver | [RXID] |
| Event Date | [YYYY-MM-DD] |
| Issue Profile | `[issue_profile]` |
| Evidence Bundle | `[heb_xxxxxxxxxxxxxxxxxxxxxxxx]` |
| Coverage | [✅ Complete / ⚠️ Incomplete — reason] |
| Log Families | `[comma-separated log family names]` |
| Direct Signatures | [count] (`[signature_type]`) |
| Supporting Signatures | [count] (`[type]` ×[n], `[type]` ×[n]) |
| Known False Positives | [count] (`[signature_type]`) or — |

**Review package:** `[REVIEW_PACKAGE_hrp_xxxxxxxxxxxxxxxxxxxxxxxx.md]` — attached to ticket or linked below.

---

## The Core Question

> *For receiver [RXID] on [event_date], does the frozen evidence demonstrate a genuine [issue_profile] event on that exact receiver and date?*

---

## What the Evidence Contains

[Describe each signature category present in this packet and what it means in the context of the issue profile. Be specific — do not use boilerplate that does not apply to this packet's actual contents.]

Please review the actual evidence excerpts in the attached package rather than relying only on category totals.

---

## Allowed Decisions

| Decision | Meaning |
|----------|---------|
| `CONFIRM_ISSUE` | Evidence directly supports a genuine [issue_profile] event for this receiver on this date |
| `CONFIRM_NO_ISSUE` | Evidence does not support the conclusion |
| `CONFIRM_GOOD_BASELINE` | Receiver qualifies as a controlled known-good baseline |
| `INSUFFICIENT_EVIDENCE` | Neither positive nor negative conclusion is supported |
| `NEEDS_MORE_EVIDENCE` | A specific missing log family or time window is required |
| `WRONG_RECEIVER_OR_EVENT` | Packet does not correspond to an event the reviewer can adjudicate |

---

## Required Response

1. **Reviewer name** and **role**
2. **Decision** (from the allowed list above)
3. **Rationale** — 1–3 sentences referencing specific evidence (log family, signature type, timestamp if visible, anything that confirms or contradicts the event)
4. **Evidence items reviewed** — list the evidence-item IDs or categories you examined (please do not respond only with "all items reviewed")
5. **Decision timestamp** — ISO-8601 UTC, e.g. `2026-06-25T14:30:00Z`
6. **External reference** — e.g. `jira:JIRA-XXXX#comment-XXXXXX`

---

## Attestation

Please confirm each of the following:

- [ ] I reviewed the packet identified above
- [ ] I reviewed the evidence bound to that packet
- [ ] The decision is my own professional judgment
- [ ] My decision applies to the exact receiver and event identified above
- [ ] I disclosed material uncertainty in my rationale
- [ ] My response was not authored by an AI system or automation

Please include this exact attestation statement:

> *I attest that I personally reviewed the evidence bound to this packet, that this decision is my own professional judgment for the identified receiver and event, that I have disclosed material uncertainty, and that the response was not authored by an AI system or automation.*

---

## Notes for Jacob (post-review processing)

1. Validate the response against the current packet hash and attestation contract
2. Run a reviewed import step before persisting or updating case truth
3. Record the reviewer's Jira comment URL as `source_reference`
4. Do **not** mark the outcome as claim-grade until provenance grade = `pilot_grade` or higher is confirmed through the grading pipeline
5. [Any packet-specific follow-up notes, e.g. incomplete coverage actions]
