from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any


class FakeToolClient:
    def __init__(self, fail_tool_once: str | None = None, receipt_check_error_mode: bool = False, canary_no_logs_mode: str = "no_logs"):
        self.calls: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0
        self.fail_tool_once = fail_tool_once
        self.receipt_check_error_mode = receipt_check_error_mode
        self.canary_no_logs_mode = canary_no_logs_mode  # "no_logs", "auth_fail", "timeout", "tool_not_found", "unexpected_logs", "schema_error"
        self.failed = False
        self.counts: Counter[str] = Counter()
        self.queues: dict[str, dict[str, Any]] = {}
        self.bundles: dict[tuple[str, str], dict[str, Any]] = {}
        self.packets: dict[tuple[str, str], dict[str, Any]] = {}
        self.trackers: dict[str, dict[str, Any]] = {}

    async def call(self, server: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.calls.append({"server": server, "tool": tool_name, "arguments": dict(arguments)})
        self.counts[tool_name] += 1
        try:
            if self.fail_tool_once == tool_name and not self.failed:
                self.failed = True
                raise RuntimeError("simulated one-shot failure")
            return self._response(tool_name, arguments)
        finally:
            self.active -= 1

    def _response(self, tool: str, args: dict[str, Any]) -> Any:
        if tool == "get_tool_info":
            return {"git_sha": "abc123", "code_version": "0.5.16", "lambda_function": "s3-stb-logs-mcp-dev", "s3_bucket": "test-bucket"}
        if tool == "get_heavy_auth_status":
            return {"ok": True, "active": True, "status": "ACTIVE"}
        if tool == "get_issue_profile_catalog":
            return {
                "profiles": {
                    "pmgr_crash_instability": {
                        "title": "PMGR crash instability",
                        "strong_patterns": ["PMGR_UNEXPECTED_EXIT", "RTRD_REBOOT"],
                        "supporting_patterns": ["COLD_BOOT_DETECTED"],
                    },
                    "guide_1031": {
                        "title": "Guide 1031",
                        "strong_patterns": ["(^|_)1031($|_)|^1031$", "LLOTT_SCHEDULE_DOWNLOAD_ERROR"],
                    },
                }
            }
        if tool == "list_parsed_dates":
            return {"dates": [
                "2026-07-21", "2026-07-20", "2026-07-19", "2026-07-18",
                "2026-07-17", "2026-07-16", "2026-07-15", "2026-07-14",
            ]}
        if tool == "list_dates":
            rx = str(args.get("receiver_id", ""))
            if rx == "R0000000001":
                from nightly_rca.transport import ToolTransportError
                mode = self.canary_no_logs_mode
                if mode == "no_logs":
                    raise ToolTransportError(f"No logs found for receiver {rx}")
                elif mode == "auth_fail":
                    raise ToolTransportError("401 Unauthorized: authentication failure")
                elif mode == "authz_fail":
                    raise ToolTransportError("403 Forbidden: AccessDenied")
                elif mode == "timeout":
                    raise ToolTransportError("timeout: connection timed out after 120s")
                elif mode == "tool_not_found":
                    raise ToolTransportError("tool_not_found: list_dates is not registered")
                elif mode == "schema_error":
                    raise ToolTransportError("schema validation error: invalid_params for list_dates")
                elif mode == "unexpected_logs":
                    return {"dates": ["2026-07-21", "2026-07-20"]}
                else:
                    raise ToolTransportError(f"No logs found for receiver {rx}")
            return {"dates": ["2026-07-21"]}
        if tool == "search_investigation_cases":
            return {"cases": [{
                "case_id": "case-existing", "issue_profile": "pmgr_crash_instability",
                "receiver_id": "R100", "event_date": "2026-07-20", "outcome_status": "unreviewed",
            }]}
        if tool == "list_human_review_queue_items":
            queue_id = str(args.get("queue_id") or "")
            stored = self.queues.get(queue_id)
            if stored is not None:
                items = list(stored.get("items", []))
                return {
                    "ok": True, "result": "HUMAN_REVIEW_QUEUE_ITEMS",
                    "queue_id": queue_id, "queue_hash": stored.get("queue_hash"),
                    "items": items, "item_count": len(items),
                }
            return {"items": [{"queue_item_id": "q1"}], "item_count": 1}
        if tool == "human_review_dashboard_status":
            return {"ok": True, "status": "READY", "cards": 3}
        if tool == "list_packet_integrity_failures":
            return {"items": []}
        if tool == "list_upload_trackers":
            rows = list(self.trackers.values())
            if args.get("pending_only"):
                rows = [row for row in rows if str(row.get("workflow_status") or "").upper() not in {"READY_FOR_ANALYSIS", "COMPLETE", "CLOSED"}]
            return {"ok": True, "items": rows, "item_count": len(rows)}
        if tool == "record_upload_tracker":
            rx = str(args.get("receiver_id") or "R000")
            profile = str(args.get("issue_profile") or "auto")
            req_id = str(args.get("request_id") or "req-1")
            tracker_id = f"upload_{profile}_{rx}_{req_id}"
            prior = dict(self.trackers.get(tracker_id, {}))
            row = {
                **prior, **dict(args), "ok": True, "result": "UPLOAD_TRACKER_RECORDED",
                "tracker_id": tracker_id,
                "receipt_status": prior.get("receipt_status", "upload_requested_pending_receipt"),
                "workflow_status": args.get("workflow_status") or prior.get("workflow_status") or "WAITING_FOR_LOGS",
                "landed_log_types": prior.get("landed_log_types", []),
                "missing_log_types": prior.get("missing_log_types", []),
                "history": prior.get("history", []), "write_performed": not bool(prior),
            }
            self.trackers[tracker_id] = row
            return row
        if tool == "update_upload_tracker_from_s3":
            tracker_id = str(args.get("tracker_id") or "")
            existing = self.trackers.get(tracker_id, {})
            rx = str(existing.get("receiver_id") or args.get("receiver_id") or "R000")
            # In receipt_check_error_mode, fast-fail so phase 5 falls back to cohort.
            if self.receipt_check_error_mode:
                return {"ok": False, "result": "UPLOAD_TRACKER_ERROR",
                        "error_code": "receipt_check_error",
                        "tracker_id": tracker_id or f"tr-{rx}",
                        "receipt_status": "receipt_check_error",
                        "workflow_status": "ERROR", "dry_run": False, "write_performed": False}
            # The second reconciliation transitions the durable tracker to ready.
            poll_count = self.counts.get("update_upload_tracker_from_s3", 0)
            if poll_count >= 2:
                row = {**existing, "ok": True, "result": "UPLOAD_TRACKER_UPDATED",
                       "tracker_id": tracker_id or f"tr-{rx}", "receipt_status": "logs_landed",
                       "workflow_status": "READY_FOR_ANALYSIS",
                       "landed_log_types": ["procmgr", "android_main"], "missing_log_types": []}
            else:
                row = {**existing, "ok": True, "result": "UPLOAD_TRACKER_UPDATED",
                       "tracker_id": tracker_id or f"tr-{rx}", "receipt_status": "upload_requested_pending_receipt",
                       "workflow_status": "WAITING_FOR_LOGS",
                       "landed_log_types": [], "missing_log_types": ["procmgr"]}
            if args.get("dry_run"):
                return {**row, "result": f"{row.get('result', 'UPLOAD_TRACKER_UPDATED')}_PREVIEW", "dry_run": True, "write_performed": False}
            if row.get("tracker_id"):
                self.trackers[str(row["tracker_id"])] = row
            return {**row, "dry_run": False, "write_performed": True}
        if tool == "get_learning_system_status":
            return {"ok": True, "status": "HEALTHY"}
        if tool in {"human_review_fix_lineage", "human_review_fix_lineage_refresh"}:
            key = args.get("logical_case_key", "")
            if "case-existing" in key:
                pass
            if key == "GLOBAL":
                return {"status": "FIX_LINEAGE_PARTIAL", "jira_issue_ids": [], "jira_fix_ids": [], "pr_ids": [], "commit_sha": ""}
            if "R100|2026-07-20" in key:
                return {"status": "FIX_LINEAGE_PARTIAL", "jira_issue_ids": ["ATVDI-1"], "jira_fix_ids": [], "pr_ids": [], "commit_sha": "", "write_performed": tool.endswith("refresh")}
            return {"status": "FIX_LINEAGE_PARTIAL", "jira_issue_ids": [], "jira_fix_ids": [], "pr_ids": [], "commit_sha": "", "write_performed": tool.endswith("refresh")}
        if tool == "record_case_outcome":
            return {"ok": True, "write_performed": True}
        if tool == "count_alerts":
            names = args.get("alert_name") or args.get("alert_names", "")
            buckets = []
            if "PMGR_UNEXPECTED_EXIT" in names:
                buckets.append({"key": "PMGR_UNEXPECTED_EXIT", "count": 12000})
            if "1031" in names:
                buckets.append({"key": "1031", "count": 5000})
            return {"buckets": buckets}
        if tool == "list_anomalies":
            return {"anomalies": [
                {"alert_name": "PMGR_UNEXPECTED_EXIT", "score": 98, "count": 12000, "receiver_ids": ["R100"]},
                {"alert_name": "MYSTERY_RENDER_LOOP", "score": 99, "count": 250, "receiver_ids": ["R200"]},
            ]}
        if tool == "plan_profile_investigation":
            profile = args.get("profile", "")
            receiver = "R200" if profile.startswith("auto_") else ("R1031" if profile == "guide_1031" else "R100")
            return {"candidates": [{"receiver_id": receiver, "date": "2026-07-21"}]}
        if tool == "verify_receiver_log_coverage":
            receiver = str(args.get("receiver_ids", "")).split(",")[0]
            return {"coverage": [{"receiver_id": receiver, "status": "complete", "required_logs_present": True, "available_log_types": ["stbc_main"]}]}
        if tool == "grasshopper_plan_profile_upload":
            # Real nested Grasshopper contract: executor → {"status": "success", "plan": {...}}
            profile = args.get("profile", "atv_core")
            receiver = args.get("receiver_id", "R000")
            return {
                "status": "success",
                "profile": profile,
                "receiver_id": receiver,
                "plan": {
                    "selected_file_count": 10,
                    "selected_file_ids": list(range(1, 11)),
                    "expanded_log_types": ["procmgr", "sg_server", "qt_gui", "reactuijava"],
                    "missing_profile_log_types": [],
                    "deferred_profile_log_types_by_upload_mode": [],
                    "files_available_by_type": {"procmgr": 3, "sg_server": 3, "qt_gui": 2, "reactuijava": 2},
                },
            }
        if tool == "grasshopper_upload_profile_logs":
            # Real nested Grasshopper contract: executor → {"status": "success", "response": {"status": "success", "request_id": "..."}}
            dry = args.get("dry_run", False)
            return {
                "status": "success",
                "uploaded_file_ids": list(range(1, 11)),
                "destination": "s3://grasshopper-uploads/test/",
                "response": {
                    "status": "success",
                    "request_id": "gh-req-abc123" if not dry else "",
                },
            }
        if tool == "verify_profile_upload_receipt":
            return {"ok": True, "status": "OK", "receipt_status": "COMPLETE"}
        if tool == "register_issue_profile":
            return {"ok": True, "write_performed": True, "profile_id": args.get("issue_profile")}
        if tool == "record_investigation_case":
            return {"ok": True, "write_performed": True, "case_id": f"case-{self.counts[tool]}"}
        if tool == "build_human_evidence_bundle":
            cid = str(args.get("case_id") or "")
            queue_id = str(args.get("queue_id") or "")
            assert queue_id in self.queues, "bundle attempted before its authoritative queue existed"
            bundle_hash = f"bh-{cid}"
            if args.get("dry_run"):
                return {"ok": True, "result": "HUMAN_EVIDENCE_BUNDLE_PREVIEW",
                        "bundle_hash": bundle_hash, "expected_bundle_hash": bundle_hash,
                        "blocking_codes": [], "write_performed": False}
            assert args.get("confirm_build") == "BUILD_HUMAN_EVIDENCE_BUNDLE"
            assert args.get("expected_bundle_hash") == bundle_hash
            key = (queue_id, cid)
            if key in self.bundles:
                return {**self.bundles[key], "ok": True, "result": "HUMAN_EVIDENCE_BUNDLE_ALREADY_EXISTS", "write_performed": False}
            row = {"bundle_id": f"bundle-{cid}", "bundle_hash": bundle_hash}
            self.bundles[key] = row
            return {**row, "ok": True, "result": "HUMAN_EVIDENCE_BUNDLE_BUILT", "write_performed": True}
        if tool == "create_human_review_queue":
            queue_id = str(args.get("queue_id") or "")
            raw_case_ids = args.get("case_ids") or ""
            case_ids = [str(x) for x in (raw_case_ids if isinstance(raw_case_ids, list) else str(raw_case_ids).split(",")) if str(x)]
            queue_hash = "queue-hash-" + "-".join(case_ids)
            items = [{"queue_item_id": f"{queue_id}:{cid}", "case_id": cid} for cid in case_ids]
            if args.get("dry_run"):
                return {"ok": True, "result": "HUMAN_REVIEW_QUEUE_PREVIEW",
                        "queue_hash": queue_hash, "item_count": len(items), "items": items,
                        "write_performed": False}
            assert args.get("confirm_create") == "CREATE_HUMAN_REVIEW_QUEUE"
            assert args.get("expected_queue_hash") == queue_hash
            prior = self.queues.get(queue_id)
            if prior is not None:
                assert prior.get("queue_hash") == queue_hash, "immutable queue id reused with changed content"
                return {"ok": True, "result": "HUMAN_REVIEW_QUEUE_ALREADY_EXISTS",
                        "queue_id": queue_id, "queue_hash": queue_hash,
                        "item_count": len(prior.get("items", [])), "write_performed": False}
            self.queues[queue_id] = {"queue_hash": queue_hash, "items": items}
            return {"ok": True, "result": "HUMAN_REVIEW_QUEUE_CREATED",
                    "queue_id": queue_id, "queue_hash": queue_hash,
                    "item_count": len(items), "write_performed": True}
        if tool == "export_human_adjudication_packet":
            cid = str(args.get("case_id") or "")
            queue_id = str(args.get("queue_id") or "")
            assert (queue_id, cid) in self.bundles, "packet attempted before its evidence bundle existed"
            packet_hash = f"ph-{cid}"
            if args.get("dry_run"):
                return {"ok": True, "result": "HUMAN_ADJUDICATION_PACKET_PREVIEW",
                        "packet_hash": packet_hash, "expected_packet_hash": packet_hash,
                        "blocking_codes": [], "write_performed": False}
            assert args.get("confirm_export") == "EXPORT_HUMAN_ADJUDICATION_PACKET"
            assert args.get("expected_packet_hash") == packet_hash
            key = (queue_id, cid)
            if key in self.packets:
                return {**self.packets[key], "ok": True, "result": "HUMAN_ADJUDICATION_PACKET_ALREADY_EXISTS", "write_performed": False}
            row = {"packet_id": f"packet-{cid}", "packet_hash": packet_hash}
            self.packets[key] = row
            return {**row, "ok": True, "result": "HUMAN_ADJUDICATION_PACKET_EXPORTED", "write_performed": True}
        if tool == "audit_human_adjudication_packet":
            return {"ok": True, "integrity_status": "PASS"}
        if tool == "human_review_materialize_engineer_contexts":
            return {"ok": True, "write_performed": bool(args.get("persist")), "status": "COMPLETE"}
        if tool == "validate_human_adjudication_packet_readiness":
            return {"ok": True, "reviewer_ready": True}
        if tool == "audit_human_review_queue_integrity":
            return {"ok": True, "integrity_ok": True, "status": "PASS"}
        if tool == "audit_case_adjudication_staleness":
            return {"ok": True, "current": True, "status": "CURRENT"}
        raise AssertionError(f"No fake response for {tool} {args}")
