"""Run-state model and atomic checkpoint storage."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()




def normalize_requested_log_types(value: Any, *, max_depth: int = 2, max_items: int = 64) -> list[str]:
    """Safely normalize native, CSV, JSON-list, and one nested legacy list.

    The function is bounded and never mutates the stored raw representation.
    Malformed values are treated as one literal token rather than recursively
    deserialized without limit.
    """

    def _once(item: Any, depth: int) -> list[str]:
        if item is None:
            return []
        if isinstance(item, (list, tuple, set)):
            if depth > max_depth + 1:
                return []
            out: list[str] = []
            children = sorted(item, key=str) if isinstance(item, set) else list(item)
            for child in children[:max_items]:
                out.extend(_once(child, depth + 1))
                if len(out) >= max_items:
                    break
            return out[:max_items]
        text = str(item).strip()
        if not text:
            return []
        if depth < max_depth and text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, list):
                return _once(parsed, depth + 1)
        if "," in text and not (text.startswith("{") or text.startswith("[")):
            return [part.strip() for part in text.split(",") if part.strip()][:max_items]
        return [text[:128]]

    result: list[str] = []
    for token in _once(value, 0):
        clean = str(token).strip()[:128]
        if clean and clean not in result:
            result.append(clean)
        if len(result) >= max_items:
            break
    return result

def normalize_pending_identifier_provenance(
    row: dict[str, Any],
    *,
    current_run_id: str,
) -> dict[str, Any]:
    """Normalize tracker/local/external identifiers without guessing provenance.

    Legacy ``request_id`` values are retained for compatibility but are never
    silently promoted to an external Grasshopper request ID.
    """

    out = dict(row)
    tracker_id = str(out.get("tracker_id") or "")
    explicit_local = str(out.get("local_correlation_id") or "")
    external = str(out.get("grasshopper_request_id") or "")
    legacy_request = str(out.get("request_id") or "")
    declared = str(out.get("identifier_provenance") or "")

    local = explicit_local
    legacy_untyped = str(out.get("legacy_untyped_request_id") or "")
    if not local and legacy_request:
        if tracker_id or external or declared.startswith("TYPED_") or declared == "NIGHTLY_RCA_TYPED_IDENTIFIERS":
            local = legacy_request
        else:
            legacy_untyped = legacy_request

    if tracker_id and local and external:
        provenance = "TYPED_TRACKER_LOCAL_AND_EXTERNAL"
    elif tracker_id and local:
        provenance = "TYPED_TRACKER_AND_LOCAL"
    elif tracker_id and external:
        provenance = "TYPED_TRACKER_AND_EXTERNAL"
    elif tracker_id:
        provenance = "TYPED_TRACKER_ONLY"
    elif local and external:
        provenance = "TYPED_LOCAL_AND_EXTERNAL"
    elif local:
        provenance = "TYPED_LOCAL_CORRELATION_ONLY"
    elif external:
        provenance = "TYPED_EXTERNAL_REQUEST_ONLY"
    elif legacy_untyped:
        provenance = "LEGACY_UNTYPED_REQUEST_ID"
    else:
        provenance = declared or "NO_IDENTIFIER"

    origin_run_id = str(out.get("origin_run_id") or "")
    historical = not origin_run_id or origin_run_id != str(current_run_id or "")
    if tracker_id:
        physical_identity = f"tracker:{tracker_id}"
    elif local:
        physical_identity = f"local:{local}"
    elif legacy_untyped:
        physical_identity = f"legacy:{legacy_untyped}"
    elif external:
        physical_identity = f"external:{external}"
    else:
        fallback = "|".join(str(out.get(key) or "") for key in ("profile_id", "issue_profile", "receiver_id", "candidate_id"))
        physical_identity = f"logical-fallback:{fallback}" if fallback.strip("|") else ""

    raw_requested_log_types = out.get("requested_log_types")
    normalized_requested_log_types = normalize_requested_log_types(raw_requested_log_types)
    raw_receipt_status = str(out.get("receipt_status") or out.get("status") or "")
    workflow_status = str(out.get("workflow_status") or "").upper()
    pending_receipt_states = {
        "submitted", "upload_requested_pending_receipt", "waiting_for_logs",
        "not_ready", "pending",
    }
    normalized_pending_state = (
        "UPLOAD_REQUESTED_PENDING_RECEIPT"
        if workflow_status == "WAITING_FOR_LOGS"
        and raw_receipt_status.strip().lower() in pending_receipt_states
        else ""
    )

    out.update({
        "tracker_id": tracker_id,
        "local_correlation_id": local,
        "grasshopper_request_id": external,
        "legacy_untyped_request_id": legacy_untyped,
        "identifier_provenance": provenance,
        "origin_run_id": origin_run_id,
        "historical_carry_forward": historical,
        "currentness": (
            "CURRENT_RUN" if origin_run_id and origin_run_id == str(current_run_id or "")
            else "HISTORICAL_CARRY_FORWARD" if origin_run_id
            else "ORIGIN_RUN_UNKNOWN"
        ),
        "record_scope": "PHYSICAL_TRACKER_ROW",
        "physical_identity": physical_identity,
        "raw_requested_log_types": raw_requested_log_types,
        "normalized_requested_log_types": normalized_requested_log_types,
        "raw_receipt_status": raw_receipt_status,
        "normalized_pending_state": normalized_pending_state,
    })
    if local:
        out["request_id"] = local
    return out




def evaluate_duplicate_preflight(
    rows: list[dict[str, Any]],
    *,
    inventory_available: bool,
    receiver_id: str,
    requested_log_types: Any = (),
    grasshopper_profile: str = "",
) -> dict[str, Any]:
    """Fail-closed durable duplicate preflight for one upload request."""
    if not inventory_available:
        return {
            "status": "INCOMPLETE",
            "reason": "TRACKER_INVENTORY_UNAVAILABLE",
            "relevant_pending_tracker_ids": [],
        }
    requested = set(normalize_requested_log_types(requested_log_types))
    profile = str(grasshopper_profile or "").strip()
    terminal_receipts = {
        "receipt_complete", "logs_landed", "ready_for_analysis", "complete",
        "closed", "failed_terminal", "cancelled",
    }
    terminal_workflows = {"READY_FOR_ANALYSIS", "COMPLETE", "CLOSED", "FAILED_TERMINAL", "CANCELLED"}
    relevant: list[str] = []
    for raw in rows or []:
        if not isinstance(raw, dict) or str(raw.get("receiver_id") or "") != str(receiver_id or ""):
            continue
        row = normalize_pending_identifier_provenance(dict(raw), current_run_id="")
        receipt = str(row.get("receipt_status") or row.get("status") or "").lower()
        workflow = str(row.get("workflow_status") or "").upper()
        if receipt in terminal_receipts or workflow in terminal_workflows:
            continue
        row_types = set(row.get("normalized_requested_log_types") or ())
        row_profile = str(row.get("grasshopper_profile") or row.get("profile_id") or row.get("issue_profile") or "")
        if (requested and row_types.intersection(requested)) or (profile and row_profile == profile):
            relevant.append(str(row.get("tracker_id") or row.get("request_id") or row.get("legacy_untyped_request_id") or ""))
    relevant = sorted({value for value in relevant if value})
    return {
        "status": "BLOCK" if relevant else "PASS",
        "reason": "RELEVANT_PENDING_TRACKER" if relevant else "NO_RELEVANT_PENDING_TRACKER",
        "relevant_pending_tracker_ids": relevant,
    }


@dataclass
class StepRecord:
    phase: str
    step: str
    tool: str | None
    server: str | None
    arguments: dict[str, Any]
    status: str
    started_at: str
    completed_at: str
    elapsed_ms: int
    response: Any = None
    error: str | None = None
    attempt: int = 1


@dataclass
class RunState:
    schema_version: int
    run_id: str
    mode: str
    role: str
    started_at: str
    completed_at: str | None = None
    status: str = "RUNNING"
    current_phase: str | None = None
    completed_phases: list[str] = field(default_factory=list)
    active_date_window: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, int] = field(default_factory=lambda: {
        "tool_calls": 0,
        "tool_not_found": 0,
        "step_failed": 0,
        "role_errors": 0,
        "persist_failures": 0,
        "write_blocked": 0,
        "expected_negative_canary_pass": 0,
    })

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunState":
        return cls(**raw)


class RunStore:
    def __init__(self, output_dir: Path, run_id: str):
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.run_dir = self.output_dir / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "state.json"
        self.events_path = self.run_dir / "events.jsonl"
        # Cross-run durable work ledger. This intentionally lives outside a
        # specific run directory so a later cron invocation can resume it.
        self.pending_path = self.output_dir / "pending_investigations.json"
        self.pending_load_status = "NOT_EVALUATED"
        self.pending_load_error_class = ""

    @staticmethod
    def _atomic_json(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(value, fh, indent=2, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def save(self, state: RunState) -> None:
        payload = state.to_dict()
        payload["state_hash"] = stable_hash(payload)
        self._atomic_json(self.state_path, payload)
        latest = {
            "run_id": state.run_id,
            "state_path": str(self.state_path),
            "status": state.status,
            "updated_at": utc_now(),
        }
        self._atomic_json(self.output_dir / "latest.json", latest)

    def save_phase(self, phase: str, payload: Any) -> None:
        safe = phase.replace(".", "_").replace("/", "_")
        self._atomic_json(self.run_dir / f"phase_{safe}.json", payload)

    def append_event(self, event: dict[str, Any]) -> None:
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=True, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def write_text(self, name: str, text: str) -> Path:
        path = self.run_dir / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return path


    def load_pending_investigations(self) -> list[dict[str, Any]]:
        """Load durable unfinished work and expose bounded availability state.

        A missing ledger is a valid empty local source. Corrupt/unreadable
        material is an availability failure and must block duplicate-sensitive
        submissions rather than masquerading as an empty inventory.
        """
        self.pending_load_error_class = ""
        if not self.pending_path.exists():
            self.pending_load_status = "NOT_PRESENT"
            return []
        try:
            with open(self.pending_path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            self.pending_load_status = "ERROR"
            self.pending_load_error_class = type(exc).__name__
            return []
        rows = raw.get("items", []) if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            self.pending_load_status = "ERROR"
            self.pending_load_error_class = "INVALID_LEDGER_SCHEMA"
            return []
        self.pending_load_status = "AVAILABLE"
        return [
            normalize_pending_identifier_provenance(dict(row), current_run_id=self.run_id)
            for row in rows
            if isinstance(row, dict)
        ]

    @property
    def pending_inventory_available(self) -> bool:
        return self.pending_load_status in {"NOT_PRESENT", "AVAILABLE"}

    def save_pending_investigations(self, items: list[dict[str, Any]]) -> None:
        """Atomically replace the durable pending-work ledger with deduplicated rows."""
        dedup: dict[str, dict[str, Any]] = {}
        now = utc_now()
        for row in items:
            if not isinstance(row, dict):
                continue
            row_copy = normalize_pending_identifier_provenance(
                dict(row), current_run_id=self.run_id
            )
            key = str(row_copy.get("physical_identity") or "")
            if key:
                row_copy.setdefault("created_at", now)
                dedup[key] = row_copy
        ordered = sorted(
            dedup.values(),
            key=lambda row: (
                str(row.get("next_check_at") or ""),
                str(row.get("profile_id") or row.get("issue_profile") or ""),
                str(row.get("receiver_id") or ""),
            ),
        )
        self._atomic_json(
            self.pending_path,
            {
                "schema_version": "nightly_rca_pending.v1",
                "updated_at": utc_now(),
                "item_count": len(ordered),
                "count_scope": {
                    "collection": "pending_investigations",
                    "record_scope": "PHYSICAL_TRACKER_ROW",
                    "deduplication_key": "typed_physical_identity",
                },
                "items": ordered,
            },
        )

    @classmethod
    def load(cls, state_path: Path) -> tuple["RunStore", RunState]:
        with open(state_path, encoding="utf-8") as fh:
            raw = json.load(fh)
        observed_hash = raw.pop("state_hash", None)
        if observed_hash is not None and observed_hash != stable_hash(raw):
            raise ValueError(f"state integrity check failed: {state_path}")
        state = RunState.from_dict(raw)
        store = cls(state_path.parents[2], state.run_id)
        return store, state
