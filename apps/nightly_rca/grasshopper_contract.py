"""
Typed adapter for Grasshopper upload plan and upload response contracts.

Normalizes the nested Grasshopper response structure into well-typed,
immutable records for reliable consumption by the nightly RCA pipeline.

The Jake executor returns:
    {"status": "OK", "response": <raw grasshopper response>}

The raw Grasshopper response uses its own contract:
    {"status": "success", "profile": "...", "plan": {...}, ...}

This module validates both layers independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ── Acquisition States ────────────────────────────────────────────────────────

class AcquisitionState(str, Enum):
    UPLOAD_PLAN_FAILED = "UPLOAD_PLAN_FAILED"
    UPLOAD_BLOCKED_UNKNOWN_BATCH_SIZE = "UPLOAD_BLOCKED_UNKNOWN_BATCH_SIZE"
    UPLOAD_BLOCKED_BATCH_LIMIT = "UPLOAD_BLOCKED_BATCH_LIMIT"
    UPLOAD_BLOCKED_DUPLICATE_PREFLIGHT_UNAVAILABLE = "UPLOAD_BLOCKED_DUPLICATE_PREFLIGHT_UNAVAILABLE"
    UPLOAD_BLOCKED_DUPLICATE_REQUEST = "UPLOAD_BLOCKED_DUPLICATE_REQUEST"
    UPLOAD_SKIPPED_NO_FILES = "UPLOAD_SKIPPED_NO_FILES"
    UPLOAD_PLAN_ONLY = "UPLOAD_PLAN_ONLY"
    UPLOAD_REJECTED = "UPLOAD_REJECTED"
    UPLOAD_PROTOCOL_ERROR = "UPLOAD_PROTOCOL_ERROR"
    UPLOAD_ACCEPTANCE_UNKNOWN = "UPLOAD_ACCEPTANCE_UNKNOWN"
    UPLOAD_ACCEPTED = "UPLOAD_ACCEPTED"
    UPLOAD_ACCEPTED_TRACKER_WRITE_FAILED = "UPLOAD_ACCEPTED_TRACKER_WRITE_FAILED"
    UPLOAD_SUBMITTED_TRACKED = "UPLOAD_SUBMITTED_TRACKED"


ACQUISITION_STATE_META: dict[AcquisitionState, dict[str, str]] = {
    AcquisitionState.UPLOAD_PLAN_FAILED: {
        "classification": "error",
        "operator_message": "Grasshopper upload planning failed. No upload was submitted.",
        "metric": "nightly_rca.grasshopper.plan_failed",
        "retry_policy": "retry_next_run",
    },
    AcquisitionState.UPLOAD_BLOCKED_UNKNOWN_BATCH_SIZE: {
        "classification": "blocked",
        "operator_message": "Grasshopper returned a plan whose selected-file count could not be read. No upload was submitted.",
        "metric": "nightly_rca.grasshopper.blocked_unknown_size",
        "retry_policy": "retry_next_run",
    },
    AcquisitionState.UPLOAD_BLOCKED_BATCH_LIMIT: {
        "classification": "blocked",
        "operator_message": "Grasshopper plan exceeds the safety cap. No upload was submitted.",
        "metric": "nightly_rca.grasshopper.blocked_batch_limit",
        "retry_policy": "manual_review",
    },
    AcquisitionState.UPLOAD_BLOCKED_DUPLICATE_PREFLIGHT_UNAVAILABLE: {
        "classification": "blocked",
        "operator_message": "Upload tracker inventory is unavailable, so duplicate safety cannot be proven. No upload was submitted.",
        "metric": "nightly_rca.grasshopper.duplicate_preflight_unavailable",
        "retry_policy": "retry_after_inventory_recovery",
    },
    AcquisitionState.UPLOAD_BLOCKED_DUPLICATE_REQUEST: {
        "classification": "blocked",
        "operator_message": "An equivalent pending upload tracker already exists. No duplicate upload was submitted.",
        "metric": "nightly_rca.grasshopper.duplicate_blocked",
        "retry_policy": "reconcile_existing_request",
    },
    AcquisitionState.UPLOAD_SKIPPED_NO_FILES: {
        "classification": "no_action",
        "operator_message": "Grasshopper selected zero matching files for the resolved profile. No upload was submitted.",
        "metric": "nightly_rca.grasshopper.skipped_no_files",
        "retry_policy": "retry_next_run",
    },
    AcquisitionState.UPLOAD_PLAN_ONLY: {
        "classification": "dry_run",
        "operator_message": "Upload planning completed in dry-run mode. No upload was submitted.",
        "metric": "nightly_rca.grasshopper.plan_only",
        "retry_policy": "none",
    },
    AcquisitionState.UPLOAD_REJECTED: {
        "classification": "rejected",
        "operator_message": "Grasshopper explicitly rejected the submitted upload request.",
        "metric": "nightly_rca.grasshopper.rejected",
        "retry_policy": "manual_review",
    },
    AcquisitionState.UPLOAD_PROTOCOL_ERROR: {
        "classification": "protocol_error",
        "operator_message": "Grasshopper rejected the request wire contract. No request ID was created; automatic retry is disabled.",
        "metric": "nightly_rca.grasshopper.protocol_error",
        "retry_policy": "manual_review",
    },
    AcquisitionState.UPLOAD_ACCEPTANCE_UNKNOWN: {
        "classification": "uncertain",
        "operator_message": "A submission may have occurred, but acceptance could not be confirmed. Do not retry until reconciled.",
        "metric": "nightly_rca.grasshopper.acceptance_unknown",
        "retry_policy": "no_retry_until_reconciled",
    },
    AcquisitionState.UPLOAD_ACCEPTED: {
        "classification": "success",
        "operator_message": "Grasshopper accepted the upload request.",
        "metric": "nightly_rca.grasshopper.accepted",
        "retry_policy": "none",
    },
    AcquisitionState.UPLOAD_ACCEPTED_TRACKER_WRITE_FAILED: {
        "classification": "partial_success",
        "operator_message": "Upload was accepted but the tracker write failed. Manual reconciliation required.",
        "metric": "nightly_rca.grasshopper.accepted_tracker_failed",
        "retry_policy": "manual_reconciliation",
    },
    AcquisitionState.UPLOAD_SUBMITTED_TRACKED: {
        "classification": "success",
        "operator_message": "Upload submitted and durably tracked.",
        "metric": "nightly_rca.grasshopper.submitted_tracked",
        "retry_policy": "none",
    },
}


# ── Identifier Provenance ─────────────────────────────────────────────────────

UNKNOWN_PROVENANCE = "UNKNOWN_PROVENANCE"


@dataclass(frozen=True)
class IdentifierSet:
    """Distinct identifiers with clear provenance."""
    local_correlation_id: str = ""
    grasshopper_request_id: str = ""
    tracker_id: str = ""
    origin_run_id: str = ""
    identifier_provenance: str = UNKNOWN_PROVENANCE


# ── Profile Metadata ──────────────────────────────────────────────────────────

NIGHTLY_MAX_UPLOAD_FILES = 50


@dataclass(frozen=True)
class ProfileMetadata:
    """Propagated profile metadata from the S3 issue profile registry."""
    issue_profile: str = ""
    grasshopper_profile: str = ""
    upload_mode: str = "balanced"
    max_files_per_type: int = 15
    max_total_files: int = 120
    core_log_types: tuple[str, ...] = ()
    supplemental_log_types: tuple[str, ...] = ()

    @property
    def effective_max_total_files(self) -> int:
        """Nightly hard cap clamps profile max_total_files to 50."""
        return min(self.max_total_files, NIGHTLY_MAX_UPLOAD_FILES)

    @property
    def effective_max_files_per_type(self) -> int:
        """Bounded max_files_per_type for safety."""
        return min(self.max_files_per_type, NIGHTLY_MAX_UPLOAD_FILES)

    @property
    def requested_log_types(self) -> tuple[str, ...]:
        """Stable de-duplicated log-family request in catalog order."""
        return tuple(dict.fromkeys((*self.core_log_types, *self.supplemental_log_types)))

    def to_public_dict(self) -> dict[str, Any]:
        """Serializable metadata safe for checkpoints and operator reports."""
        return {
            "issue_profile": self.issue_profile,
            "grasshopper_profile": self.grasshopper_profile,
            "upload_mode": self.upload_mode,
            "max_files_per_type": self.max_files_per_type,
            "max_total_files": self.max_total_files,
            "effective_max_files_per_type": self.effective_max_files_per_type,
            "effective_max_total_files": self.effective_max_total_files,
            "core_log_types": list(self.core_log_types),
            "supplemental_log_types": list(self.supplemental_log_types),
            "requested_log_types": list(self.requested_log_types),
        }


# ── Plan Result ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GrasshopperPlanResult:
    """Normalized result from a Grasshopper plan-profile-upload call.

    The executor returns {"status": "OK", "response": <raw>}.
    The raw Grasshopper response has:
        {"status": "success", "profile": "...", "plan": {...}}
    """
    # Executor layer
    executor_ok: bool = False
    executor_error: str = ""

    # Inner Grasshopper layer
    inner_status: str = ""
    profile: str = ""
    requested_profile: str = ""
    resolved_profile: str = ""
    profile_alias_used: Optional[bool] = None
    profile_identity_matches: Optional[bool] = None
    receiver_id: str = ""

    # Nested plan fields
    selected_file_count: Optional[int] = None
    selected_file_ids: tuple[int, ...] = ()
    expanded_log_types: tuple[str, ...] = ()
    missing_profile_log_types: tuple[str, ...] = ()
    deferred_profile_log_types_by_upload_mode: tuple[str, ...] = ()
    files_available_by_type: dict[str, int] = field(default_factory=dict)

    # Profile metadata (propagated)
    profile_metadata: Optional[ProfileMetadata] = None

    # Warnings/caps
    warnings: tuple[str, ...] = ()
    raw: Any = field(default=None, repr=False)

    @property
    def is_executor_success(self) -> bool:
        return self.executor_ok

    @property
    def is_inner_success(self) -> bool:
        return self.inner_status == "success"

    @property
    def has_valid_file_count(self) -> bool:
        return self.selected_file_count is not None

    @property
    def file_count_is_zero(self) -> bool:
        return self.selected_file_count == 0

    @property
    def file_count_is_positive(self) -> bool:
        return self.selected_file_count is not None and self.selected_file_count > 0

    def to_public_dict(self) -> dict[str, Any]:
        """Bounded non-secret plan evidence for state/report persistence."""
        return {
            "executor_ok": self.executor_ok,
            "inner_status": self.inner_status,
            "requested_profile": self.requested_profile,
            "resolved_profile": self.resolved_profile,
            "profile_alias_used": self.profile_alias_used,
            "profile_identity_matches": self.profile_identity_matches,
            "receiver_id": self.receiver_id,
            "selected_file_count": self.selected_file_count,
            "selected_file_ids": list(self.selected_file_ids[:200]),
            "expanded_log_types": list(self.expanded_log_types[:64]),
            "missing_profile_log_types": list(self.missing_profile_log_types[:64]),
            "deferred_profile_log_types_by_upload_mode": list(
                self.deferred_profile_log_types_by_upload_mode[:64]
            ),
            "files_available_by_type": dict(
                list(sorted(self.files_available_by_type.items()))[:64]
            ),
            "warnings": list(self.warnings[:32]),
            "executor_error_class": "PLAN_EXECUTOR_ERROR" if self.executor_error else "",
        }


# ── Upload Result ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GrasshopperUploadResult:
    """Normalized result from a Grasshopper upload-profile-logs call."""
    # Executor layer
    executor_ok: bool = False
    executor_error: str = ""

    # Outer response fields
    outer_status: str = ""
    uploaded_file_ids: tuple[int, ...] = ()
    destination: str = ""

    # Nested response (inner Grasshopper response)
    inner_status: str = ""
    inner_request_id: str = ""

    # Derived state
    write_state: str = ""
    error: str = ""
    http_status: Optional[int] = None
    upstream_code: str = ""
    request_created: Optional[bool] = None
    automatic_retry: bool = False
    retry_policy: str = "manual_review"
    requested_profile: str = ""
    resolved_profile: str = ""
    profile_alias_used: Optional[bool] = None
    raw: Any = field(default=None, repr=False)

    @property
    def is_executor_success(self) -> bool:
        return self.executor_ok

    @property
    def is_inner_success(self) -> bool:
        return self.inner_status == "success"

    @property
    def has_proven_request_id(self) -> bool:
        return bool(self.inner_request_id)

    def to_public_dict(self) -> dict[str, Any]:
        """Bounded non-secret upload evidence; never persist raw response bodies."""
        return {
            "executor_ok": self.executor_ok,
            "outer_status": self.outer_status,
            "inner_status": self.inner_status,
            "grasshopper_request_id": self.inner_request_id,
            "uploaded_file_ids": list(self.uploaded_file_ids[:200]),
            "destination": self.destination[:256],
            "write_state": self.write_state,
            "http_status": self.http_status,
            "upstream_code": self.upstream_code,
            "request_created": self.request_created,
            "automatic_retry": self.automatic_retry,
            "retry_policy": self.retry_policy,
            "requested_profile": self.requested_profile,
            "resolved_profile": self.resolved_profile,
            "profile_alias_used": self.profile_alias_used,
            "error_class": (
                "UPLOAD_PROTOCOL_ERROR" if self.write_state == "protocol_error"
                else "UPLOAD_EXECUTOR_ERROR" if self.write_state == "executor_failed"
                else "UPLOAD_REJECTED" if self.write_state in {"inner_rejected", "outer_failed"}
                else ""
            ),
        }


# ── Parsing Functions ─────────────────────────────────────────────────────────

def parse_plan_response(executor_result: dict[str, Any],
                        profile_metadata: Optional[ProfileMetadata] = None) -> GrasshopperPlanResult:
    """Parse an executor result wrapping a Grasshopper plan response.

    executor_result is {"status": "OK"|"ERROR", "response": <raw>, "error": "..."}
    """
    if not isinstance(executor_result, dict):
        return GrasshopperPlanResult(executor_ok=False, executor_error="non-dict executor result")

    executor_ok = executor_result.get("status") == "OK"
    executor_error = str(executor_result.get("error") or "") if not executor_ok else ""

    raw_response = executor_result.get("response")
    if not isinstance(raw_response, dict):
        return GrasshopperPlanResult(
            executor_ok=executor_ok,
            executor_error=executor_error or ("no response payload" if executor_ok else ""),
            raw=raw_response,
            profile_metadata=profile_metadata,
        )

    inner_status = str(raw_response.get("status") or "")
    profile = str(raw_response.get("profile") or "")
    requested_profile = str(raw_response.get("requested_profile") or profile or "")
    resolved_profile = str(raw_response.get("resolved_profile") or profile or "")
    alias_value = raw_response.get("profile_alias_used")
    profile_alias_used = alias_value if isinstance(alias_value, bool) else None
    expected_profile = str(
        profile_metadata.grasshopper_profile if profile_metadata is not None else ""
    )
    profile_identity_matches = (
        resolved_profile == expected_profile
        if expected_profile and resolved_profile else None
    )
    receiver_id = str(raw_response.get("receiver_id") or "")

    # Parse nested plan object
    plan = raw_response.get("plan")
    selected_file_count: Optional[int] = None
    selected_file_ids: tuple[int, ...] = ()
    expanded_log_types: tuple[str, ...] = ()
    missing_profile_log_types: tuple[str, ...] = ()
    deferred_log_types: tuple[str, ...] = ()
    files_available_by_type: dict[str, int] = {}
    warnings: list[str] = []

    if isinstance(plan, dict):
        # selected_file_count
        raw_count = plan.get("selected_file_count")
        if isinstance(raw_count, int):
            selected_file_count = raw_count

        # selected_file_ids
        raw_ids = plan.get("selected_file_ids")
        if isinstance(raw_ids, list):
            selected_file_ids = tuple(int(x) for x in raw_ids if isinstance(x, (int, float)))

        # expanded_log_types
        raw_elt = plan.get("expanded_log_types")
        if isinstance(raw_elt, list):
            expanded_log_types = tuple(str(x) for x in raw_elt if isinstance(x, str))

        # missing_profile_log_types
        raw_mlt = plan.get("missing_profile_log_types")
        if isinstance(raw_mlt, list):
            missing_profile_log_types = tuple(str(x) for x in raw_mlt if isinstance(x, str))

        # deferred_profile_log_types_by_upload_mode
        raw_dlt = plan.get("deferred_profile_log_types_by_upload_mode")
        if isinstance(raw_dlt, list):
            deferred_log_types = tuple(str(x) for x in raw_dlt if isinstance(x, str))

        # files_available_by_type
        raw_fabt = plan.get("files_available_by_type")
        if isinstance(raw_fabt, dict):
            files_available_by_type = {str(k): int(v) for k, v in raw_fabt.items() if isinstance(v, (int, float))}

        # warnings
        raw_w = plan.get("warnings")
        if isinstance(raw_w, list):
            warnings = [str(x) for x in raw_w if isinstance(x, str)]
    else:
        # Fallback: try top-level keys for backward compatibility
        for key in ("file_count", "files_planned", "planned_file_count", "count", "selected_file_count"):
            raw = raw_response.get(key)
            if isinstance(raw, int):
                selected_file_count = raw
                break
        files = raw_response.get("files")
        if isinstance(files, list):
            if selected_file_count is None:
                selected_file_count = len(files)
            selected_file_ids = tuple(int(x) for x in files if isinstance(x, (int, float)))

    return GrasshopperPlanResult(
        executor_ok=executor_ok,
        executor_error=executor_error,
        inner_status=inner_status,
        profile=profile,
        requested_profile=requested_profile,
        resolved_profile=resolved_profile,
        profile_alias_used=profile_alias_used,
        profile_identity_matches=profile_identity_matches,
        receiver_id=receiver_id,
        selected_file_count=selected_file_count,
        selected_file_ids=selected_file_ids,
        expanded_log_types=expanded_log_types,
        missing_profile_log_types=missing_profile_log_types,
        deferred_profile_log_types_by_upload_mode=deferred_log_types,
        files_available_by_type=files_available_by_type,
        profile_metadata=profile_metadata,
        warnings=tuple(warnings),
        raw=raw_response,
    )


def parse_upload_response(executor_result: dict[str, Any]) -> GrasshopperUploadResult:
    """Parse an executor result wrapping a Grasshopper upload response.

    Protocol failures may be surfaced at the executor, outer MCP, or nested
    Grasshopper layer. Only bounded status/code/message fields are retained.
    """
    if not isinstance(executor_result, dict):
        return GrasshopperUploadResult(executor_ok=False, executor_error="non-dict executor result")

    executor_ok = executor_result.get("status") == "OK"
    executor_error = str(executor_result.get("error") or "")[:500] if not executor_ok else ""

    raw_response = executor_result.get("response")
    outer = raw_response if isinstance(raw_response, dict) else {}
    inner_response = outer.get("response") if isinstance(outer.get("response"), dict) else {}

    outer_status = str(outer.get("status") or "")
    destination = str(outer.get("destination") or "")
    requested_profile = str(outer.get("requested_profile") or outer.get("profile") or "")
    resolved_profile = str(outer.get("resolved_profile") or outer.get("profile") or "")
    alias_value = outer.get("profile_alias_used")
    profile_alias_used = alias_value if isinstance(alias_value, bool) else None

    raw_ids = outer.get("uploaded_file_ids")
    uploaded_file_ids = tuple(
        int(x) for x in raw_ids if isinstance(x, (int, float))
    ) if isinstance(raw_ids, list) else ()

    inner_status = str(inner_response.get("status") or "")
    inner_request_id = str(inner_response.get("request_id") or "")

    def first_value(keys: tuple[str, ...]) -> Any:
        for layer in (inner_response, outer, executor_result):
            for key in keys:
                value = layer.get(key) if isinstance(layer, dict) else None
                if value not in (None, ""):
                    return value
        return None

    error_parts = [
        str(value)[:500]
        for value in (
            first_value(("error",)), first_value(("message",)), executor_error
        )
        if value
    ]
    error = " | ".join(dict.fromkeys(error_parts))[:500]
    raw_http = first_value(("http_status", "status_code"))
    try:
        http_status = int(raw_http) if raw_http is not None else None
    except (TypeError, ValueError):
        http_status = None
    upstream_code = str(first_value(("code", "error_code")) or "")[:64]
    diagnostic = " ".join((error, executor_error)).lower()
    protocol_error = (
        http_status == 406
        or upstream_code == "4002"
        or "json not readable" in diagnostic
        or ("406" in diagnostic and "4002" in diagnostic)
    )

    if protocol_error:
        write_state = "protocol_error"
    elif not executor_ok:
        write_state = "executor_failed"
    elif not isinstance(raw_response, dict):
        write_state = "unknown"
    elif outer_status == "success" and inner_status == "success" and inner_request_id:
        write_state = "accepted_confirmed"
    elif outer_status == "success" and inner_status == "success" and not inner_request_id:
        write_state = "acceptance_unknown"
    elif outer_status == "success" and inner_status != "success":
        write_state = "inner_rejected"
    elif outer_status != "success":
        write_state = "outer_failed"
    else:
        write_state = "unknown"
    request_created = True if inner_request_id else False if protocol_error else None

    return GrasshopperUploadResult(
        executor_ok=executor_ok,
        executor_error=executor_error,
        outer_status=outer_status,
        uploaded_file_ids=uploaded_file_ids,
        destination=destination,
        inner_status=inner_status,
        inner_request_id=inner_request_id,
        write_state=write_state,
        error=error,
        http_status=http_status,
        upstream_code=upstream_code,
        request_created=request_created,
        automatic_retry=False,
        retry_policy="manual_review" if protocol_error else "no_retry_until_reconciled",
        requested_profile=requested_profile,
        resolved_profile=resolved_profile,
        profile_alias_used=profile_alias_used,
        raw=raw_response,
    )


def determine_acquisition_state(
    plan: GrasshopperPlanResult,
    upload: Optional[GrasshopperUploadResult],
    dry_run: bool,
    max_upload_files: int = NIGHTLY_MAX_UPLOAD_FILES,
) -> AcquisitionState:
    """Determine the acquisition state from parsed plan and upload results."""
    if not plan.is_executor_success:
        return AcquisitionState.UPLOAD_PLAN_FAILED

    if not plan.is_inner_success:
        return AcquisitionState.UPLOAD_PLAN_FAILED

    if plan.profile_identity_matches is False:
        return AcquisitionState.UPLOAD_PLAN_FAILED

    if not plan.has_valid_file_count:
        return AcquisitionState.UPLOAD_BLOCKED_UNKNOWN_BATCH_SIZE

    if plan.selected_file_count > max_upload_files:
        return AcquisitionState.UPLOAD_BLOCKED_BATCH_LIMIT

    if plan.file_count_is_zero:
        return AcquisitionState.UPLOAD_SKIPPED_NO_FILES

    if dry_run:
        return AcquisitionState.UPLOAD_PLAN_ONLY

    if upload is None:
        return AcquisitionState.UPLOAD_PLAN_ONLY

    if upload.write_state == "protocol_error":
        return AcquisitionState.UPLOAD_PROTOCOL_ERROR

    if not upload.is_executor_success or upload.write_state == "executor_failed":
        return AcquisitionState.UPLOAD_ACCEPTANCE_UNKNOWN

    if upload.write_state == "inner_rejected":
        return AcquisitionState.UPLOAD_REJECTED

    if upload.write_state == "outer_failed":
        return AcquisitionState.UPLOAD_REJECTED

    if upload.write_state == "accepted_confirmed":
        return AcquisitionState.UPLOAD_ACCEPTED

    if upload.write_state == "acceptance_unknown":
        return AcquisitionState.UPLOAD_ACCEPTANCE_UNKNOWN

    return AcquisitionState.UPLOAD_ACCEPTANCE_UNKNOWN


def build_identifier_set(
    local_correlation_id: str,
    upload_result: Optional[GrasshopperUploadResult] = None,
    tracker_id: str = "",
    origin_run_id: str = "",
) -> IdentifierSet:
    """Build an identifier set with proven provenance."""
    grasshopper_request_id = ""
    provenance = UNKNOWN_PROVENANCE

    if upload_result and upload_result.has_proven_request_id:
        grasshopper_request_id = upload_result.inner_request_id
        provenance = "proven_from_nested_response"
    elif upload_result and upload_result.is_executor_success:
        provenance = "executor_success_no_inner_id"
    elif local_correlation_id:
        provenance = "local_only"

    return IdentifierSet(
        local_correlation_id=local_correlation_id,
        grasshopper_request_id=grasshopper_request_id,
        tracker_id=tracker_id,
        origin_run_id=origin_run_id,
        identifier_provenance=provenance,
    )


def build_profile_metadata(profile_config: dict[str, Any]) -> ProfileMetadata:
    """Build ProfileMetadata from an S3 issue profile config dict."""
    def _types(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            values = [item.strip() for item in value.split(",")]
        elif isinstance(value, (list, tuple, set)):
            values = [str(item).strip() for item in value]
        else:
            values = []
        return tuple(dict.fromkeys(item for item in values if item))

    def _positive_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    issue_profile = str(
        profile_config.get("issue_profile")
        or profile_config.get("profile_id")
        or profile_config.get("name")
        or ""
    )
    grasshopper_profile = str(profile_config.get("grasshopper_profile") or issue_profile)
    return ProfileMetadata(
        issue_profile=issue_profile,
        grasshopper_profile=grasshopper_profile,
        upload_mode=str(profile_config.get("upload_mode") or "balanced"),
        max_files_per_type=_positive_int(profile_config.get("max_files_per_type"), 15),
        max_total_files=_positive_int(profile_config.get("max_total_files"), 120),
        core_log_types=_types(profile_config.get("core_log_types")),
        supplemental_log_types=_types(profile_config.get("supplemental_log_types")),
    )


def profile_metadata_from_catalog(
    issue_profile: str,
    catalog: dict[str, dict[str, Any]] | None,
) -> ProfileMetadata:
    """Resolve one Nightly issue profile without silently discarding catalog metadata."""
    row: dict[str, Any] = {}
    if isinstance(catalog, dict):
        candidate = catalog.get(issue_profile)
        if isinstance(candidate, dict):
            row = dict(candidate)
    row.setdefault("issue_profile", issue_profile)
    row.setdefault("grasshopper_profile", issue_profile)
    return build_profile_metadata(row)


def build_grasshopper_arguments(
    metadata: ProfileMetadata,
    receiver_id: str,
    *,
    max_upload_files: int = NIGHTLY_MAX_UPLOAD_FILES,
    dry_run: bool | None = None,
) -> dict[str, Any]:
    """Build the bounded request contract shared by plan and upload calls."""
    args: dict[str, Any] = {
        "profile": metadata.grasshopper_profile,
        "receiver_id": receiver_id,
        "upload_mode": metadata.upload_mode,
        "max_files_per_type": min(metadata.effective_max_files_per_type, max_upload_files),
        "max_total_files": min(metadata.effective_max_total_files, max_upload_files),
    }
    if dry_run is not None:
        args["dry_run"] = bool(dry_run)
        args["allow_live_upload"] = not bool(dry_run)
    return args
