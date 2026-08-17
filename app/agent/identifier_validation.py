"""Typed identifier validation and provenance for model-emitted tool calls.

The validators in this module are deliberately independent of LangChain and of
any individual executor.  They run at the orchestration boundary, before a
model-emitted call can reach a real tool.

Only identifier *types*, field names, provenance labels, and validation status
are safe for audit output.  Raw identifier values are never included in an
audit record.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

IDENTIFIER_VALIDATION_SCHEMA = "diship_identifier_validation.v1"
IDENTIFIER_VALIDATION_FAILED = "TOOL_IDENTIFIER_VALIDATION_FAILED"

# Confirmed from S3 Incident Scene ``stable_scene_id(..., length=24)``.
_SCENE_ID_RE = re.compile(r"^scene-[0-9a-f]{24}$")
# Confirmed from S3 ``_validate_receiver_id``: optional R + 7-10 digits.
_RECEIVER_ID_RE = re.compile(r"^(?:R)?\d{7,10}$", re.IGNORECASE)
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_SAFE_OPAQUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@|+-]{0,255}$")

_CONTEXT_TYPES = {
    "chat_id",
    "workspace_id",
    "thread_id",
    "task_id",
    "child_run_id",
}
_DOMAIN_TYPES = {
    "scene_id",
    "case_id",
    "receiver_id",
    "request_id",
    "tracker_id",
}
_FIELD_TYPES = {
    "chat_id": "chat_id",
    "workspace_id": "workspace_id",
    "thread_id": "thread_id",
    "task_id": "task_id",
    "child_run_id": "child_run_id",
    "scene_id": "scene_id",
    "case_id": "case_id",
    "receiver_id": "receiver_id",
    "rxid": "receiver_id",
    "rx_id": "receiver_id",
    "request_id": "request_id",
    "tracker_id": "tracker_id",
}


@dataclass(frozen=True)
class IdentifierValidation:
    field_name: str
    identifier_type: str
    validation_status: str
    argument_source: str = "model_argument"
    source_identifier_type: str = ""
    tool_call_id: str = ""
    audit_event_id: str = ""

    def to_audit_dict(self) -> dict[str, Any]:
        # Intentionally excludes the identifier value and all derivatives of it.
        return asdict(self)


@dataclass(frozen=True)
class ToolArgumentValidationResult:
    allowed: bool
    result_code: str = "IDENTIFIER_VALIDATION_PASS"
    validations: tuple[IdentifierValidation, ...] = ()
    audit_records: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _source_type(source: str) -> str:
    text = str(source or "")
    if text.startswith("context."):
        candidate = text.split(".", 1)[1]
        if candidate in _CONTEXT_TYPES:
            return candidate
    return ""


def _is_uuid(value: str) -> bool:
    if not _UUID_RE.fullmatch(value):
        return False
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


def _format_valid(identifier_type: str, value: str) -> bool:
    if identifier_type == "scene_id":
        return bool(_SCENE_ID_RE.fullmatch(value))
    if identifier_type == "receiver_id":
        return bool(_RECEIVER_ID_RE.fullmatch(value))
    if identifier_type in _CONTEXT_TYPES:
        # Context values are opaque.  They are only classified here; they are
        # never permitted to auto-fill a different semantic identifier type.
        return bool(value) and len(value) <= 256 and "\x00" not in value
    if identifier_type == "request_id":
        # Request IDs may legitimately be UUIDs or prefixed server-generated IDs.
        return bool(value) and len(value) <= 256 and (bool(_SAFE_OPAQUE_RE.fullmatch(value)) or _is_uuid(value))
    if identifier_type in {"case_id", "tracker_id"}:
        return bool(_SAFE_OPAQUE_RE.fullmatch(value))
    return True


def _matching_context_type(value: str, context_identifiers: Mapping[str, Any]) -> str:
    for identifier_type in sorted(_CONTEXT_TYPES):
        candidate = context_identifiers.get(identifier_type)
        if candidate is not None and str(candidate) == value:
            return identifier_type
    return ""


def validate_tool_arguments(
    args: Mapping[str, Any] | None,
    *,
    context_identifiers: Mapping[str, Any] | None = None,
    argument_sources: Mapping[str, str] | None = None,
    tool_call_id: str = "",
    audit_event_id: str = "",
) -> ToolArgumentValidationResult:
    """Validate typed identifier fields without recording their values.

    ``context_identifiers`` and ``argument_sources`` are server-owned metadata.
    A model cannot make a context identifier semantically valid merely by
    emitting a UUID-shaped string in a domain argument.
    """

    source_args = args or {}
    contexts = context_identifiers or {}
    sources = argument_sources or {}
    validations: list[IdentifierValidation] = []
    blocked = False

    for field_name, raw_value in source_args.items():
        identifier_type = _FIELD_TYPES.get(str(field_name))
        if not identifier_type:
            continue
        value = "" if raw_value is None else str(raw_value)
        argument_source = str(sources.get(str(field_name)) or "model_argument")[:128]
        declared_source_type = _source_type(argument_source)
        matching_context_type = _matching_context_type(value, contexts)
        source_identifier_type = declared_source_type or matching_context_type

        status = "VALID"
        if identifier_type in _DOMAIN_TYPES and source_identifier_type in _CONTEXT_TYPES:
            status = "CROSS_TYPE_CONTEXT_IDENTIFIER"
        elif not _format_valid(identifier_type, value):
            status = "INVALID_FORMAT"

        if status != "VALID":
            blocked = True
        validations.append(
            IdentifierValidation(
                field_name=str(field_name)[:128],
                identifier_type=identifier_type,
                validation_status=status,
                argument_source=argument_source,
                source_identifier_type=source_identifier_type,
                tool_call_id=str(tool_call_id)[:128],
                audit_event_id=str(audit_event_id)[:128],
            )
        )

    result_code = IDENTIFIER_VALIDATION_FAILED if blocked else "IDENTIFIER_VALIDATION_PASS"
    audit_records = tuple(item.to_audit_dict() for item in validations)
    return ToolArgumentValidationResult(
        allowed=not blocked,
        result_code=result_code,
        validations=tuple(validations),
        audit_records=audit_records,
    )
