"""Result-status and evidence-coverage semantics for tool execution.

An executor error is never evidence that a record, object, log, or receiver is
absent.  Status and coverage are deliberately separate fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class EvidenceStatus:
    result_status: str
    coverage: str
    negative_conclusion_allowed: bool
    executed: bool
    error_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_tool_outcome(
    *,
    executed: bool,
    error_type: str | None = None,
    result_code: str = "",
    result: Any = None,
) -> EvidenceStatus:
    """Return a conservative, deterministic status/coverage classification."""

    if error_type or result_code == "TOOL_EXECUTION_ERROR":
        return EvidenceStatus(
            result_status="TOOL_EXECUTION_ERROR",
            coverage="UNKNOWN",
            negative_conclusion_allowed=False,
            executed=bool(executed),
            error_type=str(error_type or "")[:128],
        )

    if not executed:
        return EvidenceStatus(
            result_status=result_code or "TOOL_EXECUTION_BLOCKED",
            coverage="NOT_EVALUATED",
            negative_conclusion_allowed=False,
            executed=False,
        )

    explicit_coverage = ""
    explicit_negative: bool | None = None
    if isinstance(result, Mapping):
        explicit_coverage = str(result.get("coverage") or result.get("coverage_status") or "")[:128]
        if "negative_conclusion_allowed" in result:
            explicit_negative = bool(result.get("negative_conclusion_allowed"))
        elif "negative_conclusions_allowed" in result:
            explicit_negative = bool(result.get("negative_conclusions_allowed"))

    coverage = explicit_coverage or "RESULT_AVAILABLE_COVERAGE_UNSPECIFIED"
    negative_allowed = bool(explicit_negative) if explicit_negative is not None else False
    return EvidenceStatus(
        result_status=result_code or "TOOL_EXECUTION_COMPLETED",
        coverage=coverage,
        negative_conclusion_allowed=negative_allowed,
        executed=True,
    )
