"""Trusted Nightly-RCA selective visual context adapter.

The adapter is intentionally explicit: a trusted caller supplies a run ID and
optionally 1..5 exact L#### event locators.  It does not perform natural-language
retrieval, does not create attachment records, and does not call a provider.
"""
from __future__ import annotations

import base64
import hashlib
import io
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from app.agent.utils import image_content
from app.attachment.utils import downsize_image_if_oversized
from apps.nightly_rca.t2i_selective import (
    MAX_SELECTIVE_EVENTS,
    materialize_event_micro_atlas,
    materialize_global_summary_atlas,
)

RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")
QUALIFIED_IMAGE_SIDE = 1092
QUALIFIED_MAX_IMAGE_RES = QUALIFIED_IMAGE_SIDE * QUALIFIED_IMAGE_SIDE
FROZEN_FONT_SHA256 = "39c29931201f08dd89fdba4129c76288e6baeecf7c94fe4f6a757f2b50718b1b"
FROZEN_FONT_BOLD_SHA256 = "f8aa70b5d4210d77624f5b5b5cf094fadf9fd019caf5b528fa5af42fc4ac43a5"


class NightlyRCAVisualContextError(ValueError):
    """Trusted Nightly-RCA visual context could not be safely materialized."""


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SelectiveVisualSettings:
    enabled: bool
    runs_root: Path
    max_events: int
    font_sha256: str
    font_bold_sha256: str
    allowed_emails: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "SelectiveVisualSettings":
        repo_root = Path(__file__).resolve().parents[2]
        default_runs_root = repo_root / "var" / "nightly_rca_v6" / "runs"
        raw_max = os.getenv(
            "NIGHTLY_RCA_T2I_SELECTIVE_MAX_EVENTS",
            str(MAX_SELECTIVE_EVENTS),
        )
        try:
            max_events = int(raw_max)
        except ValueError as exc:
            raise NightlyRCAVisualContextError(
                "NIGHTLY_RCA_T2I_SELECTIVE_MAX_EVENTS must be an integer"
            ) from exc
        if not 1 <= max_events <= MAX_SELECTIVE_EVENTS:
            raise NightlyRCAVisualContextError(
                "NIGHTLY_RCA_T2I_SELECTIVE_MAX_EVENTS must remain within 1..5"
            )

        runs_root = Path(
            os.getenv("NIGHTLY_RCA_T2I_RUNS_ROOT", str(default_runs_root))
        ).expanduser()
        allowed_emails = tuple(
            sorted(
                {
                    item.strip().lower()
                    for item in os.getenv(
                        "NIGHTLY_RCA_T2I_SELECTIVE_ALLOWED_EMAILS", ""
                    ).split(",")
                    if item.strip()
                }
            )
        )
        return cls(
            enabled=_env_bool("NIGHTLY_RCA_T2I_SELECTIVE_CONTEXT_ENABLED", False),
            runs_root=runs_root,
            max_events=max_events,
            font_sha256=os.getenv(
                "NIGHTLY_RCA_T2I_ATLAS_FONT_SHA256",
                FROZEN_FONT_SHA256,
            ).strip().lower(),
            font_bold_sha256=os.getenv(
                "NIGHTLY_RCA_T2I_ATLAS_FONT_BOLD_SHA256",
                FROZEN_FONT_BOLD_SHA256,
            ).strip().lower(),
            allowed_emails=allowed_emails,
        )


def _validate_request(
    request: Mapping[str, Any],
    settings: SelectiveVisualSettings,
    *,
    requester_email: str,
) -> tuple[str, str, list[str], Path]:
    if not isinstance(request, Mapping):
        raise NightlyRCAVisualContextError("nightly_rca_visual_context must be an object")
    if not settings.enabled:
        raise NightlyRCAVisualContextError(
            "Nightly-RCA selective visual context is disabled"
        )
    email = str(requester_email or "").strip().lower()
    if not settings.allowed_emails or email not in settings.allowed_emails:
        raise NightlyRCAVisualContextError(
            "requester is not authorized for Nightly-RCA selective visual context"
        )

    run_id = str(request.get("run_id") or "").strip()
    mode = str(request.get("mode") or "").strip()
    labels = [str(item).strip() for item in (request.get("event_labels") or [])]

    if not RUN_ID_RE.fullmatch(run_id):
        raise NightlyRCAVisualContextError(f"invalid Nightly-RCA run_id: {run_id!r}")
    if mode not in {"events", "global"}:
        raise NightlyRCAVisualContextError(f"unsupported visual context mode: {mode!r}")
    if mode == "events":
        if not 1 <= len(labels) <= settings.max_events:
            raise NightlyRCAVisualContextError(
                f"events mode requires 1..{settings.max_events} event labels"
            )
    elif labels:
        raise NightlyRCAVisualContextError("global mode does not accept event_labels")

    runs_root = settings.runs_root.resolve()
    run_dir = (runs_root / run_id).resolve()
    try:
        run_dir.relative_to(runs_root)
    except ValueError as exc:
        raise NightlyRCAVisualContextError("run path escapes configured runs root") from exc
    if run_dir.parent != runs_root:
        raise NightlyRCAVisualContextError("run_id must resolve to one direct run directory")

    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        raise NightlyRCAVisualContextError(f"Nightly-RCA events source missing for run {run_id}")
    return run_id, mode, labels, events_path


async def _qualified_jpeg_from_candidate_png(
    png_path: Path,
    *,
    max_resolution: int,
) -> tuple[io.BytesIO, dict[str, Any]]:
    """Apply the qualified attachment resize and deterministic JPEG boundary.

    D3F/D3H empirically froze the model-facing transform as Candidate-A
    2048x2048 PNG -> 1092x1092 LANCZOS pixels -> JPEG quality=85.  The existing
    attachment resize helper provides the pixel transform; this adapter performs
    only the final deterministic JPEG encoding in memory and never uploads it.
    """
    if int(max_resolution) != QUALIFIED_MAX_IMAGE_RES:
        raise NightlyRCAVisualContextError(
            "MAX_IMAGE_RES representation drift: "
            f"expected={QUALIFIED_MAX_IMAGE_RES} got={max_resolution}"
        )

    with png_path.open("rb") as source:
        resized = await downsize_image_if_oversized(
            source,
            png_path.name,
            max_resolution,
            bg_tracker=None,
            s3_prefix=None,
        )
    if resized is None:
        raise NightlyRCAVisualContextError("qualified image resize failed")

    resized.seek(0)
    raw = resized.read()
    if not raw.startswith(b"\xff\xd8\xff"):
        raise NightlyRCAVisualContextError(
            "qualified attachment preprocessing format drift: expected JPEG bytes"
        )
    with Image.open(io.BytesIO(raw)) as image:
        if image.size != (QUALIFIED_IMAGE_SIDE, QUALIFIED_IMAGE_SIDE):
            raise NightlyRCAVisualContextError(
                "qualified image geometry drift: "
                f"expected={QUALIFIED_IMAGE_SIDE}x{QUALIFIED_IMAGE_SIDE} got={image.width}x{image.height}"
            )

    jpeg = io.BytesIO(raw)
    jpeg.seek(0)
    return jpeg, {
        "width": QUALIFIED_IMAGE_SIDE,
        "height": QUALIFIED_IMAGE_SIDE,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "effective_mime": "image/jpeg",
    }


async def build_nightly_rca_visual_content(
    request: Mapping[str, Any],
    *,
    request_id: str,
    requester_email: str,
    max_resolution: int,
    visual_settings: SelectiveVisualSettings | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return model content blocks and provenance for one trusted Nightly request."""
    settings = visual_settings or SelectiveVisualSettings.from_env()
    run_id, mode, labels, events_path = _validate_request(
        request, settings, requester_email=requester_email
    )

    request_sha = hashlib.sha256(str(request_id or "").encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="nightly_rca_t2i_selective.") as tmp:
        out_dir = Path(tmp)
        if mode == "events":
            materialized = materialize_event_micro_atlas(
                events_path,
                out_dir,
                event_labels=labels,
                request_id=request_id,
                expected_font_sha256=settings.font_sha256,
                expected_font_bold_sha256=settings.font_bold_sha256,
            )
        else:
            materialized = materialize_global_summary_atlas(
                events_path,
                out_dir,
                request_id=request_id,
                expected_font_sha256=settings.font_sha256,
                expected_font_bold_sha256=settings.font_bold_sha256,
            )

        png_path = Path(materialized["image_path"])
        model_image, model_meta = await _qualified_jpeg_from_candidate_png(
            png_path,
            max_resolution=max_resolution,
        )
        # Preserve the attachment boundary that was benchmarked: declared PNG
        # metadata with byte-signature normalization to effective JPEG.
        blocks = image_content(png_path.name, "image/png", model_image)

    image_blocks = [block for block in blocks if block.get("type") == "image"]
    if len(image_blocks) != 1:
        raise NightlyRCAVisualContextError(
            f"expected exactly one model image block; got {len(image_blocks)}"
        )
    image_block = image_blocks[0]
    if image_block.get("mime_type") != "image/jpeg":
        raise NightlyRCAVisualContextError(
            f"effective model MIME drift: {image_block.get('mime_type')!r}"
        )
    decoded = base64.b64decode(str(image_block.get("data") or ""))
    if hashlib.sha256(decoded).hexdigest() != model_meta["sha256"]:
        raise NightlyRCAVisualContextError("model image byte provenance mismatch")

    provenance = {
        "schema": "nightly-rca-selective-context/1.0",
        "run_id": run_id,
        "mode": mode,
        "event_labels": labels,
        "source_sha256": materialized["source_sha256"],
        "artifact_key": materialized["artifact_key"],
        "candidate_png_sha256": materialized["image"]["sha256"],
        "candidate_png_width": materialized["image"]["width"],
        "candidate_png_height": materialized["image"]["height"],
        "model_image_sha256": model_meta["sha256"],
        "model_image_width": model_meta["width"],
        "model_image_height": model_meta["height"],
        "model_image_mime": model_meta["effective_mime"],
        "request_id_sha256": request_sha,
        "attachment_persisted": False,
        "model_image_count": 1,
    }
    return blocks, provenance
