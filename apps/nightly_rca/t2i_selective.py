"""Deterministic selective T2I Log Atlas materializer.

This module reuses the production Candidate-A raster grammar from
``apps.nightly_rca.t2i_atlas`` while restricting each model-facing artifact to
one page selected by trusted event locators, or to one deterministic global
summary page.

It performs no provider, MCP, HTTP, database, or storage calls and never
mutates the source ``events.jsonl``.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

from . import t2i_atlas as base

SCHEMA = "t2i-selective-visual/1.0"
EVENT_LABEL_RE = re.compile(r"^L\d{4}$")
MAX_SELECTIVE_EVENTS = 5


class T2ISelectiveError(base.T2IAtlasError):
    """Selective materialization failed before a complete artifact was published."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_fonts(
    *,
    expected_font_sha256: str,
    expected_font_bold_sha256: str,
) -> dict[str, str]:
    actual_regular = _sha256_path(base.FONT_REG)
    actual_bold = _sha256_path(base.FONT_BOLD)
    expected_regular = str(expected_font_sha256 or "").strip().lower()
    expected_bold = str(expected_font_bold_sha256 or "").strip().lower()

    if expected_regular and actual_regular != expected_regular:
        raise T2ISelectiveError(
            "regular font SHA256 drift: "
            f"expected={expected_regular} got={actual_regular}"
        )
    if expected_bold and actual_bold != expected_bold:
        raise T2ISelectiveError(
            "bold font SHA256 drift: "
            f"expected={expected_bold} got={actual_bold}"
        )
    return {
        "regular_sha256": actual_regular,
        "bold_sha256": actual_bold,
    }


def _load_source(events_path: Path):
    events_path = Path(events_path)
    if not events_path.is_file():
        raise T2ISelectiveError(f"events source missing: {events_path}")
    raw_source, records = base._read_source(events_path)
    source_sha = _sha256_bytes(raw_source)
    registry, template_map = base._template_registry(records)
    by_label = {f"L{rec.source_line:04d}": rec for rec in records}
    return raw_source, records, source_sha, registry, template_map, by_label


def _normalize_event_labels(event_labels: Sequence[str]) -> list[str]:
    if isinstance(event_labels, (str, bytes)):
        raise T2ISelectiveError("event_labels must be a sequence, not a string")
    labels = [str(label).strip() for label in event_labels]
    if not 1 <= len(labels) <= MAX_SELECTIVE_EVENTS:
        raise T2ISelectiveError(
            f"events mode requires 1..{MAX_SELECTIVE_EVENTS} labels; got {len(labels)}"
        )
    if any(not EVENT_LABEL_RE.fullmatch(label) for label in labels):
        raise T2ISelectiveError(f"invalid event label set: {labels!r}")
    if len(set(labels)) != len(labels):
        raise T2ISelectiveError("duplicate event labels are not allowed")
    return labels


def _wrap_line(text: str, kind: str) -> list[tuple[str, str]]:
    Image, ImageDraw, _ImageFont, font, _bold, _header, _small = base._load_fonts()
    max_px = base.W - 2 * base.MARGIN
    wrapped = base._hard_wrap(
        text,
        font=font,
        max_px=max_px,
        Image=Image,
        ImageDraw=ImageDraw,
    )
    return [(kind, row) for row in wrapped]


def _build_event_page(
    records: list[Any],
    registry: list[dict[str, Any]],
    template_map: dict[tuple[Any, ...], str],
    by_label: dict[str, Any],
    labels: list[str],
    source_sha: str,
) -> tuple[bytes, dict[str, Any]]:
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise T2ISelectiveError(f"unknown event labels: {missing}")

    reg_by_id = {row["id"]: row for row in registry}
    template_ids: list[str] = []
    for label in labels:
        rec = by_label[label]
        template_id = template_map[base._template_key(rec.value)]
        if template_id not in template_ids:
            template_ids.append(template_id)

    page: list[tuple[str, str]] = []
    for template_id in template_ids:
        row = reg_by_id[template_id]
        server = base.SERVER_ALIAS.get(str(row["server"]), str(row["server"]))
        text = (
            f"{template_id} {server}/{row['tool']} "
            f"uses={row['count']} first={row['first_src_line']:04d}"
        )
        page.extend(_wrap_line(text, "template"))

    Image, ImageDraw, _ImageFont, font, _bold, _header, _small = base._load_fonts()
    max_px = base.W - 2 * base.MARGIN
    compact_rows: list[str] = []
    event_index: list[dict[str, Any]] = []

    for label in labels:
        rec = by_label[label]
        template_id = template_map[base._template_key(rec.value)]
        raw_rows, displayed_fields = base._event_rows(label, template_id, rec)

        compact_rows.append(
            json.dumps(
                {
                    "label": label,
                    "template": template_id,
                    "source_line": rec.source_line,
                    "raw_sha256": rec.raw_sha256,
                    "rows": raw_rows,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )

        page_start = len(page) + 1
        rendered_count = 0
        for row_index, raw_row in enumerate(raw_rows):
            wrapped = base._wrap_pipe(
                raw_row,
                font=font,
                max_px=max_px,
                Image=Image,
                ImageDraw=ImageDraw,
            )
            for wrapped_index, rendered in enumerate(wrapped):
                kind = (
                    "event_primary"
                    if row_index == 0 and wrapped_index == 0
                    else "event_detail"
                )
                page.append((kind, rendered))
                rendered_count += 1

        event_index.append(
            {
                "label": label,
                "source_line": rec.source_line,
                "raw_sha256": rec.raw_sha256,
                "template_id": template_id,
                "displayed_fields": displayed_fields,
                "page": 1,
                "page_row_start": page_start,
                "page_row_count": rendered_count,
            }
        )

    if len(page) > base.HARD_MAX_ROWS_PER_PAGE:
        raise T2ISelectiveError(
            f"selective event page exceeds one-page capacity: rows={len(page)} "
            f"limit={base.HARD_MAX_ROWS_PER_PAGE}"
        )

    compact_text = "\n".join(compact_rows) + "\n"
    compact_sha = _sha256_bytes(compact_text.encode())
    scope = "SELECTIVE selected=" + ",".join(labels)
    png = base._render_png(
        page,
        page_number=1,
        total_pages=1,
        scope=scope,
        source_sha256=source_sha,
        compact_sha256=compact_sha,
    )
    return png, {
        "mode": "events",
        "labels": labels,
        "template_ids": template_ids,
        "row_count": len(page),
        "compact_sha256": compact_sha,
        "event_index": event_index,
    }


def _build_global_page(
    records: list[Any],
    registry: list[dict[str, Any]],
    template_map: dict[tuple[Any, ...], str],
    source_sha: str,
) -> tuple[bytes, dict[str, Any]]:
    if not records:
        raise T2ISelectiveError("cannot summarize an empty events source")

    server_counts = collections.Counter(
        str(rec.value.get("server") or "") for rec in records
    )
    tool_counts = collections.Counter(
        str(rec.value.get("tool") or "")
        for rec in records
        if rec.value.get("tool")
    )
    failed: list[tuple[str, str, str, str]] = []
    for rec in records:
        if str(rec.value.get("status") or "") == "OK":
            continue
        template_id = template_map[base._template_key(rec.value)]
        arguments = rec.value.get("arguments") or {}
        receiver = arguments.get("receiver_id", "") if isinstance(arguments, dict) else ""
        failed.append(
            (
                f"L{rec.source_line:04d}",
                template_id,
                str(rec.value.get("tool") or ""),
                str(receiver),
            )
        )

    first, last = records[0], records[-1]
    first_template = template_map[base._template_key(first.value)]
    last_template = template_map[base._template_key(last.value)]
    top = tool_counts.most_common(4)

    lines = [
        "GLOBAL SUMMARY",
        f"PAGE_COUNT=1 | SOURCE_EVENT_COUNT={len(records)} | TEMPLATE_COUNT={len(registry)}",
        "SERVER_EVENT_COUNTS="
        + " ".join(f"{key}={value}" for key, value in sorted(server_counts.items())),
    ]
    for label, template_id, tool, receiver in failed:
        lines.extend(
            [
                f"FAILED_EVENT_LABEL={label} | FAILED_EVENT_TEMPLATE={template_id}",
                f"FAILED_EVENT_TOOL={tool} | FAILED_EVENT_RECEIVER={receiver}",
            ]
        )
    lines.extend(
        [
            f"FIRST_EVENT=L{first.source_line:04d}/{first_template}/"
            f"{first.value.get('server')}/{first.value.get('tool')}",
            f"LAST_EVENT=L{last.source_line:04d}/{last_template}/"
            f"{last.value.get('server')}/{last.value.get('tool')}",
            "TOP_TOOL_COUNTS=" + " ".join(f"{key}={value}" for key, value in top),
        ]
    )

    page: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        page.extend(_wrap_line(line, "anomaly" if index == 0 else "template"))
    if len(page) > base.HARD_MAX_ROWS_PER_PAGE:
        raise T2ISelectiveError(
            f"global summary exceeds one-page capacity: rows={len(page)}"
        )

    compact_text = "\n".join(lines) + "\n"
    compact_sha = _sha256_bytes(compact_text.encode())
    png = base._render_png(
        page,
        page_number=1,
        total_pages=1,
        scope="GLOBAL_01 deterministic run summary",
        source_sha256=source_sha,
        compact_sha256=compact_sha,
    )
    return png, {
        "mode": "global",
        "labels": [],
        "row_count": len(page),
        "compact_sha256": compact_sha,
        "summary_schema": [
            "page_count",
            "source_event_count",
            "template_count",
            "server_event_counts",
            "failed_events",
            "first_event",
            "last_event",
            "top_tool_counts",
        ],
    }


def _publish_one_page(
    *,
    events_path: Path,
    out_dir: Path,
    mode: str,
    labels: list[str],
    request_id: str,
    expected_font_sha256: str,
    expected_font_bold_sha256: str,
) -> dict[str, Any]:
    raw_source, records, source_sha, registry, template_map, by_label = _load_source(
        events_path
    )
    font_identity = _validate_fonts(
        expected_font_sha256=expected_font_sha256,
        expected_font_bold_sha256=expected_font_bold_sha256,
    )

    if mode == "events":
        png, meta = _build_event_page(
            records, registry, template_map, by_label, labels, source_sha
        )
    elif mode == "global":
        png, meta = _build_global_page(records, registry, template_map, source_sha)
    else:
        raise T2ISelectiveError(f"unsupported selective mode: {mode!r}")

    request_sha = _sha256_bytes(str(request_id or "").encode())
    identity_payload = {
        "schema": SCHEMA,
        "source_sha256": source_sha,
        "mode": mode,
        "labels": labels,
        "font_regular_sha256": font_identity["regular_sha256"],
        "font_bold_sha256": font_identity["bold_sha256"],
        "base_codec": base.CODEC,
        "base_profile": base.PROFILE,
    }
    artifact_key = _sha256_bytes(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir = out_dir / f"t2i_selective_{artifact_key[:20]}"
    image_name = "selective_atlas.png"
    manifest_name = "selective_manifest.json"

    manifest = {
        **identity_payload,
        "artifact_key": artifact_key,
        "source_bytes": len(raw_source),
        "source_records": len(records),
        "template_count": len(registry),
        "page_count": 1,
        "image": {
            "name": image_name,
            "sha256": _sha256_bytes(png),
            "bytes": len(png),
            "width": base.W,
            "height": base.H,
            "font_px": base.FONT_SIZE,
            "rows": meta["row_count"],
        },
        "selection": meta,
    }

    if final_dir.exists():
        manifest_path = final_dir / manifest_name
        image_path = final_dir / image_name
        if not manifest_path.is_file() or not image_path.is_file():
            raise T2ISelectiveError(f"incomplete existing selective artifact: {final_dir}")
        existing = json.loads(manifest_path.read_text())
        if (
            existing.get("artifact_key") != artifact_key
            or existing.get("image", {}).get("sha256") != manifest["image"]["sha256"]
            or _sha256_path(image_path) != manifest["image"]["sha256"]
        ):
            raise T2ISelectiveError(f"conflicting existing selective artifact: {final_dir}")
        return {
            **manifest,
            "artifact_dir": str(final_dir),
            "image_path": str(image_path),
            "manifest_path": str(manifest_path),
            "request_id_sha256": request_sha,
            "idempotent_reuse": True,
        }

    staging = Path(
        tempfile.mkdtemp(prefix=".t2i_selective.staging.", dir=str(out_dir))
    )
    try:
        base._atomic_bytes(staging / image_name, png)
        base._atomic_text(
            staging / manifest_name,
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        os.replace(staging, final_dir)
    except Exception:
        if staging.exists():
            import shutil

            shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        **manifest,
        "artifact_dir": str(final_dir),
        "image_path": str(final_dir / image_name),
        "manifest_path": str(final_dir / manifest_name),
        "request_id_sha256": request_sha,
        "idempotent_reuse": False,
    }


def materialize_event_micro_atlas(
    events_path: Path,
    out_dir: Path,
    *,
    event_labels: Sequence[str],
    request_id: str,
    expected_font_sha256: str,
    expected_font_bold_sha256: str,
) -> dict[str, Any]:
    """Materialize exactly one Candidate-A page for 1..5 trusted L#### labels."""
    labels = _normalize_event_labels(event_labels)
    return _publish_one_page(
        events_path=Path(events_path),
        out_dir=Path(out_dir),
        mode="events",
        labels=labels,
        request_id=request_id,
        expected_font_sha256=expected_font_sha256,
        expected_font_bold_sha256=expected_font_bold_sha256,
    )


def materialize_global_summary_atlas(
    events_path: Path,
    out_dir: Path,
    *,
    request_id: str,
    expected_font_sha256: str,
    expected_font_bold_sha256: str,
) -> dict[str, Any]:
    """Materialize the frozen one-page global summary for one Nightly run."""
    return _publish_one_page(
        events_path=Path(events_path),
        out_dir=Path(out_dir),
        mode="global",
        labels=[],
        request_id=request_id,
        expected_font_sha256=expected_font_sha256,
        expected_font_bold_sha256=expected_font_bold_sha256,
    )
