"""Deterministic T2I Log Atlas v3 Candidate-A renderer.

This module is a local observability artifact generator.  It never calls a
provider or MCP tool and it never mutates the source events file.  Candidate A
is the representation selected by the frozen 2026-08-12 blind benchmark.

The renderer preserves these invariants:
- source JSONL SHA256/byte count and every non-empty source line are frozen;
- malformed JSONL is a hard generation error (no silent event loss);
- every source event receives exactly one L#### label and a per-record raw SHA;
- template identity is structural only (server/tool/argument/response schema);
- every raster event is traceable through the manifest event_index;
- output writes are atomic and deterministic for identical source + font bytes;
- pagination never silently drops event groups or anomaly rows.
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CODEC = "t2i-log-atlas/3.1"
PROFILE = "candidate_a"

W = H = 2048
MARGIN = 24
HEADER_H = 72
FOOTER_H = 28
FONT_SIZE = 18
LINE_H = 20
TARGET_ROWS_PER_PAGE = 90
HARD_MAX_ROWS_PER_PAGE = 96

FONT_REG = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")

ALIASES = {
    "rsp.receiver_id": "RX",
    "arg.receiver_id": "RX",
    "arg.receiver_ids": "RX",
    "rsp.result": "RES",
    "rsp.result_code": "RESC",
    "rsp.receipt_status": "RCPT",
    "rsp.workflow_status": "WF",
    "rsp.ok": "OK",
    "rsp.issue_profile": "IP",
    "arg.profile": "PROF",
    "rsp.profile": "PROF",
    "rsp.candidate_count": "CAND",
    "rsp.receiver_count": "RXN",
    "rsp.tracker_count": "TRK",
    "rsp.case_count": "CASE",
    "rsp.cases_returned": "CASES_RET",
    "rsp.failure_count": "FAIL",
    "rsp.integrity_code_counts": "ICODES",
    "rsp.total_alert_count": "ALERTS",
    "rsp.document_count": "DOCS",
    "rsp.total_matches": "MATCHES",
    "rsp.profile_count": "PCNT",
    "rsp.runtime_count": "RUNC",
    "rsp.security_note": "SEC",
    "rsp.schema_version": "SCHEMA",
    "error": "ERR",
    "rsp.notes": "NOTE",
    "rsp.newest": "NEWEST",
    "rsp.oldest": "OLDEST",
    "rsp.returned_item_count": "RET",
    "rsp.upstream_error_type": "UPERR",
    "arg.outcome_status": "OUTCOME",
}

ROW1_KEYS = [
    "rsp.receiver_id",
    "arg.receiver_id",
    "arg.receiver_ids",
    "rsp.result",
    "rsp.result_code",
    "rsp.receipt_status",
    "rsp.workflow_status",
]
ROW2_KEYS = [
    "rsp.ok",
    "rsp.issue_profile",
    "arg.profile",
    "rsp.profile",
    "rsp.candidate_count",
    "rsp.receiver_count",
    "rsp.tracker_count",
    "rsp.case_count",
    "rsp.cases_returned",
    "rsp.failure_count",
    "rsp.integrity_code_counts",
    "rsp.total_alert_count",
    "rsp.document_count",
    "rsp.total_matches",
    "rsp.profile_count",
    "rsp.runtime_count",
    "rsp.security_note",
    "rsp.schema_version",
    "error",
    "rsp.newest",
    "rsp.oldest",
    "rsp.returned_item_count",
    "rsp.upstream_error_type",
    "arg.outcome_status",
]

SERVER_ALIAS = {
    "s3_stb_logs": "S3",
    "rtr_alerts_mcp": "RTR",
    "grasshopper_mcp": "GH",
}


class T2IAtlasError(RuntimeError):
    """Generation failed before a complete, self-consistent atlas was written."""


@dataclass(frozen=True)
class SourceRecord:
    source_line: int
    raw: str
    raw_sha256: str
    value: dict[str, Any]


@dataclass
class RenderGroup:
    kind: str
    lines: list[tuple[str, str]]
    label: str = ""
    template_id: str = ""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _atomic_text(path: Path, text: str) -> None:
    _atomic_bytes(path, text.encode("utf-8"))


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _read_source(path: Path) -> tuple[bytes, list[SourceRecord]]:
    raw_bytes = path.read_bytes()
    try:
        source_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise T2IAtlasError(f"source is not valid UTF-8: {exc}") from exc

    records: list[SourceRecord] = []
    malformed: list[tuple[int, str]] = []
    for source_line, line in enumerate(source_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as exc:
            malformed.append((source_line, f"{exc.msg}@{exc.colno}"))
            continue
        if not isinstance(obj, dict):
            malformed.append((source_line, "top-level JSON value is not an object"))
            continue
        records.append(
            SourceRecord(
                source_line=source_line,
                raw=stripped,
                raw_sha256=_sha256_bytes(stripped.encode("utf-8")),
                value=obj,
            )
        )

    if malformed:
        preview = ", ".join(f"line {line}: {err}" for line, err in malformed[:5])
        raise T2IAtlasError(
            f"malformed source records={len(malformed)}; refusing silent loss ({preview})"
        )
    if not records:
        raise T2IAtlasError("source contains zero JSON event records")
    return raw_bytes, records


def _struct_sig(value: Any) -> tuple[Any, ...]:
    if isinstance(value, dict):
        return ("dict", tuple(sorted(value.keys())))
    if isinstance(value, list):
        if not value:
            return ("list_empty",)
        return ("list", _struct_sig(value[0]))
    return ("scalar",)


def _template_key(rec: dict[str, Any]) -> tuple[Any, ...]:
    server = str(rec.get("server") or "")
    tool = str(rec.get("tool") or "")
    args = rec.get("arguments", {})
    response = rec.get("response", {})
    arg_keys = tuple(sorted(args.keys())) if isinstance(args, dict) else ()
    if isinstance(response, dict):
        response_keys = tuple(sorted(response.keys()))
    elif isinstance(response, list) and response and isinstance(response[0], dict):
        response_keys = tuple(sorted(response[0].keys()))
    else:
        response_keys = ()
    return (server, tool, arg_keys, _struct_sig(response), response_keys)


def _template_registry(records: list[SourceRecord]) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], str]]:
    registry: list[dict[str, Any]] = []
    mapping: dict[tuple[Any, ...], dict[str, Any]] = {}
    for rec in records:
        key = _template_key(rec.value)
        if key not in mapping:
            entry = {
                "id": f"T{len(registry) + 1:03d}",
                "server": key[0],
                "tool": key[1],
                "arg_keys": list(key[2]),
                "resp_keys": list(key[4]),
                "first_src_line": rec.source_line,
                "count": 0,
            }
            mapping[key] = entry
            registry.append(entry)
        mapping[key]["count"] += 1
    return registry, {key: entry["id"] for key, entry in mapping.items()}


def _bounded_scalar(value: Any) -> str:
    rendered = str(value)
    if len(rendered) > 120:
        return rendered[:117] + "..."
    return rendered


def _extract_vars(rec: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}

    args = rec.get("arguments", {})
    if isinstance(args, dict):
        for key, value in args.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                values[f"arg.{key}"] = _bounded_scalar(value)
            elif isinstance(value, (list, dict)):
                values[f"arg.{key}"] = _bounded_scalar(
                    json.dumps(value, sort_keys=True, ensure_ascii=False)
                )

    response = rec.get("response", {})
    if isinstance(response, dict):
        for key, value in response.items():
            if str(key).startswith("_"):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                values[f"rsp.{key}"] = _bounded_scalar(value)
            elif isinstance(value, list) and len(value) <= 3:
                values[f"rsp.{key}"] = _bounded_scalar(
                    json.dumps(value, sort_keys=True, ensure_ascii=False)
                )
            elif isinstance(value, dict) and len(value) <= 4:
                values[f"rsp.{key}"] = _bounded_scalar(
                    json.dumps(value, sort_keys=True, ensure_ascii=False)
                )
    elif isinstance(response, list):
        values["rsp._list_len"] = str(len(response))
        if response and isinstance(response[0], dict):
            for key, value in list(response[0].items())[:6]:
                if isinstance(value, (str, int, float, bool)) or value is None:
                    values[f"rsp[0].{key}"] = _bounded_scalar(value)

    error = rec.get("error")
    if error:
        # v2 compatibility: error was clipped to 120 characters before the
        # canonical-stream truncation stage, so it remains display-eligible.
        values["error"] = str(error)[:120]
    return values


def _good(value: Any) -> bool:
    rendered = str(value)
    return (
        rendered not in {"", "[]", "{}", "None"}
        and "<redacted>" not in rendered
        and "..." not in rendered
    )


def _event_rows(label: str, template_id: str, rec: SourceRecord) -> tuple[list[str], list[str]]:
    values = _extract_vars(rec.value)
    primary = [
        label,
        template_id,
        f"ST={rec.value.get('status', '')}",
        f"E={rec.value.get('elapsed_ms', '')}ms",
        f"H={rec.raw_sha256[:8]}",
    ]
    displayed_fields: list[str] = []
    seen: set[str] = set()

    for key in ROW1_KEYS:
        if key in values and _good(values[key]):
            alias = ALIASES[key]
            if alias == "RX" and alias in seen:
                continue
            seen.add(alias)
            primary.append(f"{alias}={values[key]}")
            displayed_fields.append(key)

    detail: list[str] = []
    seen.clear()
    for key in ROW2_KEYS:
        if key in values and _good(values[key]):
            alias = ALIASES[key]
            if alias == "PROF" and alias in seen:
                continue
            seen.add(alias)
            detail.append(f"{alias}={values[key]}")
            displayed_fields.append(key)

    if not detail:
        for key, value in values.items():
            if (
                key.startswith(("arg.", "rsp."))
                and _good(value)
                and key not in ROW1_KEYS
                and key != "rsp.notes"
            ):
                detail.append(f"{key}={value}")
                displayed_fields.append(key)
                if len(detail) >= 2:
                    break

    rows = [" | ".join(primary), "  " + " | ".join(detail)]
    note = values.get("rsp.notes")
    if note and _good(note):
        rows.append("  NOTE=" + str(note))
        if "rsp.notes" not in displayed_fields:
            displayed_fields.append("rsp.notes")
    return rows, displayed_fields


def _anomaly_lines(records: list[SourceRecord], template_ids: list[str]) -> list[str]:
    failed: list[str] = []
    false_ok: list[tuple[str, str, str]] = []
    workflows: collections.Counter[str] = collections.Counter()
    receipts: collections.Counter[str] = collections.Counter()
    tools: collections.Counter[str] = collections.Counter()

    for index, rec in enumerate(records, start=1):
        value = rec.value
        label = f"L{rec.source_line:04d}"
        template_id = template_ids[index - 1]
        tool = str(value.get("tool") or "")
        if tool:
            tools[tool] += 1
        if str(value.get("status") or "") != "OK":
            args = value.get("arguments") or {}
            receiver = args.get("receiver_id", "") if isinstance(args, dict) else ""
            failed.append(
                f"{label} {template_id} "
                f"{SERVER_ALIAS.get(str(value.get('server') or ''), str(value.get('server') or ''))}/"
                f"{tool} RX={receiver}"
            )
        response = value.get("response")
        if isinstance(response, dict):
            if response.get("ok") is False:
                false_ok.append(
                    (
                        label,
                        template_id,
                        str(response.get("result_code") or response.get("result") or ""),
                    )
                )
            if response.get("workflow_status") is not None:
                workflows[str(response["workflow_status"])] += 1
            if response.get("receipt_status") is not None:
                receipts[str(response["receipt_status"])] += 1

    lines = ["ANOMALY INDEX"]
    lines.extend("STEP_FAILED: " + row for row in failed)
    lines.extend(
        f"RSP_OK_FALSE: {label} {template_id} {result}"
        for label, template_id, result in false_ok
    )
    lines.append("TRACKER_OUTCOMES: " + " ".join(f"{k}={v}" for k, v in workflows.items()))
    lines.append("RECEIPT_STATUS: " + " ".join(f"{k}={v}" for k, v in receipts.items()))
    top = tools.most_common(6)
    lines.append("TOP TOOLS: " + " ".join(f"{k}={v}" for k, v in top[:3]))
    lines.append("TOP TOOLS: " + " ".join(f"{k}={v}" for k, v in top[3:6]))
    return lines


def _load_fonts():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise T2IAtlasError(f"Pillow unavailable: {exc}") from exc
    for path in (FONT_REG, FONT_BOLD):
        if not path.is_file():
            raise T2IAtlasError(f"required deterministic font missing: {path}")
    return (
        Image,
        ImageDraw,
        ImageFont,
        ImageFont.truetype(str(FONT_REG), FONT_SIZE),
        ImageFont.truetype(str(FONT_BOLD), FONT_SIZE),
        ImageFont.truetype(str(FONT_BOLD), 24),
        ImageFont.truetype(str(FONT_REG), 14),
    )


def _text_width(text: str, font: Any, Image: Any, ImageDraw: Any) -> int:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    return int(draw.textbbox((0, 0), text, font=font)[2])


def _hard_wrap(text: str, *, font: Any, max_px: int, Image: Any, ImageDraw: Any) -> list[str]:
    if _text_width(text, font, Image, ImageDraw) <= max_px:
        return [text]
    prefix = "  " if text.startswith("  ") else ""
    content = text[len(prefix):]
    output: list[str] = []
    while content:
        lo, hi, best = 1, len(content), 1
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = prefix + content[:mid]
            if _text_width(candidate, font, Image, ImageDraw) <= max_px:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        output.append(prefix + content[:best])
        content = content[best:]
        prefix = "  "
    return output


def _wrap_pipe(text: str, *, font: Any, max_px: int, Image: Any, ImageDraw: Any) -> list[str]:
    if _text_width(text, font, Image, ImageDraw) <= max_px:
        return [text]
    tokens = text.split(" | ")
    output: list[str] = []
    current = tokens[0]
    for token in tokens[1:]:
        candidate = current + " | " + token
        if _text_width(candidate, font, Image, ImageDraw) <= max_px:
            current = candidate
        else:
            output.extend(
                _hard_wrap(current, font=font, max_px=max_px, Image=Image, ImageDraw=ImageDraw)
            )
            current = "  " + token
    output.extend(_hard_wrap(current, font=font, max_px=max_px, Image=Image, ImageDraw=ImageDraw))
    return output


def _prepare_groups(
    records: list[SourceRecord],
    registry: list[dict[str, Any]],
    template_map: dict[tuple[Any, ...], str],
) -> tuple[list[RenderGroup], list[dict[str, Any]]]:
    Image, ImageDraw, _ImageFont, font, _bold, _header, _small = _load_fonts()
    max_px = W - 2 * MARGIN

    groups: list[RenderGroup] = []
    for template in registry:
        server = SERVER_ALIAS.get(str(template["server"]), str(template["server"]))
        text = (
            f"{template['id']} {server}/{template['tool']} "
            f"uses={template['count']} first={template['first_src_line']:04d}"
        )
        wrapped = _hard_wrap(text, font=font, max_px=max_px, Image=Image, ImageDraw=ImageDraw)
        groups.append(RenderGroup("template", [("template", row) for row in wrapped]))

    event_index: list[dict[str, Any]] = []
    template_ids: list[str] = []
    for index, rec in enumerate(records, start=1):
        label = f"L{rec.source_line:04d}"
        template_id = template_map[_template_key(rec.value)]
        template_ids.append(template_id)
        raw_rows, displayed_fields = _event_rows(label, template_id, rec)
        rendered: list[tuple[str, str]] = []
        for row_index, row in enumerate(raw_rows):
            wrapped = _wrap_pipe(
                row, font=font, max_px=max_px, Image=Image, ImageDraw=ImageDraw
            )
            for wrapped_index, wrapped_row in enumerate(wrapped):
                row_type = (
                    "event_primary"
                    if row_index == 0 and wrapped_index == 0
                    else "event_detail"
                )
                rendered.append((row_type, wrapped_row))
        groups.append(
            RenderGroup(
                "event",
                rendered,
                label=label,
                template_id=template_id,
            )
        )
        event_index.append(
            {
                "label": label,
                "source_line": rec.source_line,
                "raw_sha256": rec.raw_sha256,
                "tmpl_id": template_id,
                "server": str(rec.value.get("server") or ""),
                "tool": str(rec.value.get("tool") or ""),
                "status": str(rec.value.get("status") or ""),
                "elapsed_ms": rec.value.get("elapsed_ms"),
                "displayed_fields": displayed_fields,
                "page": None,
                "page_row_start": None,
                "page_row_count": len(rendered),
            }
        )

    for line in _anomaly_lines(records, template_ids):
        wrapped = _hard_wrap(
            line, font=font, max_px=max_px, Image=Image, ImageDraw=ImageDraw
        )
        groups.append(RenderGroup("anomaly", [("anomaly", row) for row in wrapped]))

    return groups, event_index


def _paginate(
    groups: list[RenderGroup],
    event_index: list[dict[str, Any]],
) -> list[list[tuple[str, str]]]:
    pages: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []

    event_lookup = {row["label"]: row for row in event_index}

    for group in groups:
        group_len = len(group.lines)
        if group_len > HARD_MAX_ROWS_PER_PAGE:
            raise T2IAtlasError(
                f"single render group exceeds hard page capacity: kind={group.kind} rows={group_len}"
            )

        if current and len(current) + group_len > TARGET_ROWS_PER_PAGE:
            pages.append(current)
            current = []

        # A group may be between the soft target and hard physical capacity.
        if len(current) + group_len > HARD_MAX_ROWS_PER_PAGE:
            if current:
                pages.append(current)
                current = []
            if group_len > HARD_MAX_ROWS_PER_PAGE:
                raise T2IAtlasError("render group cannot fit on a page")

        if group.kind == "event":
            row = event_lookup[group.label]
            row["page"] = len(pages) + 1
            row["page_row_start"] = len(current) + 1
        current.extend(group.lines)

    if current:
        pages.append(current)
    if not pages:
        raise T2IAtlasError("pagination produced zero pages")
    if any(len(page) > HARD_MAX_ROWS_PER_PAGE for page in pages):
        raise T2IAtlasError("pagination exceeded physical page capacity")
    return pages


def _page_scope(
    page: list[tuple[str, str]],
    *,
    page_number: int,
    registry: list[dict[str, Any]],
    event_index: list[dict[str, Any]],
) -> str:
    labels = [
        row["label"]
        for row in event_index
        if row.get("page") == page_number
    ]
    template_ids: list[str] = []
    for kind, text in page:
        if kind != "template":
            continue
        first = text.strip().split(" ", 1)[0]
        if first.startswith("T") and first[1:].isdigit():
            template_ids.append(first)
    has_anomaly = any(kind == "anomaly" for kind, _ in page)

    parts: list[str] = []
    if template_ids:
        parts.append(f"templates {template_ids[0]}-{template_ids[-1]}")
    if labels:
        parts.append(f"events {labels[0]}-{labels[-1]}")
    if has_anomaly:
        parts.append("anomaly index")
    return " + ".join(parts) or f"page {page_number}"


def _render_png(
    page: list[tuple[str, str]],
    *,
    page_number: int,
    total_pages: int,
    scope: str,
    source_sha256: str,
    compact_sha256: str,
) -> bytes:
    Image, ImageDraw, _ImageFont, font, bold, header_font, small_font = _load_fonts()
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, W, HEADER_H], fill="black")
    draw.text(
        (MARGIN, 10),
        (
            f"T2I-LOG-ATLAS v3.1 | page {page_number}/{total_pages} | "
            f"native=2048x2048 | font={FONT_SIZE}px | rows={len(page)}"
        ),
        font=header_font,
        fill="white",
    )
    draw.text(
        (MARGIN, 43),
        f"{scope} | source_sha256={source_sha256}",
        font=small_font,
        fill="white",
    )

    y = HEADER_H + 10
    for kind, text in page:
        if kind == "event_primary":
            draw.rectangle(
                [MARGIN - 4, y - 1, W - MARGIN + 4, y + LINE_H - 1],
                fill=(242, 242, 242),
            )
            active_font = bold
        elif kind == "anomaly":
            draw.rectangle(
                [MARGIN - 4, y - 1, W - MARGIN + 4, y + LINE_H - 1],
                fill=(232, 232, 232),
            )
            active_font = bold if text == "ANOMALY INDEX" else font
        else:
            active_font = font

        width = int(draw.textbbox((0, 0), text, font=active_font)[2])
        if width > W - 2 * MARGIN:
            raise T2IAtlasError(f"horizontal overflow after wrapping: {text}")
        draw.text((MARGIN, y), text, font=active_font, fill="black")
        y += LINE_H

    if y > H - FOOTER_H:
        raise T2IAtlasError(f"vertical overflow y={y}")

    footer_y = H - FOOTER_H
    draw.line((0, footer_y, W, footer_y), fill=(100, 100, 100), width=1)
    draw.text(
        (MARGIN, footer_y + 5),
        (
            "schema=critical-fields-v1 | H=raw_sha256[:8] | "
            f"compact_sha256={compact_sha256[:16]}..."
        ),
        font=small_font,
        fill=(40, 40, 40),
    )

    output = io.BytesIO()
    img.save(output, "PNG", optimize=False)
    return output.getvalue()


def _tree_fingerprints(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256_path(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def generate_t2i_v3_atlas(
    events_path: Path,
    out_dir: Path,
    *,
    profile: str = PROFILE,
    expected_font_sha256: str = "",
    expected_font_bold_sha256: str = "",
) -> dict[str, Any]:
    """Generate the selected production T2I atlas from one completed events.jsonl.

    Artifacts are built in a private staging directory and published with one
    directory rename.  If an identical final directory already exists, the
    operation is idempotent.  A conflicting existing atlas is never overwritten.
    """
    events_path = Path(events_path)
    out_dir = Path(out_dir)
    if profile != PROFILE:
        raise T2IAtlasError(f"unsupported atlas profile: {profile!r}")
    if not events_path.is_file():
        raise T2IAtlasError(f"events source missing: {events_path}")

    raw_source, records = _read_source(events_path)
    source_sha = _sha256_bytes(raw_source)

    # Font bytes are part of the raster identity.  Production passes D0-frozen
    # hashes so OS/font-package drift is explicit instead of silently changing
    # the model-facing representation.
    actual_font_sha = _sha256_path(FONT_REG)
    actual_font_bold_sha = _sha256_path(FONT_BOLD)
    if expected_font_sha256 and actual_font_sha != expected_font_sha256.lower():
        raise T2IAtlasError(
            "regular font SHA256 drift: "
            f"expected={expected_font_sha256.lower()} got={actual_font_sha}"
        )
    if expected_font_bold_sha256 and actual_font_bold_sha != expected_font_bold_sha256.lower():
        raise T2IAtlasError(
            "bold font SHA256 drift: "
            f"expected={expected_font_bold_sha256.lower()} got={actual_font_bold_sha}"
        )

    registry, template_map = _template_registry(records)
    groups, event_index = _prepare_groups(records, registry, template_map)
    pages = _paginate(groups, event_index)

    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir = out_dir / "t2i_atlas_v3_a"
    staging = Path(
        tempfile.mkdtemp(prefix=".t2i_atlas_v3_a.staging.", dir=str(out_dir))
    )

    try:
        compact_records: list[str] = []
        for rec, index_row in zip(records, event_index):
            raw_rows, displayed_fields = _event_rows(
                index_row["label"], index_row["tmpl_id"], rec
            )
            compact_records.append(
                _json_text(
                    {
                        "label": index_row["label"],
                        "template": index_row["tmpl_id"],
                        "source_line": rec.source_line,
                        "raw_sha256": rec.raw_sha256,
                        "status": str(rec.value.get("status") or ""),
                        "elapsed_ms": rec.value.get("elapsed_ms"),
                        "displayed_fields": displayed_fields,
                        "rows": raw_rows,
                    }
                )
            )
        compact_text = "\n".join(compact_records) + "\n"
        compact_bytes = compact_text.encode("utf-8")
        compact_sha = _sha256_bytes(compact_bytes)
        compact_name = "t2i_log_atlas_v3_a_records.jsonl"
        compact_staging = staging / compact_name
        _atomic_bytes(compact_staging, compact_bytes)

        page_entries: list[dict[str, Any]] = []
        for page_number, page in enumerate(pages, start=1):
            scope = _page_scope(
                page,
                page_number=page_number,
                registry=registry,
                event_index=event_index,
            )
            png = _render_png(
                page,
                page_number=page_number,
                total_pages=len(pages),
                scope=scope,
                source_sha256=source_sha,
                compact_sha256=compact_sha,
            )
            page_name = f"t2i_log_atlas_v3_a_page{page_number}.png"
            page_staging = staging / page_name
            _atomic_bytes(page_staging, png)
            page_entries.append(
                {
                    "page": page_number,
                    "name": page_name,
                    "sha256": _sha256_bytes(png),
                    "bytes": len(png),
                    "width": W,
                    "height": H,
                    "font_px": FONT_SIZE,
                    "rows": len(page),
                    "scope": scope,
                }
            )

        font_sha = actual_font_sha
        font_bold_sha = actual_font_bold_sha
        manifest = {
            "codec": CODEC,
            "profile": PROFILE,
            "source": {
                "path": str(events_path),
                "sha256": source_sha,
                "bytes": len(raw_source),
                "records": len(records),
                "malformed": 0,
            },
            "render": {
                "width": W,
                "height": H,
                "font_px": FONT_SIZE,
                "line_height_px": LINE_H,
                "target_rows_per_page": TARGET_ROWS_PER_PAGE,
                "hard_max_rows_per_page": HARD_MAX_ROWS_PER_PAGE,
                "font_path": str(FONT_REG),
                "font_sha256": font_sha,
                "font_bold_path": str(FONT_BOLD),
                "font_bold_sha256": font_bold_sha,
                "font_identity_enforced": bool(
                    expected_font_sha256 or expected_font_bold_sha256
                ),
                "font_identity_match": (
                    (not expected_font_sha256 or font_sha == expected_font_sha256.lower())
                    and (
                        not expected_font_bold_sha256
                        or font_bold_sha == expected_font_bold_sha256.lower()
                    )
                ),
            },
            "template_count": len(registry),
            "template_registry": registry,
            "event_index_count": len(event_index),
            "event_index": event_index,
            "compact_records": {
                "name": compact_name,
                "sha256": compact_sha,
                "bytes": len(compact_bytes),
                "records": len(records),
            },
            "page_count": len(page_entries),
            "pages": page_entries,
            "publish_protocol": "atomic_directory_v1",
            "integrity": {
                "source_events_indexed_exactly_once": (
                    len(event_index) == len(records)
                    and len({row["label"] for row in event_index}) == len(records)
                    and len({row["source_line"] for row in event_index}) == len(records)
                ),
                "all_event_pages_assigned": all(
                    row["page"] is not None for row in event_index
                ),
                "no_page_exceeds_hard_capacity": all(
                    page["rows"] <= HARD_MAX_ROWS_PER_PAGE for page in page_entries
                ),
            },
        }
        if not all(manifest["integrity"].values()):
            raise T2IAtlasError(
                f"integrity invariant failed: {manifest['integrity']}"
            )

        manifest_name = "t2i_log_atlas_v3_a_manifest.json"
        manifest_staging = staging / manifest_name
        _atomic_text(
            manifest_staging,
            json.dumps(
                manifest, indent=2, sort_keys=True, ensure_ascii=False
            )
            + "\n",
        )

        # Verify every staged artifact against its manifest before publication.
        if _sha256_path(compact_staging) != compact_sha:
            raise T2IAtlasError("compact-record staging hash mismatch")
        for page in page_entries:
            if _sha256_path(staging / page["name"]) != page["sha256"]:
                raise T2IAtlasError(
                    f"page staging hash mismatch: {page['name']}"
                )

        if final_dir.exists():
            if not final_dir.is_dir():
                raise T2IAtlasError(
                    f"atlas publish target exists and is not a directory: {final_dir}"
                )
            if _tree_fingerprints(final_dir) != _tree_fingerprints(staging):
                raise T2IAtlasError(
                    "existing atlas directory differs from deterministic regeneration; "
                    "refusing overwrite"
                )
            published_from_existing = True
        else:
            os.replace(staging, final_dir)
            published_from_existing = False

        manifest_path = final_dir / manifest_name
        compact_path = final_dir / compact_name
        return {
            "status": "GENERATED",
            "codec": CODEC,
            "profile": PROFILE,
            "source_path": str(events_path),
            "source_sha256": source_sha,
            "source_records": len(records),
            "template_count": len(registry),
            "page_count": len(page_entries),
            "page_paths": [
                str(final_dir / page["name"]) for page in page_entries
            ],
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_path(manifest_path),
            "compact_records_path": str(compact_path),
            "compact_records_sha256": compact_sha,
            "font_sha256": font_sha,
            "font_bold_sha256": font_bold_sha,
            "publish_protocol": "atomic_directory_v1",
            "idempotent_existing_match": published_from_existing,
        }
    finally:
        # If publication succeeded staging no longer exists.  If generation or
        # comparison failed, remove only the private staging directory.
        if staging.exists():
            shutil.rmtree(staging)
