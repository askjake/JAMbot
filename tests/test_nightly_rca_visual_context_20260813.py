import base64
import hashlib
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from app.agent.nightly_rca_visual_context import (
    FROZEN_FONT_BOLD_SHA256,
    FROZEN_FONT_SHA256,
    NightlyRCAVisualContextError,
    QUALIFIED_MAX_IMAGE_RES,
    SelectiveVisualSettings,
    _qualified_jpeg_from_candidate_png,
    build_nightly_rca_visual_content,
)
from app.message.schemas import InputUserMessage
from apps.nightly_rca import t2i_atlas as base


def _write_events(path: Path, count: int = 5) -> None:
    rows = []
    for index in range(1, count + 1):
        rows.append(
            {
                "server": "s3_stb_logs",
                "tool": "search_logs",
                "arguments": {"receiver_id": f"R{index:010d}"},
                "status": "OK",
                "elapsed_ms": 200 + index,
                "response": {
                    "receiver_id": f"R{index:010d}",
                    "result": f"RESULT_{index}",
                    "ok": True,
                },
            }
        )
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _visual_settings(runs_root: Path, *, enabled: bool = True) -> SelectiveVisualSettings:
    return SelectiveVisualSettings(
        enabled=enabled,
        runs_root=runs_root,
        max_events=5,
        font_sha256=base._sha256_path(base.FONT_REG),
        font_bold_sha256=base._sha256_path(base.FONT_BOLD),
        allowed_emails=("person@example.com",),
    )


@pytest.mark.asyncio
async def test_qualified_preprocess_matches_frozen_resize_plus_jpeg85(tmp_path: Path):
    source = tmp_path / "candidate.png"
    image = Image.new("RGB", (2048, 2048), "white")
    for x in range(0, 2048, 64):
        for y in range(0, 2048, 64):
            if (x // 64 + y // 64) % 2:
                for xx in range(x, min(x + 32, 2048)):
                    image.putpixel((xx, y), (10, 20, 30))
    image.save(source, "PNG", optimize=False)

    actual, meta = await _qualified_jpeg_from_candidate_png(
        source,
        max_resolution=QUALIFIED_MAX_IMAGE_RES,
    )

    with Image.open(source) as original:
        resized = original.resize((1092, 1092), Image.Resampling.LANCZOS)
        expected = io.BytesIO()
        resized.convert("RGB").save(expected, "JPEG", quality=85)

    assert actual.getvalue() == expected.getvalue()
    assert meta["width"] == 1092
    assert meta["height"] == 1092
    assert meta["effective_mime"] == "image/jpeg"


@pytest.mark.asyncio
async def test_build_context_requires_explicit_enable(tmp_path: Path):
    runs = tmp_path / "runs"
    run = runs / "20260813T120000Z"
    run.mkdir(parents=True)
    _write_events(run / "events.jsonl")

    with pytest.raises(NightlyRCAVisualContextError, match="disabled"):
        await build_nightly_rca_visual_content(
            {"run_id": run.name, "mode": "events", "event_labels": ["L0001"]},
            request_id="disabled",
            requester_email="person@example.com",
            max_resolution=QUALIFIED_MAX_IMAGE_RES,
            visual_settings=_visual_settings(runs, enabled=False),
        )


@pytest.mark.asyncio
async def test_build_context_one_model_image_no_storage_upload(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs"
    run = runs / "20260813T120000Z"
    run.mkdir(parents=True)
    _write_events(run / "events.jsonl")

    async def forbidden_upload(*args, **kwargs):
        raise AssertionError("server-generated selective evidence must not upload")

    monkeypatch.setattr(
        "app.attachment.utils.storage_upload_to_prefix",
        forbidden_upload,
    )

    blocks, provenance = await build_nightly_rca_visual_content(
        {
            "run_id": run.name,
            "mode": "events",
            "event_labels": ["L0001", "L0003", "L0005"],
        },
        request_id="chat:checkpoint",
        requester_email="person@example.com",
        max_resolution=QUALIFIED_MAX_IMAGE_RES,
        visual_settings=_visual_settings(runs),
    )

    assert [block["type"] for block in blocks] == ["text", "image"]
    image_block = blocks[1]
    assert image_block["mime_type"] == "image/jpeg"
    decoded = base64.b64decode(image_block["data"])
    assert decoded.startswith(b"\xff\xd8\xff")
    with Image.open(io.BytesIO(decoded)) as image:
        assert image.size == (1092, 1092)
    assert provenance["event_labels"] == ["L0001", "L0003", "L0005"]
    assert provenance["attachment_persisted"] is False
    assert provenance["model_image_count"] == 1


@pytest.mark.asyncio
async def test_request_validation_rejects_path_or_mode_abuse(tmp_path: Path):
    runs = tmp_path / "runs"
    runs.mkdir()
    settings = _visual_settings(runs)
    for request in (
        {"run_id": "../etc", "mode": "events", "event_labels": ["L0001"]},
        {"run_id": "20260813T120000Z", "mode": "global", "event_labels": ["L0001"]},
        {"run_id": "20260813T120000Z", "mode": "events", "event_labels": []},
    ):
        with pytest.raises(NightlyRCAVisualContextError):
            await build_nightly_rca_visual_content(
                request,
                request_id="bad",
                requester_email="person@example.com",
                max_resolution=QUALIFIED_MAX_IMAGE_RES,
                visual_settings=settings,
            )


def test_input_message_schema_carries_explicit_trusted_context():
    msg = InputUserMessage(
        content="inspect these events",
        nightly_rca_visual_context={
            "run_id": "20260813T120000Z",
            "mode": "events",
            "event_labels": ["L0001", "L0003"],
        },
    )
    assert msg.nightly_rca_visual_context is not None
    assert msg.nightly_rca_visual_context.event_labels == ["L0001", "L0003"]


@pytest.mark.asyncio
async def test_agent_orders_server_visual_before_attachments_and_text(monkeypatch):
    import app.agent.service as agent_module

    server_blocks = [
        {"type": "text", "text": "server image"},
        {"type": "image", "source_type": "base64", "mime_type": "image/jpeg", "data": "AA=="},
    ]

    async def fake_build(*args, **kwargs):
        return server_blocks, {"schema": "test", "model_image_count": 1}

    monkeypatch.setattr(agent_module, "build_nightly_rca_visual_content", fake_build)

    @asynccontextmanager
    async def fake_db_ctx():
        yield object()

    monkeypatch.setattr(agent_module, "get_db_session_ctxmgr", fake_db_ctx)

    class FakeAttachmentService:
        async def download_attachment_internal(self, db, attachment_ids, email, vault_key):
            image = io.BytesIO(b"\x89PNG\r\n\x1a\n")
            status = SimpleNamespace(media_type="image/png", filename="user.png")
            return {attachment_ids[0]: (status, image)}

    class FakeState:
        values = {"messages": []}

    class FakeGraph:
        def __init__(self):
            self.last_input = None

        async def aget_state(self, config):
            return FakeState()

        def astream(self, *, input, config, stream_mode):
            self.last_input = input

            async def _iter():
                if False:
                    yield None

            return _iter()

    service = agent_module.AgentService.__new__(agent_module.AgentService)
    service.graph = FakeGraph()
    service.attmnt_service = FakeAttachmentService()

    await service.process_new_user_message(
        "person@example.com",
        "chat-id",
        "human text",
        ["attachment-id"],
        nightly_rca_visual_context={
            "run_id": "20260813T120000Z",
            "mode": "events",
            "event_labels": ["L0001"],
        },
    )

    content = service.graph.last_input["messages"][0].content
    assert content[0:2] == server_blocks
    assert content[2]["type"] == "text" and "User uploaded an image" in content[2]["text"]
    assert content[3]["type"] == "image"
    assert content[-1] == {"type": "text", "text": "human text"}

@pytest.mark.asyncio
async def test_build_context_requires_allowlisted_requester(tmp_path: Path):
    runs = tmp_path / "runs"
    run = runs / "20260813T120000Z"
    run.mkdir(parents=True)
    _write_events(run / "events.jsonl")

    with pytest.raises(NightlyRCAVisualContextError, match="not authorized"):
        await build_nightly_rca_visual_content(
            {"run_id": run.name, "mode": "events", "event_labels": ["L0001"]},
            request_id="unauthorized",
            requester_email="other@example.com",
            max_resolution=QUALIFIED_MAX_IMAGE_RES,
            visual_settings=_visual_settings(runs),
        )
