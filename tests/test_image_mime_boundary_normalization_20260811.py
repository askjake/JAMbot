from __future__ import annotations

import base64
import io

from PIL import Image

from app.agent.utils import (
    _detect_image_mime_from_bytes,
    _normalize_declared_image_mime,
    image_content,
)


def _image_bytes(fmt: str) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (4, 4), (12, 34, 56)).save(out, format=fmt)
    return out.getvalue()


def _image_block(result: list[dict]) -> dict:
    blocks = [x for x in result if isinstance(x, dict) and x.get("type") == "image"]
    assert len(blocks) == 1
    return blocks[0]


def test_detector_recognizes_png_jpeg_gif_and_webp_signatures():
    assert _detect_image_mime_from_bytes(b"\x89PNG\r\n\x1a\nrest") == "image/png"
    assert _detect_image_mime_from_bytes(b"\xff\xd8\xffrest") == "image/jpeg"
    assert _detect_image_mime_from_bytes(b"GIF87arest") == "image/gif"
    assert _detect_image_mime_from_bytes(b"GIF89arest") == "image/gif"
    assert (
        _detect_image_mime_from_bytes(b"RIFF\x00\x00\x00\x00WEBPrest")
        == "image/webp"
    )
    assert _detect_image_mime_from_bytes(b"not-an-image") is None


def test_normalizes_jpg_alias_only():
    assert _normalize_declared_image_mime("image/jpg") == "image/jpeg"
    assert _normalize_declared_image_mime(" IMAGE/PNG ") == "image/png"
    assert _normalize_declared_image_mime("image/jpeg") == "image/jpeg"


def test_matching_png_remains_png_and_bytes_are_preserved():
    source = _image_bytes("PNG")
    block = _image_block(image_content("probe.png", "image/png", io.BytesIO(source)))
    assert block["mime_type"] == "image/png"
    assert base64.b64decode(block["data"]) == source


def test_matching_jpeg_remains_jpeg_and_bytes_are_preserved():
    source = _image_bytes("JPEG")
    block = _image_block(image_content("probe.jpg", "image/jpeg", io.BytesIO(source)))
    assert block["mime_type"] == "image/jpeg"
    assert base64.b64decode(block["data"]) == source


def test_png_declared_but_jpeg_bytes_normalizes_model_boundary_to_jpeg():
    source = _image_bytes("JPEG")
    block = _image_block(
        image_content("mislabeled.png", "image/png", io.BytesIO(source))
    )
    decoded = base64.b64decode(block["data"])
    assert block["mime_type"] == "image/jpeg"
    assert decoded == source
    assert decoded.startswith(b"\xff\xd8\xff")


def test_jpeg_declared_but_png_bytes_normalizes_model_boundary_to_png():
    source = _image_bytes("PNG")
    block = _image_block(
        image_content("mislabeled.jpg", "image/jpeg", io.BytesIO(source))
    )
    decoded = base64.b64decode(block["data"])
    assert block["mime_type"] == "image/png"
    assert decoded == source
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")


def test_unknown_bytes_preserve_prior_declared_mime_behavior():
    source = b"opaque-unknown-image-payload"
    block = _image_block(
        image_content("unknown.png", "image/png", io.BytesIO(source))
    )
    assert block["mime_type"] == "image/png"
    assert base64.b64decode(block["data"]) == source
