"""
Unit tests for VL (Vision-Language) helpers in LLMClient.

All tests are offline — no API key or network access required.
"""

import asyncio
import base64
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_client import LLMClient
from llm_client.config import DEFAULT_VL_MODEL


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def tiny_png(tmp_path: Path) -> Path:
    """Create a minimal valid 1x1 red PNG file."""
    # Minimal 1x1 red PNG (67 bytes)
    png_data = (
        b"\x89PNG\r\n\x1a\n"  # PNG signature
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx"
        b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    file = tmp_path / "test.png"
    file.write_bytes(png_data)
    return file


@pytest.fixture()
def tiny_jpg(tmp_path: Path) -> Path:
    """Create a minimal JPEG file (just enough bytes to have a valid extension)."""
    # Minimal JFIF header
    jpg_data = b"\xff\xd8\xff\xe0" + b"\x00" * 20 + b"\xff\xd9"
    file = tmp_path / "test.jpg"
    file.write_bytes(jpg_data)
    return file


# ── encode_image ─────────────────────────────────────────────────────────────


def test_encode_image_returns_valid_data_url(tiny_png: Path):
    result = LLMClient.encode_image(tiny_png)
    assert result.startswith("data:image/png;base64,")
    # Verify the base64 portion decodes successfully
    b64_part = result.split(",", 1)[1]
    decoded = base64.b64decode(b64_part)
    assert decoded == tiny_png.read_bytes()


def test_encode_image_jpg(tiny_jpg: Path):
    result = LLMClient.encode_image(tiny_jpg)
    assert result.startswith("data:image/jpeg;base64,")


def test_encode_image_raises_on_missing_file():
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        LLMClient.encode_image("/nonexistent/path/image.png")


def test_encode_image_raises_on_unknown_extension(tmp_path: Path):
    unknown = tmp_path / "file.xyz123"
    unknown.write_bytes(b"some data")
    with pytest.raises(ValueError, match="Cannot determine MIME type"):
        LLMClient.encode_image(unknown)


# ── build_vision_content ─────────────────────────────────────────────────────


def test_build_vision_content_single_image(tiny_png: Path):
    content = LLMClient.build_vision_content("Describe this.", tiny_png)
    assert isinstance(content, list)
    assert len(content) == 2  # 1 image + 1 text

    image_part = content[0]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert image_part["image_url"]["detail"] == "auto"

    text_part = content[1]
    assert text_part["type"] == "text"
    assert text_part["text"] == "Describe this."


def test_build_vision_content_multiple_images(tiny_png: Path, tiny_jpg: Path):
    content = LLMClient.build_vision_content(
        "Compare these images.", [tiny_png, tiny_jpg]
    )
    assert len(content) == 3  # 2 images + 1 text
    assert content[0]["type"] == "image_url"
    assert content[1]["type"] == "image_url"
    assert content[2]["type"] == "text"
    assert content[2]["text"] == "Compare these images."


def test_build_vision_content_with_url():
    url = "https://example.com/image.png"
    content = LLMClient.build_vision_content("Describe this.", url)
    assert len(content) == 2
    assert content[0]["image_url"]["url"] == url  # URL passed through unchanged


def test_build_vision_content_detail_parameter(tiny_png: Path):
    content = LLMClient.build_vision_content(
        "Analyze.", tiny_png, detail="high"
    )
    assert content[0]["image_url"]["detail"] == "high"


# ── chat_vision / achat_vision ───────────────────────────────────────────────


def _make_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


class _SyncClientStub:
    def __init__(self, create_fn):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create_fn))

    def close(self):
        return None


class _AsyncClientStub:
    def __init__(self, create_fn):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create_fn))

    async def close(self):
        return None


def test_chat_vision_uses_vl_model_by_default(tiny_png: Path):
    """Verify that chat_vision defaults to DEFAULT_VL_MODEL."""
    llm = LLMClient(api_key="test-key", verbose=False)
    captured: dict = {}

    def create_fn(*, model, messages, **kwargs):
        captured["model"] = model
        captured["messages"] = messages
        return _make_response("Table detected.")

    llm._client = _SyncClientStub(create_fn)

    reply = llm.chat_vision("Describe this.", tiny_png)
    assert reply == "Table detected."
    assert captured["model"] == DEFAULT_VL_MODEL


def test_chat_vision_passes_vision_content(tiny_png: Path):
    """Verify that the message sent to the API has vision content parts."""
    llm = LLMClient(api_key="test-key", verbose=False)
    captured: dict = {}

    def create_fn(*, model, messages, **kwargs):
        captured["messages"] = messages
        return _make_response("ok")

    llm._client = _SyncClientStub(create_fn)

    llm.chat_vision("Describe this image.", tiny_png)

    user_msg = captured["messages"][1]
    assert user_msg["role"] == "user"
    assert isinstance(user_msg["content"], list)

    # Should have image_url part + text part
    types = [part["type"] for part in user_msg["content"]]
    assert types == ["image_url", "text"]


def test_chat_vision_respects_model_override(tiny_png: Path):
    llm = LLMClient(api_key="test-key", verbose=False)
    captured: dict = {}

    def create_fn(*, model, messages, **kwargs):
        captured["model"] = model
        return _make_response("ok")

    llm._client = _SyncClientStub(create_fn)

    llm.chat_vision("Describe.", tiny_png, model="custom/vl-model")
    assert captured["model"] == "custom/vl-model"


def test_achat_vision_uses_vl_model(tiny_png: Path):
    """Verify async vision chat defaults to VL model."""
    llm = LLMClient(api_key="test-key", verbose=False)
    captured: dict = {}

    async def create_fn(*, model, messages, **kwargs):
        captured["model"] = model
        return _make_response("Async table detected.")

    llm._aclient = _AsyncClientStub(create_fn)

    async def _run():
        return await llm.achat_vision("Describe.", tiny_png)

    reply = asyncio.run(_run())
    assert reply == "Async table detected."
    assert captured["model"] == DEFAULT_VL_MODEL


# ── _validate_messages accepts vision format ─────────────────────────────────


def test_validate_messages_accepts_list_content():
    """Vision messages have content as a list — validation should pass."""
    llm = LLMClient(api_key="test-key", verbose=False)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "text", "text": "Describe this."},
            ],
        },
    ]
    # Should not raise
    llm._validate_messages(messages)


def test_validate_messages_rejects_invalid_content_type():
    """content must be str or list, not int."""
    llm = LLMClient(api_key="test-key", verbose=False)
    messages = [{"role": "user", "content": 42}]
    with pytest.raises(TypeError, match="must be str or list"):
        llm._validate_messages(messages)
