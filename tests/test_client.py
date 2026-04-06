import asyncio
import time
from types import SimpleNamespace

import pytest

from llm_client import LLMClient


def _make_response(text: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _make_chunk(token: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=token))]
    )


class _AsyncStream:
    def __init__(self, tokens: list[str]):
        self._tokens = iter(tokens)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            token = next(self._tokens)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        await asyncio.sleep(0)
        return _make_chunk(token)


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


def test_retry_then_fallback(monkeypatch):
    class RetryableError(Exception):
        pass

    llm = LLMClient(
        api_key="test-key",
        model="primary",
        fallback_models=["backup"],
        max_retries=2,
        initial_wait=0,
        verbose=False,
    )
    calls: list[str] = []

    def create_fn(*, model, messages, **kwargs):
        calls.append(model)
        if model == "primary":
            raise RetryableError("busy")
        return _make_response("ok from backup")

    llm._client = _SyncClientStub(create_fn)
    monkeypatch.setattr(
        llm, "_is_retryable_exception", lambda exc: isinstance(exc, RetryableError)
    )

    reply = llm.chat("hello")
    assert reply == "ok from backup"
    assert calls == ["primary", "primary", "backup"]


def test_default_request_kwargs_are_merged():
    llm = LLMClient(
        api_key="test-key",
        default_request_kwargs={"temperature": 0.1, "max_tokens": 100},
        verbose=False,
    )
    captured: dict = {}

    def create_fn(*, model, messages, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return _make_response("ok")

    llm._client = _SyncClientStub(create_fn)

    llm.chat("hello", temperature=0.9)
    assert captured["kwargs"]["max_tokens"] == 100
    assert captured["kwargs"]["temperature"] == 0.9


def test_stream_callback_collects_tokens():
    llm = LLMClient(api_key="test-key", verbose=False)

    def create_fn(*, model, messages, stream=False, **kwargs):
        assert stream is True
        return iter([_make_chunk("hello"), _make_chunk(" "), _make_chunk("world")])

    llm._client = _SyncClientStub(create_fn)
    seen: list[str] = []
    text = llm.chat("hi", stream=True, on_token=seen.append, stream_to_stdout=False)
    assert text == "hello world"
    assert seen == ["hello", " ", "world"]


def test_batch_chat_preserves_input_order(monkeypatch):
    llm = LLMClient(api_key="test-key", verbose=False)

    delays = {"first": 0.05, "second": 0.01, "third": 0.02}

    def fake_chat(prompt: str, **kwargs):
        time.sleep(delays[prompt])
        return prompt.upper()

    monkeypatch.setattr(llm, "chat", fake_chat)
    replies = llm.batch_chat(["first", "second", "third"], max_concurrency=3)
    assert replies == ["FIRST", "SECOND", "THIRD"]


def test_batch_chat_can_return_exceptions(monkeypatch):
    llm = LLMClient(api_key="test-key", verbose=False)

    def fake_chat(prompt: str, **kwargs):
        if prompt == "bad":
            raise RuntimeError("boom")
        return prompt

    monkeypatch.setattr(llm, "chat", fake_chat)
    replies = llm.batch_chat(["ok", "bad"], return_exceptions=True)
    assert replies[0] == "ok"
    assert isinstance(replies[1], RuntimeError)


def test_abatch_chat_preserves_input_order(monkeypatch):
    llm = LLMClient(api_key="test-key", verbose=False)
    delays = {"ab": 0.03, "cd": 0.01, "ef": 0.02}

    async def fake_achat(prompt: str, **kwargs):
        await asyncio.sleep(delays[prompt])
        return prompt[::-1]

    monkeypatch.setattr(llm, "achat", fake_achat)
    replies = asyncio.run(llm.abatch_chat(["ab", "cd", "ef"], max_concurrency=2))
    assert replies == ["ba", "dc", "fe"]


def test_achat_stream_callback_collects_tokens():
    llm = LLMClient(api_key="test-key", verbose=False)

    async def create_fn(*, model, messages, stream=False, **kwargs):
        assert stream is True
        return _AsyncStream(["a", "b", "c"])

    llm._aclient = _AsyncClientStub(create_fn)
    seen: list[str] = []

    async def _run():
        return await llm.achat(
            "hi",
            stream=True,
            on_token=seen.append,
            stream_to_stdout=False,
        )

    text = asyncio.run(_run())
    assert text == "abc"
    assert seen == ["a", "b", "c"]


def test_batch_chat_rejects_non_positive_concurrency():
    llm = LLMClient(api_key="test-key", verbose=False)
    with pytest.raises(ValueError, match="max_concurrency"):
        llm.batch_chat(["a"], max_concurrency=0)


def test_abatch_chat_rejects_non_positive_concurrency():
    llm = LLMClient(api_key="test-key", verbose=False)

    async def _run():
        await llm.abatch_chat(["a"], max_concurrency=0)

    with pytest.raises(ValueError, match="max_concurrency"):
        asyncio.run(_run())


def test_sdk_retries_disabled_to_avoid_nested_retry_loops():
    llm = LLMClient(api_key="test-key", verbose=False)
    assert llm._client.max_retries == 0
    assert llm._aclient.max_retries == 0
    llm.close()
    asyncio.run(llm.aclose())


def test_fallback_models_are_deduplicated(monkeypatch):
    class RetryableError(Exception):
        pass

    llm = LLMClient(
        api_key="test-key",
        model="primary",
        fallback_models=["primary", "backup", "backup", "secondary"],
        max_retries=1,
        initial_wait=0,
        verbose=False,
    )
    calls: list[str] = []

    def create_fn(*, model, messages, **kwargs):
        calls.append(model)
        if model == "primary":
            raise RetryableError("busy")
        return _make_response("ok")

    llm._client = _SyncClientStub(create_fn)
    monkeypatch.setattr(
        llm, "_is_retryable_exception", lambda exc: isinstance(exc, RetryableError)
    )

    reply = llm.chat("hello")
    assert reply == "ok"
    assert llm.fallback_models == ["backup", "secondary"]
    assert calls == ["primary", "backup"]
