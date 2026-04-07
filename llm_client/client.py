"""
LLMClient - reusable OpenAI-compatible client with sync + async interfaces.

Features:
    - Single-turn and multi-turn chat
    - Streaming and non-streaming responses
    - Retry with exponential backoff for transient failures
    - Model fallback chain
    - Concurrent batch helpers (threaded and asyncio)
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import mimetypes
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Awaitable, Callable, Sequence

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAI,
    RateLimitError,
)

from .config import (
    DEFAULT_API_KEY,
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_BACKOFF_JITTER,
    DEFAULT_BASE_URL,
    DEFAULT_FALLBACKS,
    DEFAULT_INITIAL_WAIT,
    DEFAULT_MAX_BACKOFF,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    DEFAULT_VL_FALLBACKS,
    DEFAULT_VL_MODEL,
    PLACEHOLDER_API_KEYS,
)

TokenCallback = Callable[[str], None]
AsyncTokenCallback = Callable[[str], Awaitable[None] | None]


class LLMClient:
    """High-level wrapper around OpenAI-compatible chat-completions APIs."""

    _RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        fallback_models: list[str] | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float | None = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_wait: float = DEFAULT_INITIAL_WAIT,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
        backoff_jitter: float = DEFAULT_BACKOFF_JITTER,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        system_prompt: str = "You are a helpful assistant.",
        site_url: str | None = "http://localhost",
        site_name: str | None = "LLM Client",
        default_request_kwargs: dict[str, Any] | None = None,
        stream_to_stdout: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            api_key: Provider API key; falls back to LLM_API_KEY/OPENROUTER_API_KEY.
            model: Primary model ID.
            fallback_models: Ordered fallback model IDs.
            base_url: API base URL for OpenAI-compatible providers.
            timeout: Per-request timeout in seconds.
            max_retries: Retry attempts per model for retryable failures.
            initial_wait: Initial retry wait in seconds.
            backoff_factor: Exponential backoff factor.
            max_backoff: Maximum retry wait.
            backoff_jitter: Random jitter ratio applied to wait time.
            max_concurrency: Default max parallel requests for batch helpers.
            system_prompt: Default system prompt if one is not provided.
            site_url: Optional HTTP-Referer header for provider analytics.
            site_name: Optional X-Title header for provider analytics.
            default_request_kwargs: Default kwargs merged into every request.
            stream_to_stdout: Print stream tokens to stdout by default.
            verbose: Print retry/fallback messages to stderr.
        """
        resolved_key = api_key or DEFAULT_API_KEY
        if not resolved_key or resolved_key in PLACEHOLDER_API_KEYS:
            raise ValueError(
                "No API key provided. Set LLM_API_KEY or OPENROUTER_API_KEY in .env "
                "or pass api_key= to LLMClient()."
            )
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if initial_wait < 0:
            raise ValueError("initial_wait must be >= 0")
        if backoff_factor < 1:
            raise ValueError("backoff_factor must be >= 1")
        if max_backoff <= 0:
            raise ValueError("max_backoff must be > 0")
        if backoff_jitter < 0:
            raise ValueError("backoff_jitter must be >= 0")

        self.model = model
        resolved_fallbacks = fallback_models if fallback_models is not None else list(DEFAULT_FALLBACKS)
        self.fallback_models = self._dedupe_fallbacks(
            primary_model=model,
            fallback_models=resolved_fallbacks,
        )
        self.max_retries = max_retries
        self.initial_wait = initial_wait
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.backoff_jitter = backoff_jitter
        self.max_concurrency = max_concurrency
        self.system_prompt = system_prompt
        self.default_request_kwargs = dict(default_request_kwargs or {})
        self.stream_to_stdout = stream_to_stdout
        self.verbose = verbose

        default_headers: dict[str, str] = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if site_name:
            default_headers["X-Title"] = site_name

        client_kwargs: dict[str, Any] = {
            "base_url": base_url,
            "api_key": resolved_key,
            # Disable SDK-level retries to avoid nested retries with this wrapper.
            "max_retries": 0,
        }
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        if default_headers:
            client_kwargs["default_headers"] = default_headers

        self._client = OpenAI(**client_kwargs)
        self._aclient = AsyncOpenAI(**client_kwargs)

    # Public API

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        stream: bool = False,
        on_token: TokenCallback | None = None,
        stream_to_stdout: bool | None = None,
        **kwargs: Any,
    ) -> str:
        """Single-turn sync chat helper."""
        messages = self._build_messages(prompt, system)
        return self.chat_messages(
            messages,
            model=model,
            stream=stream,
            on_token=on_token,
            stream_to_stdout=stream_to_stdout,
            **kwargs,
        )

    async def achat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        stream: bool = False,
        on_token: AsyncTokenCallback | None = None,
        stream_to_stdout: bool | None = None,
        **kwargs: Any,
    ) -> str:
        """Single-turn async chat helper."""
        messages = self._build_messages(prompt, system)
        return await self.achat_messages(
            messages,
            model=model,
            stream=stream,
            on_token=on_token,
            stream_to_stdout=stream_to_stdout,
            **kwargs,
        )

    def chat_vision(
        self,
        prompt: str,
        images: str | Path | list[str | Path],
        *,
        system: str | None = None,
        model: str | None = None,
        detail: str = "auto",
        stream: bool = False,
        on_token: TokenCallback | None = None,
        stream_to_stdout: bool | None = None,
        **kwargs: Any,
    ) -> str:
        """Single-turn sync vision chat — send images + text to a VL model.

        Args:
            prompt: Text prompt describing what to do with the image(s).
            images: One or more file paths (or URLs) to include.
            model: Model override; defaults to ``DEFAULT_VL_MODEL``.
            detail: Image detail level ("auto", "low", "high").
        """
        resolved_model = model or DEFAULT_VL_MODEL
        messages = self._build_vision_messages(prompt, images, system, detail)
        return self.chat_messages(
            messages,
            model=resolved_model,
            stream=stream,
            on_token=on_token,
            stream_to_stdout=stream_to_stdout,
            **kwargs,
        )

    async def achat_vision(
        self,
        prompt: str,
        images: str | Path | list[str | Path],
        *,
        system: str | None = None,
        model: str | None = None,
        detail: str = "auto",
        stream: bool = False,
        on_token: AsyncTokenCallback | None = None,
        stream_to_stdout: bool | None = None,
        **kwargs: Any,
    ) -> str:
        """Single-turn async vision chat — send images + text to a VL model."""
        resolved_model = model or DEFAULT_VL_MODEL
        messages = self._build_vision_messages(prompt, images, system, detail)
        return await self.achat_messages(
            messages,
            model=resolved_model,
            stream=stream,
            on_token=on_token,
            stream_to_stdout=stream_to_stdout,
            **kwargs,
        )

    def chat_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        stream: bool = False,
        on_token: TokenCallback | None = None,
        stream_to_stdout: bool | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Send sync chat completion with a full message list.

        `kwargs` are passed to `chat.completions.create()`.
        """
        self._validate_messages(messages)
        request_kwargs = self._merge_request_kwargs(kwargs)

        if stream:
            return self._stream(
                messages,
                model=model,
                on_token=on_token,
                stream_to_stdout=stream_to_stdout,
                **request_kwargs,
            )
        return self._complete(messages, model=model, **request_kwargs)

    async def achat_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        stream: bool = False,
        on_token: AsyncTokenCallback | None = None,
        stream_to_stdout: bool | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Send async chat completion with a full message list.

        `kwargs` are passed to `chat.completions.create()`.
        """
        self._validate_messages(messages)
        request_kwargs = self._merge_request_kwargs(kwargs)

        if stream:
            return await self._astream(
                messages,
                model=model,
                on_token=on_token,
                stream_to_stdout=stream_to_stdout,
                **request_kwargs,
            )
        return await self._acomplete(messages, model=model, **request_kwargs)

    def create(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return raw sync chat-completion response object."""
        self._validate_messages(messages)
        request_kwargs = self._merge_request_kwargs(kwargs)
        return self._call_with_retry(
            lambda candidate_model: self._client.chat.completions.create(
                model=candidate_model,
                messages=messages,
                **request_kwargs,
            ),
            override_model=model,
        )

    async def acreate(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return raw async chat-completion response object."""
        self._validate_messages(messages)
        request_kwargs = self._merge_request_kwargs(kwargs)
        return await self._acall_with_retry(
            lambda candidate_model: self._aclient.chat.completions.create(
                model=candidate_model,
                messages=messages,
                **request_kwargs,
            ),
            override_model=model,
        )

    def batch_chat(
        self,
        prompts: Sequence[str],
        *,
        system: str | None = None,
        model: str | None = None,
        max_concurrency: int | None = None,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[str] | list[str | Exception]:
        """
        Concurrent sync chat calls over prompts while preserving input order.
        """
        if kwargs.get("stream"):
            raise ValueError("batch_chat does not support stream=True")

        concurrency = self.max_concurrency if max_concurrency is None else max_concurrency
        if concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

        results: list[str | Exception | None] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_index = {
                executor.submit(
                    self.chat,
                    prompt,
                    system=system,
                    model=model,
                    stream=False,
                    **kwargs,
                ): index
                for index, prompt in enumerate(prompts)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    if return_exceptions:
                        results[index] = exc
                    else:
                        raise

        finalized: list[str | Exception] = []
        for result in results:
            if result is None:
                raise RuntimeError("batch_chat produced an incomplete result set")
            finalized.append(result)
        return finalized

    async def abatch_chat(
        self,
        prompts: Sequence[str],
        *,
        system: str | None = None,
        model: str | None = None,
        max_concurrency: int | None = None,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[str] | list[str | Exception]:
        """
        Concurrent async chat calls over prompts while preserving input order.
        """
        if kwargs.get("stream"):
            raise ValueError("abatch_chat does not support stream=True")

        concurrency = self.max_concurrency if max_concurrency is None else max_concurrency
        if concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

        semaphore = asyncio.Semaphore(concurrency)

        async def _run_one(prompt: str) -> str:
            async with semaphore:
                return await self.achat(
                    prompt,
                    system=system,
                    model=model,
                    stream=False,
                    **kwargs,
                )

        tasks = [asyncio.create_task(_run_one(prompt)) for prompt in prompts]
        responses = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        return list(responses)

    def close(self) -> None:
        """Close sync transport."""
        self._client.close()

    async def aclose(self) -> None:
        """Close async transport."""
        await self._aclient.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    # Internal helpers

    def _complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        response = self.create(messages, model=model, **kwargs)
        return self._extract_response_text(response)

    async def _acomplete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        response = await self.acreate(messages, model=model, **kwargs)
        return self._extract_response_text(response)

    def _stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        on_token: TokenCallback | None = None,
        stream_to_stdout: bool | None = None,
        **kwargs: Any,
    ) -> str:
        stream = self._call_with_retry(
            lambda candidate_model: self._client.chat.completions.create(
                model=candidate_model,
                messages=messages,
                stream=True,
                **kwargs,
            ),
            override_model=model,
        )
        should_print = self._resolve_stream_to_stdout(stream_to_stdout)
        collected: list[str] = []
        for chunk in stream:
            token = self._extract_delta_text(chunk)
            if not token:
                continue
            collected.append(token)
            if on_token is not None:
                on_token(token)
            if should_print:
                print(token, end="", flush=True)
        if should_print:
            print()
        return "".join(collected)

    async def _astream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        on_token: AsyncTokenCallback | None = None,
        stream_to_stdout: bool | None = None,
        **kwargs: Any,
    ) -> str:
        stream = await self._acall_with_retry(
            lambda candidate_model: self._aclient.chat.completions.create(
                model=candidate_model,
                messages=messages,
                stream=True,
                **kwargs,
            ),
            override_model=model,
        )
        should_print = self._resolve_stream_to_stdout(stream_to_stdout)
        collected: list[str] = []
        async for chunk in stream:
            token = self._extract_delta_text(chunk)
            if not token:
                continue
            collected.append(token)
            if on_token is not None:
                maybe_awaitable = on_token(token)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
            if should_print:
                print(token, end="", flush=True)
        if should_print:
            print()
        return "".join(collected)

    def _call_with_retry(
        self,
        make_call: Callable[[str], Any],
        *,
        override_model: str | None = None,
    ) -> Any:
        models = self._resolve_models(override_model)
        last_error: Exception | None = None

        for model in models:
            for attempt in range(1, self.max_retries + 1):
                try:
                    return make_call(model)
                except Exception as exc:
                    if not self._is_retryable_exception(exc):
                        raise
                    last_error = exc
                    if attempt >= self.max_retries:
                        if self.verbose:
                            print(
                                f"{model} exhausted retries ({self.max_retries}).",
                                file=sys.stderr,
                            )
                        break
                    wait = self._retry_delay(attempt)
                    if self.verbose:
                        print(
                            f"{self._error_label(exc)} on {model}. "
                            f"Retry {attempt}/{self.max_retries} in {wait:.2f}s.",
                            file=sys.stderr,
                        )
                    time.sleep(wait)

        raise RuntimeError("All models failed after retry/fallback.") from last_error

    async def _acall_with_retry(
        self,
        make_call: Callable[[str], Awaitable[Any]],
        *,
        override_model: str | None = None,
    ) -> Any:
        models = self._resolve_models(override_model)
        last_error: Exception | None = None

        for model in models:
            for attempt in range(1, self.max_retries + 1):
                try:
                    return await make_call(model)
                except Exception as exc:
                    if not self._is_retryable_exception(exc):
                        raise
                    last_error = exc
                    if attempt >= self.max_retries:
                        if self.verbose:
                            print(
                                f"{model} exhausted retries ({self.max_retries}).",
                                file=sys.stderr,
                            )
                        break
                    wait = self._retry_delay(attempt)
                    if self.verbose:
                        print(
                            f"{self._error_label(exc)} on {model}. "
                            f"Retry {attempt}/{self.max_retries} in {wait:.2f}s.",
                            file=sys.stderr,
                        )
                    await asyncio.sleep(wait)

        raise RuntimeError("All models failed after retry/fallback.") from last_error

    def _resolve_models(self, override_model: str | None) -> list[str]:
        if override_model:
            return [override_model]
        return [self.model, *self.fallback_models]

    @staticmethod
    def _dedupe_fallbacks(primary_model: str, fallback_models: Sequence[str]) -> list[str]:
        """Remove duplicate fallback models and exclude the primary model."""
        deduped: list[str] = []
        seen = {primary_model}
        for candidate in fallback_models:
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)
        return deduped

    def _retry_delay(self, attempt: int) -> float:
        wait = self.initial_wait * (self.backoff_factor ** (attempt - 1))
        wait = min(wait, self.max_backoff)
        if self.backoff_jitter > 0:
            jitter = random.uniform(-self.backoff_jitter, self.backoff_jitter)
            wait = wait * (1 + jitter)
        return max(wait, 0.0)

    def _is_retryable_exception(self, exc: Exception) -> bool:
        if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
            return True
        if isinstance(exc, APIStatusError):
            return getattr(exc, "status_code", None) in self._RETRYABLE_STATUS_CODES
        return False

    def _error_label(self, exc: Exception) -> str:
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            return exc.__class__.__name__
        return f"{exc.__class__.__name__}({status_code})"

    def _merge_request_kwargs(self, call_kwargs: dict[str, Any]) -> dict[str, Any]:
        merged = dict(self.default_request_kwargs)
        merged.update(call_kwargs)
        return merged

    def _build_messages(self, prompt: str, system: str | None) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": system or self.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def _build_vision_messages(
        self,
        prompt: str,
        images: str | Path | list[str | Path],
        system: str | None,
        detail: str = "auto",
    ) -> list[dict[str, Any]]:
        """Build a message list with vision content parts."""
        content = self.build_vision_content(prompt, images, detail=detail)
        return [
            {"role": "system", "content": system or self.system_prompt},
            {"role": "user", "content": content},
        ]

    # Vision helpers

    @staticmethod
    def encode_image(image_path: str | Path) -> str:
        """Encode a local image file to a base64 data-URL.

        Args:
            image_path: Path to an image file (png, jpg, gif, webp, etc.).

        Returns:
            A ``data:<mime>;base64,...`` string suitable for the OpenAI vision API.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the MIME type cannot be determined.
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            # Fallback for common image extensions
            ext_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
                ".svg": "image/svg+xml",
            }
            mime_type = ext_map.get(path.suffix.lower())
        if mime_type is None:
            raise ValueError(
                f"Cannot determine MIME type for: {path.name}. "
                f"Supported: png, jpg, jpeg, gif, webp, bmp, svg."
            )
        raw = path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime_type};base64,{b64}"

    @staticmethod
    def build_vision_content(
        prompt: str,
        images: str | Path | list[str | Path],
        *,
        detail: str = "auto",
    ) -> list[dict[str, Any]]:
        """Build a ``content`` array with image and text parts.

        Args:
            prompt: Text prompt.
            images: Single image path/URL or list of image paths/URLs.
                    Local file paths are base64-encoded automatically.
                    HTTP(S) URLs are passed through as-is.
            detail: Image detail level ("auto", "low", "high").

        Returns:
            A list of content-part dicts ready for the ``messages`` field.
        """
        if isinstance(images, (str, Path)):
            images = [images]

        content: list[dict[str, Any]] = []
        for img in images:
            img_str = str(img)
            if img_str.startswith(("http://", "https://")):
                url = img_str
            else:
                url = LLMClient.encode_image(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": url, "detail": detail},
            })
        content.append({"type": "text", "text": prompt})
        return content

    def _validate_messages(self, messages: list[dict[str, Any]]) -> None:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise TypeError(f"messages[{index}] must be a dict")
            if "role" not in message:
                raise ValueError(f"messages[{index}] missing 'role'")
            if "content" not in message:
                raise ValueError(f"messages[{index}] missing 'content'")
            content = message["content"]
            if not isinstance(content, (str, list)):
                raise TypeError(
                    f"messages[{index}]['content'] must be str or list, "
                    f"got {type(content).__name__}"
                )

    def _resolve_stream_to_stdout(self, stream_to_stdout: bool | None) -> bool:
        if stream_to_stdout is None:
            return self.stream_to_stdout
        return stream_to_stdout

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        content = getattr(message, "content", None) if message is not None else None
        if content is None and isinstance(message, dict):
            content = message.get("content")
        return LLMClient._content_to_text(content)

    @staticmethod
    def _extract_delta_text(chunk: Any) -> str:
        choices = getattr(chunk, "choices", None)
        if not choices:
            return ""
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None and isinstance(choice, dict):
            delta = choice.get("delta")
        content = getattr(delta, "content", None) if delta is not None else None
        if content is None and isinstance(delta, dict):
            content = delta.get("content")
        return LLMClient._content_to_text(content)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "".join(chunks)
        return str(content)
