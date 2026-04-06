"""
llm_client — Reusable OpenRouter / OpenAI-compatible chat client.

Quick start::

    from llm_client import LLMClient

    llm = LLMClient()  # reads LLM_API_KEY / OPENROUTER_API_KEY from .env
    reply = llm.chat("Hello!")
"""

from .client import LLMClient
from .config import (
    DEFAULT_FALLBACKS,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    DEFAULT_VL_MODEL,
)

__all__ = [
    "LLMClient",
    "DEFAULT_MODEL",
    "DEFAULT_VL_MODEL",
    "DEFAULT_FALLBACKS",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_MAX_CONCURRENCY",
]
