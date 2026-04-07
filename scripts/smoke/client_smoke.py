"""
Quick smoke test for LLMClient (provider-agnostic).

Usage:
    Activate the project environment (`.venv` or `.conda/llm_test`)
    python scripts/smoke/client_smoke.py

Optional env vars:
    LLM_API_KEY      (fallback: OPENROUTER_API_KEY)
    LLM_BASE_URL     (default: client default base_url)
    LLM_MODEL        (optional per-call model override)
"""

import os
import sys

from llm_client import LLMClient
from llm_client.config import PLACEHOLDER_API_KEYS


def main():
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model_override = os.getenv("LLM_MODEL")

    if not api_key or api_key in PLACEHOLDER_API_KEYS:
        print("Set LLM_API_KEY or OPENROUTER_API_KEY in .env before running this script.")
        sys.exit(1)

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    llm = LLMClient(**kwargs)

    # Non-streaming
    print(f"\n{'=' * 60}")
    print("Non-streaming test")
    print(f"{'=' * 60}")
    reply = llm.chat(
        "Hello! Please introduce yourself in one sentence.",
        model=model_override,
    )
    print(f"Reply: {reply}")

    # Streaming
    print(f"\n{'=' * 60}")
    print("Streaming test")
    print(f"{'=' * 60}")
    print("Reply: ", end="")
    llm.chat(
        "Explain what your API provider endpoint does in 2-3 sentences.",
        stream=True,
        model=model_override,
    )

    print(f"\n{'=' * 60}")
    print("Smoke test completed.")


if __name__ == "__main__":
    main()
