"""
Async usage examples for llm_client.

Usage:
    Activate the project environment (`.venv` or `.conda/llm_test`)
    python -m pip install -e .
    python examples/async_usage.py
"""

import asyncio

from llm_client import LLMClient


async def main() -> None:
    async with LLMClient() as llm:
        print("=" * 60)
        print("1) Async single chat")
        print("=" * 60)
        reply = await llm.achat("Explain asyncio in one sentence.")
        print(f"Reply: {reply}\n")

        print("=" * 60)
        print("2) Async concurrent batch")
        print("=" * 60)
        prompts = [
            "Say hello in one word.",
            "Name one benefit of async programming.",
            "Give one example of an LLM provider.",
        ]
        replies = await llm.abatch_chat(prompts, max_concurrency=3)
        for idx, text in enumerate(replies, start=1):
            print(f"{idx}. {text}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
