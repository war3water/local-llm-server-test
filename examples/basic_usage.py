"""
Basic usage examples for llm_client.

Usage:
    Activate the project environment (`.venv` or `.conda/llm_test`)
    python -m pip install -e .     # install the package (first time only)
    python examples/basic_usage.py
"""

from llm_client import LLMClient
from llm_client.config import DEFAULT_MODEL


def main():
    # 1. Simple one-liner
    llm = LLMClient()
    print("=" * 60)
    print("1) Simple chat")
    print("=" * 60)
    reply = llm.chat("Hello! Introduce yourself in one sentence.")
    print(f"Reply: {reply}\n")

    # 2. Streaming with callback
    print("=" * 60)
    print("2) Streaming chat")
    print("=" * 60)
    tokens: list[str] = []
    reply = llm.chat(
        "What is OpenRouter in 2 sentences?",
        stream=True,
        on_token=tokens.append,
    )
    print(f"Collected {len(tokens)} stream chunks.")
    print(f"Final reply: {reply}\n")

    # 3. Custom system prompt
    print("=" * 60)
    print("3) Custom system prompt")
    print("=" * 60)
    reply = llm.chat(
        "Write a haiku about Python.",
        system="You are a creative poet who writes only haiku.",
    )
    print(f"Reply: {reply}\n")

    # 4. Multi-turn conversation
    print("=" * 60)
    print("4) Multi-turn conversation")
    print("=" * 60)
    reply = llm.chat_messages(
        [
            {"role": "system", "content": "You are a math tutor. Be concise."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "What about 2+2+2?"},
        ]
    )
    print(f"Reply: {reply}\n")

    # 5. Sync concurrent batch
    print("=" * 60)
    print("5) Concurrent batch chat")
    print("=" * 60)
    prompts = [
        "Say hi in one word.",
        "Give one short sentence about Python.",
        "Name one open-source LLM.",
    ]
    replies = llm.batch_chat(
        prompts,
        model=DEFAULT_MODEL,
        max_concurrency=3,
    )
    for idx, text in enumerate(replies, start=1):
        print(f"{idx}. {text}")
    print()

    print("=" * 60)
    print("All examples completed.")


if __name__ == "__main__":
    main()
