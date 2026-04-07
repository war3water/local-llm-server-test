"""
Direct OpenRouter smoke test using the raw OpenAI SDK.

This intentionally mirrors low-level usage to debug provider behavior.

Usage:
    Activate the project environment (`.venv` or `.conda/llm_test`)
    python scripts/smoke/raw_sdk_smoke.py
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from llm_client.config import DEFAULT_MODEL, PLACEHOLDER_API_KEYS

load_dotenv()


def main() -> None:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key or api_key in PLACEHOLDER_API_KEYS:
        print("Set LLM_API_KEY or OPENROUTER_API_KEY in .env before running this script.")
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    completion = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "user",
                "content": "What is the meaning of life?",
            }
        ],
    )
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
