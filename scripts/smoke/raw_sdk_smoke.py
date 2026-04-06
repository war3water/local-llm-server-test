"""
Direct OpenRouter smoke test using the raw OpenAI SDK.

This intentionally mirrors low-level usage to debug provider behavior.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        raise RuntimeError("Set OPENROUTER_API_KEY in .env before running this script.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    completion = client.chat.completions.create(
        model="qwen/qwen3-next-80b-a3b-instruct:free",
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
