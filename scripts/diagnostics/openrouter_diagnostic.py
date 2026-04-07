"""
Quick diagnostic: test OpenRouter API connectivity with multiple free models.
Falls back through several models to find one that is not rate-limited.
"""

import os
import sys
import time

from dotenv import load_dotenv
from llm_client.config import PLACEHOLDER_API_KEYS
from llm_client import DEFAULT_FALLBACKS, DEFAULT_MODEL

try:
    import requests
except ImportError as exc:
    raise RuntimeError(
        "requests is required for diagnostics. Install with: pip install -e .[diagnostics]"
    ) from exc

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"


def main() -> None:
    if not API_KEY or API_KEY in PLACEHOLDER_API_KEYS:
        print("Set OPENROUTER_API_KEY in .env first.")
        sys.exit(1)

    # Step 1: Verify API key validity
    print("Testing API key validity...")
    auth_resp = requests.get(
        f"{BASE_URL}/auth/key",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=15,
    )
    if auth_resp.status_code == 200:
        data = auth_resp.json().get("data", {})
        print(f"API key valid. Label: {data.get('label', 'N/A')}")
        limit = data.get("limit")
        usage = data.get("usage")
        if limit is not None:
            print(f"Credit limit: {limit}, Used: {usage}")
        else:
            print("No spending limit set (free tier).")
    else:
        print(f"API key check failed: {auth_resp.status_code}")
        print(auth_resp.text)
        sys.exit(1)

    # Step 2: Try chat completions with several free models
    free_models = [DEFAULT_MODEL, *DEFAULT_FALLBACKS]

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "LLM Test Project",
    }
    payload_template = {"messages": [{"role": "user", "content": "Say hello in one word."}]}

    print(f"\n{'=' * 60}")
    print("Trying free models...")
    print(f"{'=' * 60}")

    for model_id in free_models:
        payload = {**payload_template, "model": model_id}
        print(f"\n-> {model_id}")
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code == 200:
                body = resp.json()
                reply = body["choices"][0]["message"]["content"]
                model_used = body.get("model", "?")
                print(f"Success. model={model_used}")
                print(f"Reply: {reply.strip()[:120]}")
            else:
                err = resp.json().get("error", {})
                print(f"Error {resp.status_code}: {err.get('message', resp.text)[:120]}")
        except Exception as exc:
            print(f"Exception: {exc}")

        time.sleep(1)

    print(f"\n{'=' * 60}")
    print("Diagnostic complete.")


if __name__ == "__main__":
    main()
