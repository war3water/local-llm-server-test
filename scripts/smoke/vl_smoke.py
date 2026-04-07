"""
VL (Vision-Language) smoke test for LLMClient.

Sends a test image to the default VL model and prints the response.

Usage:
    Activate the project environment (`.venv` or `.conda/llm_test`)
    python scripts/smoke/vl_smoke.py

Requires:
    - A valid API key in .env
    - verification_tests/table-mixed.png (included in repo)
"""

import os
import sys
from pathlib import Path

from llm_client import LLMClient, DEFAULT_VL_MODEL
from llm_client.config import PLACEHOLDER_API_KEYS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_IMAGE = PROJECT_ROOT / "verification_tests" / "table-mixed.png"


def main():
    if not TEST_IMAGE.is_file():
        raise FileNotFoundError(
            f"Test image not found: {TEST_IMAGE}\n"
            "Place a test image in verification_tests/ to run this smoke test."
        )
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key in PLACEHOLDER_API_KEYS:
        print("Set LLM_API_KEY or OPENROUTER_API_KEY in .env before running this script.")
        sys.exit(1)

    llm = LLMClient(api_key=api_key, verbose=True)

    # ── Non-streaming VL test ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"VL model: {DEFAULT_VL_MODEL}")
    print(f"Image:    {TEST_IMAGE.name}")
    print(f"{'=' * 60}")
    print("\n[Non-streaming] Asking model to describe the table...\n")

    reply = llm.chat_vision(
        prompt=(
            "Describe this image in detail. "
            "If it contains a table, list all columns and rows with their values."
        ),
        images=TEST_IMAGE,
    )
    print(f"\nReply:\n{reply}")

    # ── Streaming VL test ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("[Streaming] Asking model to extract structured data...\n")

    reply = llm.chat_vision(
        prompt="Extract the data from this table as JSON.",
        images=TEST_IMAGE,
        stream=True,
    )

    print(f"\n{'=' * 60}")
    print("VL smoke test completed.")


if __name__ == "__main__":
    main()
