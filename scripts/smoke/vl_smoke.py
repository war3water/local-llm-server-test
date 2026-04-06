"""
VL (Vision-Language) smoke test for LLMClient.

Sends a test image to the default VL model and prints the response.

Usage:
    conda activate llm_test
    python scripts/smoke/vl_smoke.py

Requires:
    - A valid API key in .env
    - verification_tests/table-mixed.png (included in repo)
"""

from pathlib import Path

from llm_client import LLMClient, DEFAULT_VL_MODEL


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_IMAGE = PROJECT_ROOT / "verification_tests" / "table-mixed.png"


def main():
    if not TEST_IMAGE.is_file():
        raise FileNotFoundError(
            f"Test image not found: {TEST_IMAGE}\n"
            "Place a test image in verification_tests/ to run this smoke test."
        )

    llm = LLMClient(verbose=True)

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
