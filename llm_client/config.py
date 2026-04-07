"""
Default configuration for the LLM client.

Values can be overridden via constructor arguments or environment variables.
"""

import os
from dotenv import load_dotenv

# Auto-load .env from cwd (or any parent) on import
load_dotenv()

PLACEHOLDER_API_KEYS = {
    "your_key_here",
    "your_openrouter_api_key_here",
}


def _parse_csv_env(key: str) -> list[str]:
    """Parse comma-separated env values and drop empty entries."""
    raw = os.getenv(key, "")
    if not raw.strip():
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


# ── API ──────────────────────────────────────────────────────────────────────
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
DEFAULT_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
DEFAULT_TIMEOUT = 60.0  # seconds

# ── Models ───────────────────────────────────────────────────────────────────
DEFAULT_MODEL = os.getenv("LLM_MODEL", "stepfun/step-3.5-flash:free")

_OPENROUTER_DEFAULT_FALLBACKS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "minimax/minimax-m2.5:free",
    "arcee-ai/trinity-large-preview:free",
    "openai/gpt-oss-20b:free",
]
DEFAULT_FALLBACKS = _parse_csv_env("LLM_FALLBACK_MODELS") or list(_OPENROUTER_DEFAULT_FALLBACKS)

# ── Vision-Language (VL) Models ──────────────────────────────────────────────
DEFAULT_VL_MODEL = os.getenv("LLM_VL_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free")
DEFAULT_VL_FALLBACKS: list[str] = _parse_csv_env("LLM_VL_FALLBACK_MODELS")

# ── Retry ────────────────────────────────────────────────────────────────────
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_WAIT = 3  # seconds, doubles each retry
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_MAX_BACKOFF = 30.0
DEFAULT_BACKOFF_JITTER = 0.15  # +/-15%

# ── Concurrency ──────────────────────────────────────────────────────────────
DEFAULT_MAX_CONCURRENCY = 5
