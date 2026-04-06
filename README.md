# LLM Chat Client

Reusable Python package for OpenAI-compatible chat and vision requests with sync and async APIs, retry plus fallback handling, and concurrent batch helpers. The default provider path is [OpenRouter](https://openrouter.ai/) through its OpenAI-compatible endpoint.

## What This Repo Contains

- `llm_client/`: installable package with `LLMClient`
- `examples/`: runnable sync and async usage examples
- `scripts/smoke/`: quick online checks for chat, vision, and raw SDK access
- `scripts/diagnostics/`: provider connectivity diagnostic
- `tests/`: offline unit tests with mocked SDK calls
- `verification_tests/`: image fixture used by the VL smoke script
- `.vscode/launch.json`: ready-to-use VS Code debug profiles

## Quick Start

### 1. Create and activate the project environment

```bash
conda env create -f environment.yaml
conda activate llm_test
```

Always run this project inside `llm_test`. Do not use the base environment.

### 2. Install the package

```bash
pip install -e .
pip install -e ".[dev,diagnostics]"
```

### 3. Add your API key

Copy `.env.example` to `.env` in the project root and replace the placeholder value:

```ini
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

`.env` is gitignored and should stay local.

### 4. Verify the repo

```bash
python scripts/smoke/client_smoke.py
python scripts/smoke/vl_smoke.py
python -m pytest -q
```

`tests/` run offline. The smoke scripts require a valid API key. `scripts/smoke/vl_smoke.py` reads `verification_tests/table-mixed.png`.

## Debug In VS Code

Open the repo in VS Code and use Run and Debug (`Ctrl+Shift+D`). The committed launch profiles are:

- `Python: Current File`
- `Python: Basic Usage Example`
- `Python: Client Smoke Test`
- `Python: VL Smoke Test`
- `Python: Pytest`

Each profile runs in the integrated terminal, loads `${workspaceFolder}/.env`, and sets `PYTHONDONTWRITEBYTECODE=1`.

## Use From Another Project

```bash
conda activate llm_test
pip install -e "/path/to/13_test_project_llm_chat"
```

```python
from llm_client import LLMClient

llm = LLMClient()
reply = llm.chat("Hello!")
```

## Core Capabilities

- `chat()` / `achat()`: single prompt chat
- `chat_messages()` / `achat_messages()`: multi-message chat
- `chat_vision()` / `achat_vision()`: vision-language requests
- `batch_chat()` / `abatch_chat()`: ordered concurrent batch calls
- `create()` / `acreate()`: raw SDK-style response access
- `encode_image()` / `build_vision_content()`: image preparation helpers

## Configuration

The client loads `.env` automatically through `python-dotenv`.

| Variable | Default |
| --- | --- |
| `OPENROUTER_API_KEY` | unset |
| `LLM_API_KEY` | unset |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | `stepfun/step-3.5-flash:free` |
| `LLM_FALLBACK_MODELS` | `deepseek/deepseek-r1-0528:free, google/gemma-3-1b-it:free, mistralai/mistral-small-3.1-24b-instruct:free, meta-llama/llama-4-scout:free` |
| `LLM_VL_MODEL` | `nvidia/nemotron-nano-12b-v2-vl:free` |
| `LLM_VL_FALLBACK_MODELS` | unset |

Constructor arguments override environment values.

## Scripts

| Path | Purpose |
| --- | --- |
| `scripts/smoke/client_smoke.py` | Online smoke test for standard chat |
| `scripts/smoke/vl_smoke.py` | Online smoke test for vision-language chat |
| `scripts/smoke/raw_sdk_smoke.py` | Online smoke test using the raw OpenAI SDK |
| `scripts/diagnostics/openrouter_diagnostic.py` | Connectivity and model diagnostic |

## Project Layout

```text
13_test_project_llm_chat/
├── llm_client/
│   ├── __init__.py
│   ├── client.py
│   └── config.py
├── examples/
│   ├── async_usage.py
│   └── basic_usage.py
├── scripts/
│   ├── diagnostics/
│   │   └── openrouter_diagnostic.py
│   └── smoke/
│       ├── client_smoke.py
│       ├── raw_sdk_smoke.py
│       └── vl_smoke.py
├── tests/
│   ├── test_client.py
│   └── test_vl.py
├── verification_tests/
│   └── table-mixed.png
├── .vscode/
│   └── launch.json
├── AGENTS.md
├── CLAUDE.md
├── GEMINI.md
├── environment.yaml
├── pyproject.toml
└── README.md
```

## Repository Hygiene

- Keep generated files out of commits: `__pycache__/`, `.pytest_cache/`, `.lprof/`, and local agent folders such as `.agent/` and `.claude/`
- Commit `.env.example` as the safe template and keep `.env` and other `.env.*` files local
- Keep `CLAUDE.md` and `GEMINI.md` in place because they point to the maintainer context file
- Treat this README as the user-facing quick start
- Treat `AGENTS.md` as the maintainer-facing project context

## Troubleshooting

If imports fail, reinstall the package in the active environment:

```bash
conda activate llm_test
pip install -e ".[dev,diagnostics]"
```

If API calls fail, confirm `.env` is present and run:

```bash
python scripts/diagnostics/openrouter_diagnostic.py
```

If VS Code debugging cannot import `llm_client`, select the interpreter from the `llm_test` environment before running a launch profile.
