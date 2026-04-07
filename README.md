# LLM Chat Client

Reusable Python package for OpenAI-compatible chat and vision requests with sync and async APIs, retry plus fallback handling, and concurrent batch helpers. The default provider path is [OpenRouter](https://openrouter.ai/) through its OpenAI-compatible endpoint.

## What This Repo Contains

- `llm_client/`: installable package with `LLMClient`
- `examples/`: runnable sync and async usage examples
- `scripts/smoke/`: online checks for chat, vision, and raw SDK access
- `scripts/diagnostics/`: provider connectivity diagnostic
- `tests/`: offline unit tests with mocked SDK calls
- `verification_tests/`: bundled image fixture for vision smoke tests
- `.vscode/launch.json`: committed VS Code debug profiles
- `.vscode/settings.json`: default VS Code interpreter and pytest settings for the local `.venv`

## Quick Start

### 1. Create and activate a project environment

Choose one environment path and keep all commands inside it.

PowerShell with Conda:

```powershell
conda env create --prefix "$PWD/.conda/llm_test" -f environment.yaml
conda activate "$PWD/.conda/llm_test"
```

PowerShell with `venv`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

`environment.yaml` is the Conda source of truth. `.venv` is the committed debug target in VS Code for this workspace.

### 2. Install the package

```powershell
python -m pip install -e .
python -m pip install -e ".[dev,diagnostics]"
```

### 3. Add your API key

Copy `.env.example` to `.env` in the project root and replace the placeholder value:

```powershell
Copy-Item .env.example .env
```

```ini
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Keep `.env` local. Keep `.env.example` as a safe template without live secrets.

### 4. Verify the repo

Offline verification:

```powershell
python -m pytest -q
```

Online verification:

```powershell
python scripts/smoke/client_smoke.py
python scripts/smoke/vl_smoke.py
python scripts/smoke/raw_sdk_smoke.py
python scripts/diagnostics/openrouter_diagnostic.py
```

`scripts/smoke/vl_smoke.py` reads `verification_tests/table-mixed.png`.

## Debug In VS Code

Open the repo in VS Code and use Run and Debug (`Ctrl+Shift+D`). The committed launch profiles are:

- `Python: Current File`
- `Python: Basic Usage Example`
- `Python: Client Smoke Test`
- `Python: VL Smoke Test`
- `Python: Pytest`

The launch profiles run in the integrated terminal, use `${workspaceFolder}` as `cwd`, load `${workspaceFolder}/.env`, and work with the interpreter configured in `.vscode/settings.json`.

## Use From Another Project

```powershell
python -m pip install -e "D:\path\to\local-llm-server-test"
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
| `LLM_FALLBACK_MODELS` | `nvidia/nemotron-3-super-120b-a12b:free, minimax/minimax-m2.5:free, arcee-ai/trinity-large-preview:free, openai/gpt-oss-20b:free` |
| `LLM_VL_MODEL` | `nvidia/nemotron-nano-12b-v2-vl:free` |
| `LLM_VL_FALLBACK_MODELS` | unset |

Constructor arguments override environment values.

## Scripts

| Path | Purpose |
| --- | --- |
| `scripts/smoke/client_smoke.py` | Online smoke test for standard chat |
| `scripts/smoke/vl_smoke.py` | Online smoke test for vision-language chat |
| `scripts/smoke/raw_sdk_smoke.py` | Online smoke test using the raw OpenAI SDK |
| `scripts/diagnostics/openrouter_diagnostic.py` | Connectivity and default-model diagnostic |

## Project Layout

```text
local-llm-server-test/
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
│   ├── launch.json
│   └── settings.json
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

If `conda` is not available in your shell, use the local `.venv` workflow:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,diagnostics]"
```

If imports fail, reinstall the package inside the active project environment:

```powershell
python -m pip install -e ".[dev,diagnostics]"
```

If API calls fail, confirm `.env` is present and run:

```powershell
python scripts/diagnostics/openrouter_diagnostic.py
```

If VS Code debugging cannot import `llm_client`, select the interpreter from the project environment before running a launch profile.
