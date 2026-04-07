# AGENTS.md - LLM Chat Test Project

Maintainer context for AI coding agents working in this repository.

## Mission

`llm_client` is a reusable Python package for OpenAI-compatible chat and vision calls. The default endpoint is OpenRouter, the default text model is `stepfun/step-3.5-flash:free`, and the package includes retry plus fallback handling, sync and async APIs, and batch helpers.

## Current Repository Shape

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

## Maintainer Workflow

Choose one isolated environment and keep every install, smoke run, and test run inside it.

PowerShell with Conda:

```powershell
conda env create --prefix "$PWD/.conda/llm_test" -f environment.yaml
conda activate "$PWD/.conda/llm_test"
python -m pip install -e .
python -m pip install -e ".[dev,diagnostics]"
```

PowerShell with `venv`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -e ".[dev,diagnostics]"
```

Verification commands:

```powershell
python -m pytest -q
python scripts/smoke/client_smoke.py
python scripts/smoke/vl_smoke.py
python scripts/smoke/raw_sdk_smoke.py
python scripts/diagnostics/openrouter_diagnostic.py
```

Use `verification_tests/table-mixed.png` as the bundled fixture for the VL smoke flow.

## Debugging

Committed VS Code launch profiles live in `.vscode/launch.json`:

- `Python: Current File`
- `Python: Basic Usage Example`
- `Python: Client Smoke Test`
- `Python: VL Smoke Test`
- `Python: Pytest`

`.vscode/settings.json` points VS Code at `${workspaceFolder}\.venv\Scripts\python.exe`, enables pytest, and keeps terminal activation inside the project environment.

## Constraints

1. Never run project commands in the base environment.
2. Never delete `CLAUDE.md` or `GEMINI.md`.
3. Never commit `.env`, `.env.*`, or live API keys. Commit `.env.example` only as the safe template.
4. Keep generated artifacts out of the repo state: `__pycache__/`, `.pytest_cache/`, `.lprof/`, `.agent/`, and `.claude/`.
5. Keep `README.md` focused on user-facing setup, verification, and debugging.
6. Keep `AGENTS.md` focused on maintainer context, repo rules, and verification flow.
7. Update both docs when package entry points, scripts, environment setup, or debug workflows change.

*Last updated: 2026-04-07*
