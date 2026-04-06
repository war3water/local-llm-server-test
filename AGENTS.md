# AGENTS.md - LLM Chat Test Project

Maintainer context for AI coding agents working in this repository.

## Mission

`llm_client` is a reusable Python package for OpenAI-compatible chat and vision calls. The default endpoint is OpenRouter, the default text model is `stepfun/step-3.5-flash:free`, and the package includes retry plus fallback handling, sync and async APIs, and batch helpers.

## Current Repository Shape

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

## Maintainer Workflow

```bash
conda env create -f environment.yaml
conda activate llm_test
pip install -e .
pip install -e ".[dev,diagnostics]"

python scripts/smoke/client_smoke.py
python scripts/smoke/vl_smoke.py
python -m pytest -q
```

Use `verification_tests/table-mixed.png` as the bundled fixture for the VL smoke flow.

## Debugging

Committed VS Code launch profiles live in `.vscode/launch.json`:

- `Python: Current File`
- `Python: Basic Usage Example`
- `Python: Client Smoke Test`
- `Python: VL Smoke Test`
- `Python: Pytest`

They run in the integrated terminal, use `${workspaceFolder}` as `cwd`, and load `${workspaceFolder}/.env`.

## Constraints

1. Never run project commands in the base Conda environment.
2. Never delete `CLAUDE.md` or `GEMINI.md`.
3. Never commit `.env` or other local `.env.*` files. Commit `.env.example` as the template.
4. Keep generated artifacts out of the repo state: `__pycache__/`, `.pytest_cache/`, `.lprof/`, `.agent/`, and `.claude/`.
5. Keep `README.md` user-facing and quick-start oriented.
6. Keep `AGENTS.md` focused on maintainer context, repo rules, and verification flow.
7. Update both docs when package entry points, scripts, environment setup, or debug workflows change.

*Last updated: 2026-04-07*
