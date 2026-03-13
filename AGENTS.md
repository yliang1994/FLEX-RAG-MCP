# Repository Guidelines

## Project Structure & Module Organization
This branch is a `clean-start` scaffold. The only committed project artifacts today are [`README.md`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/README.md), [`DEV_SPEC.md`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/DEV_SPEC.md), and reusable agent skills under [`.github/skills/`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/.github/skills). Treat `DEV_SPEC.md` as the source of truth for implementation work.

The planned runtime layout is Python-first: `src/` for application code, `tests/unit|integration|e2e/` for coverage layers, `tests/fixtures/` for sample PDFs and golden sets, `config/` for settings and prompts, and `data/` plus `logs/` for local runtime artifacts. Keep generated files out of git; `.gitignore` already excludes caches, virtual environments, traces, and local data stores.

## Build, Test, and Development Commands
There is no runnable app checked in on this branch yet. During bootstrap, contributors should work from the spec and add the standard Python entrypoints described there.

- `python -m compileall src` checks syntax and importability for early scaffold work.
- `pytest -q` is the default full test run once `pyproject.toml` and `tests/` exist.
- `pytest -q tests/unit/test_smoke_imports.py` is the expected first smoke test.
- `streamlit run src/observability/dashboard/app.py` launches the planned dashboard after implementation.
- `python scripts/ingest.py --path tests/fixtures/sample_documents/ --collection test` is the reference ingestion command defined in the spec.

## Coding Style & Naming Conventions
Use Python 3.11+ with 4-space indentation, type hints on public interfaces, and small focused modules. Follow the naming pattern already described in the spec: `base_*.py` for abstractions, `*_factory.py` for provider registries, and snake_case for modules, functions, and test files. Prefer explicit interfaces under `src/core/` and `src/libs/` over provider-specific logic leaking across layers. When formatting and linting are added, wire them through `pyproject.toml`; `pytest` and Ruff-compatible caches are already ignored.

## Testing Guidelines
Mirror the three planned layers: unit tests for contracts and factories, integration tests for storage and retrieval pipelines, and e2e tests for MCP and dashboard flows. Name files `test_<feature>.py`. Mock networked LLM and embedding providers by default; do not rely on live credentials in CI. Add fixtures under `tests/fixtures/` and prefer temporary directories for Chroma, SQLite, and trace outputs.

## Agent-Specific Instructions
Prefer repo-local skills under [`.codex/skills/`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/.codex/skills) when a task matches an available skill there. Use [`.github/skills/`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/.github/skills) only as a fallback or compatibility reference, and do not default to [`.claude/skills/`](/home/yliang/remote_codex/MODULAR-RAG-MCP-SERVER/.claude/skills) when an equivalent `.codex` skill exists.

## Commit & Pull Request Guidelines
Current history follows Conventional Commit style, for example `feat: ...` and `chore: ...`. Keep using that format with a concise scope in the subject. PRs should link the relevant `DEV_SPEC.md` task or section, summarize behavior changes, list validation commands run, and include screenshots only for dashboard or UI changes. Separate scaffolding, feature work, and cleanup into distinct commits when possible.
