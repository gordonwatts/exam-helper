# AGENTS.md

This file is for future coding agents working in this repository.

## Project Mission

Build and maintain a local-first exam authoring app for calculus-based intro physics (mechanics, E&M, waves), extensible to advanced courses.

Core requirements:
- author free-response and multiple-choice questions
- embed figures directly in question YAML (single-file portability)
- deterministic answer checking (no LLM arithmetic)
- AI-assisted drafting with human approval
- export to Word (`.docx`) for collaboration

## Current Stack

- Python 3.12
- FastAPI + Jinja + HTMX
- YAML file storage
- OpenAI official Python SDK
- `sympy` + `pint` for checking logic
- `python-docx` for export
- `pytest` for tests

## Repository Map

- `src/exam_helper/cli.py`
  - CLI entrypoint (`exam-helper`)
- `src/exam_helper/app.py`
  - FastAPI routes + UI flow
- `src/exam_helper/models.py`
  - Pydantic data schema for project/question/figures
- `src/exam_helper/repository.py`
  - project/question YAML persistence
- `src/exam_helper/checker_runtime.py`
  - deterministic checker execution and helper functions
- `src/exam_helper/validation.py`
  - project/question validation orchestration
- `src/exam_helper/ai_service.py`
  - OpenAI-backed generation helpers
- `src/exam_helper/export_docx.py`
  - Word export pipeline
- `src/exam_helper/templates/`
  - Jinja templates (question list + edit form)
- `tests/unit/`, `tests/integration/`
  - test suite
- `docs/`
  - end-user (exam author) docs

## Non-Negotiable Product Constraints

1. Question files must remain self-contained.
   - Figures stay embedded in YAML as base64 + MIME + SHA-256.
   - Do not move to required external asset files in v1.

2. Deterministic checking is mandatory.
   - LLMs may draft code/text, but correctness checks run via Python checker code.

3. AI is human-in-the-loop.
   - Never auto-apply AI output silently.
   - Keep generated drafts reviewable before acceptance.

4. OpenAI key resolution order is fixed:
   - first: `--openai-key`
   - fallback: `~/.env` (`EXAM_HELPER_OPENAI_KEY`)

## Developer Commands

- run app:
  - `uv run exam-helper serve --project my-exam`
- init project:
  - `uv run exam-helper init my-exam --name "Exam Draft" --course "Physics 1"`
- validate:
  - `uv run exam-helper validate my-exam`
- export docx:
  - `uv run exam-helper export docx my-exam --output exam.docx`
- tests:
  - `uv run --extra dev pytest -q`

## GitHub Issue Workflow

- Use `gh` for issue and PR status in this repo when available.
- List open issues with:
  - `gh issue list --state open --limit 200`
- If `gh` is not installed or not available on PATH, ask the user to install/authenticate `gh`, then retry the same command.

### Standard Issue Development Flow

1. Check out `main` and update it before starting issue work.
2. Create a dedicated branch for the issue (for example, `issue-123-short-name`) **before making any code or docs edits**.
3. Implement and test changes on that branch (never do issue implementation work directly on `main`).
4. Open a PR from the branch into `main`.
5. In the PR description, include an issue-closing reference such as `fix #123` so GitHub automatically closes the issue when the PR is merged.

## Packaging and Distribution

- Local development uses `uv run`.
- Target distribution is `uvx exam-helper` once published.
- Keep `pyproject.toml` entrypoint and template packaging rules in sync when moving files.

## Testing Expectations

When adding complex logic:
- write/update unit tests first or alongside implementation
- add integration tests for end-to-end flows when behavior crosses modules
- ensure `uv run --extra dev pytest -q` passes before finishing

### Test Run Playbook (from repo root)

- full suite:
  - `uv run --extra dev pytest -q`
- single test file:
  - `uv run --extra dev pytest -q tests/unit/test_prompt_catalog.py`
- single test function:
  - `uv run --extra dev pytest -q tests/unit/test_prompt_catalog.py -k distractors`

If a command fails due to environment/sandbox restrictions, rerun with the same command pattern rather than switching tools.

Priority test areas:
- schema and YAML validation
- figure integrity/hash behavior
- checker runtime contract and symbolic/unit helpers
- AI service behavior (mock client in tests)
- docx export stability

## Coding Guidance

- Favor small, explicit modules.
- Keep business logic out of templates.
- Preserve clean boundaries:
  - schema/model layer
  - repository/persistence layer
  - service logic (AI, validation, export)
  - transport/UI layer (FastAPI routes + Jinja)
- If frontend complexity grows, keep routes/API shapes stable to ease future SPA migration.

## DOCX Export Notes

- DOCX export is Pandoc-first for best math rendering; keep fallback behavior intact when Pandoc is unavailable.
- For MC options in Pandoc markdown, use `A)`, `B)`, ... markers (not `A.`), since this is the reliable way to get lettered Word list semantics in generated DOCX.

## Documentation Expectations

User-facing docs should prioritize exam authors (Windows/macOS):
- setup and run
- writing/editing questions
- image paste flow
- validation usage
- Word export workflow

Source contributor docs are useful but secondary.

## Git and Change Management

- Keep commits focused and descriptive.
- Do not rewrite history unless explicitly requested.
- Avoid destructive git commands.
- If you discover unexpected unrelated changes, pause and ask before proceeding.

## CI

GitHub Actions workflow is at:
- `.github/workflows/ci.yml`

If you add checks (lint/type/etc.), keep runtime practical and avoid breaking fast feedback from `pytest`.
