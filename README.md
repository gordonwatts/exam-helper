# exam-helper

Local-first exam authoring app for calculus-based intro physics (mechanics, E&M, waves), with support for advanced courses.

## Quickstart (Windows/macOS)

1. Install `uv`: https://docs.astral.sh/uv/
2. Run locally in development:
   - `uv run exam-helper serve`
3. Initialize a project:
   - `uv run exam-helper init my-exam`
4. Validate project files:
   - `uv run exam-helper validate my-exam`
5. Export to Word:
   - `uv run exam-helper export docx my-exam --output exam.docx`

## OpenAI key configuration

You can provide the key in either of these ways:

- Home directory `.env` file:
  - `OPENAI_API_KEY=...`
- Command line:
  - `uv run exam-helper serve --openai-key sk-...`

Command line key overrides `.env`.

## uvx distribution target

After publishing this package to GitHub/PyPI, the app is expected to run via:

- `uvx exam-helper serve`

## Docs

- `docs/getting-started.md`
- `docs/images-and-math.md`
- `docs/validation.md`
- `docs/export-to-word.md`
