# exam-helper

Local-first app for creating and validating physics exam questions (free-response and multiple-choice), with Word DOCX export.

## Pandoc For DOCX Math

DOCX math rendering uses Pandoc when available. This converts LaTeX-style math
delimiters (for example `$...$`, `$$...$$`, `\(...\)`, `\[...\]`) into proper
Word math objects.

Install Pandoc:

- Windows: `winget install --id JohnMacFarlane.Pandoc -e`
- macOS: `brew install pandoc`
- Linux (Debian/Ubuntu): `sudo apt-get install pandoc`

If Pandoc is not installed, export still works but math is downgraded to plain
text markdown and a warning is shown.

## Instructor Quickstart (Windows/macOS)

1. Install `uv`: https://docs.astral.sh/uv/
2. Create a project:
   - `uv run exam-helper init my-exam --name "Midterm 1" --course "Calc-based Intro Physics"`
3. Run the app:
   - `uv run exam-helper serve --project my-exam`
4. Open `http://127.0.0.1:8000`.
5. Create/edit questions, then validate:
   - `uv run exam-helper validate my-exam`
6. Export for Word collaboration:
   - `uv run exam-helper export docx my-exam --output midterm1.docx`

## OpenAI Key

Option 1: put in home `.env`:

- Windows: `C:\Users\<you>\.env`
- macOS: `/Users/<you>/.env`
- Content: `EXAM_HELPER_OPENAI_KEY=sk-...`

Option 2: pass directly on command line:

- `uv run exam-helper serve --project my-exam --openai-key sk-...`

Command line key takes precedence.

## Editor behavior

- AI actions only run when you click a button.
- AI output is applied directly to the main field (single copy), with Undo per section.
- Autosave runs while you type (1.5s debounce).
- Multiple choice editor is hidden unless question type is `Multiple Choice`.
- MC generation creates full A-E options and marks a guessed correct answer.

## Question Portability

Each question lives in one YAML file and includes embedded figures (base64 + MIME + hash), so you can email one file and keep the question self-contained.

## Future `uvx` Usage

After publishing the package:

- `uvx exam-helper serve --project my-exam`

## Documentation

- `docs/getting-started.md`
- `docs/images-and-math.md`
- `docs/validation.md`
- `docs/export-to-word.md`
