# Getting Started

## 1. Install prerequisites

- Python 3.12
- `uv` (https://docs.astral.sh/uv/)

## 2. Create an exam project

```bash
uv run exam-helper init my-exam --name "Exam Draft" --course "Physics 1" --openai-model gpt-5.2
```

This creates:

- `my-exam/project.yaml`
- `my-exam/questions/`

## 3. Start the local app

```bash
uv run exam-helper serve --project my-exam
```

Open `http://127.0.0.1:8000`.

## 4. Create and edit questions

- Click **New Question**
- Fill prompt, solution, checker code
- Paste images from clipboard directly into the editor page
- Use AI buttons when needed (`Improve Text`, `Improve Solution`, `Generate MC A-E`)
- Use `Preview Prompt` buttons to inspect what will be sent before an AI call
- Autosave runs automatically as you edit
- Configure project AI model and optional prompt overrides from the main project page
- Monitor cumulative token/cost usage on the main page (with reset button)

Optional OpenAI key sources:

- `--openai-key` CLI argument
- `~/.env` with `EXAM_HELPER_OPENAI_KEY`

## 5. Validate questions

```bash
uv run exam-helper validate my-exam
```

## 6. Export to Word DOCX

```bash
uv run exam-helper export docx my-exam --output exam.docx
```
