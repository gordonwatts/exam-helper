# Export to Word

Use DOCX export to hand off exam drafts for Word-based collaboration.

```bash
uv run exam-helper export docx my-exam --output exam.docx
```

To include worked solutions:

```bash
uv run exam-helper export docx my-exam --output exam-with-solutions.docx --include-solutions
```

The export includes:

- Word-numbered question lists
- embedded figures when renderable by Word
- Word lettered MC option lists (A/B/C labels via list numbering style)
- optional solution and rubric section
- math conversion to native Word math via Pandoc when available
- explicit warning behavior when Pandoc is unavailable or conversion fails

Implementation note:
- MC options are emitted to Pandoc markdown as `A)`, `B)`, ... (not `A.`, `B.`).
- Reason: with Pandoc DOCX conversion, `A)` reliably produces true upper-letter Word lists,
  while `A.` was inconsistent and could degrade into plain text or bullets depending on
  surrounding markdown structure.
- If Pandoc is missing/fails, export falls back to python-docx and emits a warning
  (CLI warning line and web export warning signal).
- Included solutions are formatted as indented, italic, smaller-text blocks.
- Export strips any `Problem (verbatim): ...` line from solution text in the DOCX output.

From the web app main page, use **Export DOCX** and the optional
**Include solutions** checkbox.
