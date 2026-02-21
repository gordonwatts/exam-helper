# Export to Word

Use DOCX export to hand off exam drafts for Word-based collaboration.

```bash
uv run exam-helper export docx my-exam --output exam.docx
```

To exclude worked solutions:

```bash
uv run exam-helper export docx my-exam --output exam-student.docx --no-include-solutions
```

The export includes:

- numbered format: `1. [5 points] <question text>`
- embedded figures when renderable by Word
- MC options in `A.` through `E.` form
- optional solution and rubric section
