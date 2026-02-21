from __future__ import annotations

import base64
import io
from pathlib import Path

from docx import Document
from docx.shared import Pt

from exam_helper.models import Question, QuestionType
from exam_helper.repository import ProjectRepository


def _normalize_for_docx(text: str) -> str:
    # Keep readable fallback by stripping latex delimiters for Word plain text runs.
    return (
        text.replace("\\(", "")
        .replace("\\)", "")
        .replace("\\[", "")
        .replace("\\]", "")
    )


def _compact_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    compact: list[str] = []
    prev_blank = False
    for line in lines:
        blank = line == ""
        if blank and prev_blank:
            continue
        compact.append(line)
        prev_blank = blank
    return compact


def _add_question(doc: Document, index: int, q: Question, include_solutions: bool) -> None:
    question_text = _normalize_for_docx(q.prompt_md.strip() or (q.title.strip() if q.title else ""))
    doc.add_paragraph(f"{index}. [{q.points} points] {question_text}")

    for figure in q.figures:
        raw = base64.b64decode(figure.data_base64.encode("ascii"))
        stream = io.BytesIO(raw)
        try:
            doc.add_picture(stream)
            if figure.caption:
                doc.add_paragraph(f"Figure: {figure.caption}")
        except Exception:
            doc.add_paragraph("[Figure could not be rendered in DOCX]")

    if q.question_type == QuestionType.multiple_choice and q.choices:
        sorted_choices = sorted(q.choices, key=lambda c: c.label)
        for choice in sorted_choices:
            doc.add_paragraph(f"  {choice.label}. {_normalize_for_docx(choice.content_md)}")

    if include_solutions:
        p = doc.add_paragraph("Solution:")
        p.runs[0].italic = True
        p.runs[0].font.size = Pt(10)
        for line in _compact_lines(_normalize_for_docx(q.solution.worked_solution_md)):
            if not line:
                continue
            sol = doc.add_paragraph(line)
            for run in sol.runs:
                run.italic = True
                run.font.size = Pt(10)
        if q.solution.rubric:
            rubric_head = doc.add_paragraph("Rubric")
            for run in rubric_head.runs:
                run.italic = True
                run.font.size = Pt(10)
            for item in q.solution.rubric:
                rub = doc.add_paragraph(f"- {_normalize_for_docx(item)}")
                for run in rub.runs:
                    run.italic = True
                    run.font.size = Pt(10)


def export_project_to_docx(
    project_root: Path, output_path: Path, include_solutions: bool = False
) -> None:
    repo = ProjectRepository(project_root)
    project = repo.load_project()
    questions = repo.list_questions()

    doc = Document()
    doc.add_heading(project.name or "Exam", level=1)
    doc.add_paragraph(project.course)
    for i, q in enumerate(questions, start=1):
        _add_question(doc, i, q, include_solutions=include_solutions)
    doc.save(output_path)
