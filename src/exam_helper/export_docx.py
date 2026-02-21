from __future__ import annotations

import base64
import io
from pathlib import Path

from docx import Document

from exam_helper.models import Question, QuestionType
from exam_helper.repository import ProjectRepository


def _add_question(doc: Document, index: int, q: Question, include_solutions: bool) -> None:
    question_text = q.prompt_md.strip() or (q.title.strip() if q.title else "")
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
            doc.add_paragraph(f"  {choice.label}. {choice.content_md}")

    if include_solutions:
        doc.add_paragraph("Solution:")
        doc.add_paragraph(q.solution.worked_solution_md)
        if q.solution.rubric:
            doc.add_paragraph("Rubric")
            for item in q.solution.rubric:
                doc.add_paragraph(f"- {item}")


def export_project_to_docx(
    project_root: Path, output_path: Path, include_solutions: bool = True
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
