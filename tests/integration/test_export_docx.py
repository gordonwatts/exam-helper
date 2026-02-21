from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from docx import Document

from exam_helper.export_docx import export_project_to_docx
from exam_helper.models import FigureData, MCChoice, Question, QuestionType
from exam_helper.repository import ProjectRepository


def test_export_docx_with_embedded_figure(tmp_path: Path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")

    # 1x1 transparent PNG
    b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/w8AAgMBgU7Y5e0AAAAASUVORK5CYII="
    )
    raw_hash = hashlib.sha256(base64.b64decode(b64.encode("ascii"))).hexdigest()
    fig = FigureData(
        id="fig_1",
        mime_type="image/png",
        data_base64=b64,
        sha256=raw_hash,
        caption="tiny",
    )
    q = Question(
        id="q1",
        title="t",
        prompt_md="p",
        points=5,
        figures=[fig],
        question_type=QuestionType.multiple_choice,
        choices=[
            MCChoice(label="A", content_md="A1", is_correct=True),
            MCChoice(label="B", content_md="B1", is_correct=False),
            MCChoice(label="C", content_md="C1", is_correct=False),
            MCChoice(label="D", content_md="D1", is_correct=False),
            MCChoice(label="E", content_md="E1", is_correct=False),
        ],
    )
    repo.save_question(q)

    out = tmp_path / "exam.docx"
    export_project_to_docx(tmp_path, out)
    assert out.exists()
    assert out.stat().st_size > 0
    doc = Document(out)
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "1. [5 points] p" in text
    assert "A. A1" in text
    assert "E. E1" in text
