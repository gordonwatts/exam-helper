from __future__ import annotations

from pathlib import Path

from exam_helper.models import CheckerSpec, Question
from exam_helper.repository import ProjectRepository
from exam_helper.validation import validate_project


def test_validate_project_with_checker(tmp_path: Path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    q = Question(
        id="q1",
        title="t",
        prompt_md="p",
        checker=CheckerSpec(
            python_code=(
                "def grade(student_answer, context):\n"
                "    return {'verdict': 'correct', 'score': 1.0, 'feedback': ''}\n"
            ),
            sample_answer={},
        ),
    )
    repo.save_question(q)
    assert validate_project(repo) == []
