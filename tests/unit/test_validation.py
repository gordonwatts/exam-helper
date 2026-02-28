from __future__ import annotations

from pathlib import Path

import yaml

from exam_helper.models import Question
from exam_helper.repository import ProjectRepository
from exam_helper.validation import validate_project


def test_validate_project_with_solution_code(tmp_path: Path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    q = Question(
        id="q1",
        title="t",
        prompt_md="p",
        solution={
            "python_code": (
                "def solve(params, context):\n"
                "    return {'final_answer_text': 'ok'}\n"
            ),
            "parameters": {},
        },
    )
    repo.save_question(q)
    assert validate_project(repo) == []


def test_validate_project_ignores_legacy_checker_data(tmp_path: Path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    question_path = tmp_path / "questions" / "q1.yaml"
    question_path.write_text(
        yaml.safe_dump(
            {
                "id": "q1",
                "title": "t",
                "prompt_md": "p",
                "question_type": "free_response",
                "choices": [],
                "checker": {
                    "python_code": "def grade(student_answer, context):\n    return {'verdict': 'bad'}\n"
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    assert validate_project(repo) == []


def test_validate_project_skips_soft_deleted_questions(tmp_path: Path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    repo.save_question(
        Question(
            id="q_deleted",
            title="t",
            is_deleted=True,
            solution={
                "answer_python_code": (
                    "def solve(params, context):\n"
                    "    raise RuntimeError('boom')\n"
                ),
                "parameters": {},
            },
        )
    )
    assert validate_project(repo) == []
