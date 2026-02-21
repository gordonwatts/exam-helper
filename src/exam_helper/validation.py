from __future__ import annotations

from exam_helper.checker_runtime import CheckerError, run_checker
from exam_helper.models import Question
from exam_helper.repository import ProjectRepository


def validate_question(question: Question) -> list[str]:
    errors: list[str] = []
    if question.checker.python_code.strip():
        try:
            run_checker(question.checker.python_code, question.checker.sample_answer, {})
        except CheckerError as ex:
            errors.append(f"{question.id}: {ex}")
    return errors


def validate_project(repo: ProjectRepository) -> list[str]:
    errors = repo.validate_all()
    if errors:
        return errors
    for q in repo.list_questions():
        errors.extend(validate_question(q))
    return errors
