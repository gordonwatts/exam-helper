from __future__ import annotations

from exam_helper.models import Question
from exam_helper.repository import ProjectRepository
from exam_helper.solution_runtime import (
    SolutionRuntimeError,
    run_answer_function,
    run_distractor_function,
)


def validate_question(question: Question) -> list[str]:
    errors: list[str] = []
    if question.solution.answer_python_code.strip():
        try:
            run_answer_function(question.solution.answer_python_code, question.solution.parameters)
        except SolutionRuntimeError as ex:
            errors.append(f"{question.id}: {ex}")
    if question.question_type.value == "multiple_choice":
        if len(question.solution.distractor_python_code) != 4:
            errors.append(f"{question.id}: multiple_choice requires exactly four distractor functions.")
        for d in question.solution.distractor_python_code:
            if not d.python_code.strip():
                continue
            try:
                run_distractor_function(d.python_code, question.solution.parameters)
            except SolutionRuntimeError as ex:
                errors.append(f"{question.id}:{d.id}: {ex}")
    return errors


def validate_project(repo: ProjectRepository) -> list[str]:
    errors = repo.validate_all()
    if errors:
        return errors
    for q in repo.list_questions():
        errors.extend(validate_question(q))
    return errors
