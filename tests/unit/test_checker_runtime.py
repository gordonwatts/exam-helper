from __future__ import annotations

from exam_helper.checker_runtime import run_checker, symbolic_equivalent, units_compatible


def test_symbolic_equivalence() -> None:
    assert symbolic_equivalent("2*x + 2*x", "4*x")


def test_units_compatible() -> None:
    assert units_compatible("9.8 meter/second**2", "meter/second**2")


def test_checker_contract() -> None:
    code = """
def grade(student_answer, context):
    value = float(student_answer.get("value", 0))
    if abs(value - 10.0) < 1e-6:
        return {"verdict": "correct", "score": 1.0, "feedback": "ok"}
    return {"verdict": "incorrect", "score": 0.0, "feedback": "wrong"}
"""
    result = run_checker(code, {"value": 10.0}, {})
    assert result.verdict == "correct"
