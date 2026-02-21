from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import sympy as sp
from pint import UnitRegistry


ureg = UnitRegistry()


class CheckerError(RuntimeError):
    pass


@dataclass
class CheckerResult:
    verdict: str
    score: float
    feedback: str
    raw: dict[str, Any]


def symbolic_equivalent(expr_a: str, expr_b: str) -> bool:
    a = sp.sympify(expr_a)
    b = sp.sympify(expr_b)
    return sp.simplify(a - b) == 0


def units_compatible(value_expr: str, expected_units: str) -> bool:
    value = ureg(value_expr)
    return value.check(expected_units)


def _safe_globals() -> dict[str, Any]:
    allowed_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "round": round,
        "float": float,
        "int": int,
        "str": str,
        "dict": dict,
        "list": list,
        "set": set,
        "tuple": tuple,
        "range": range,
    }
    return {
        "__builtins__": allowed_builtins,
        "math": math,
        "sp": sp,
        "ureg": ureg,
        "symbolic_equivalent": symbolic_equivalent,
        "units_compatible": units_compatible,
    }


def run_checker(
    python_code: str, student_answer: dict[str, Any], context: dict[str, Any] | None = None
) -> CheckerResult:
    if not python_code.strip():
        raise CheckerError("Checker code is empty.")
    ns: dict[str, Any] = {}
    try:
        exec(python_code, _safe_globals(), ns)
    except Exception as ex:
        raise CheckerError(f"Checker compile error: {ex}") from ex

    grade = ns.get("grade")
    if not callable(grade):
        raise CheckerError("Checker must define callable grade(student_answer, context).")

    try:
        raw = grade(student_answer, context or {})
    except Exception as ex:
        raise CheckerError(f"Checker runtime error: {ex}") from ex
    if not isinstance(raw, dict):
        raise CheckerError("Checker must return a dict.")

    verdict = raw.get("verdict")
    if verdict not in {"correct", "partial", "incorrect"}:
        raise CheckerError("Checker verdict must be correct, partial, or incorrect.")
    score = float(raw.get("score", 1.0 if verdict == "correct" else 0.0))
    feedback = str(raw.get("feedback", ""))
    return CheckerResult(verdict=verdict, score=score, feedback=feedback, raw=raw)
