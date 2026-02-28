from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import sympy as sp
from pint import UnitRegistry

from exam_helper.models import MCChoice


ureg = UnitRegistry()


class SolutionRuntimeError(RuntimeError):
    pass


@dataclass
class AnswerRunResult:
    answer_md: str
    final_answer: str


@dataclass
class DistractorRunResult:
    distractor_md: str
    rationale: str


@dataclass
class HarnessRunResult:
    choices: list[MCChoice]
    collisions: list[str]


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


def _run_callable(python_code: str, fn_name: str, params: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    if not python_code.strip():
        raise SolutionRuntimeError("Python code is empty.")
    ns: dict[str, Any] = {}
    try:
        exec(python_code, _safe_globals(), ns)
    except Exception as ex:
        raise SolutionRuntimeError(f"Solution compile error: {ex}") from ex
    fn = ns.get(fn_name)
    if not callable(fn):
        raise SolutionRuntimeError(f"Solution code must define callable {fn_name}(params).")
    safe_params = params or {}
    if not isinstance(safe_params, dict):
        raise SolutionRuntimeError("Solution params must be a mapping.")
    try:
        raw = fn(safe_params)
    except Exception as ex:
        raise SolutionRuntimeError(f"Solution runtime error: {ex}") from ex
    return raw, ns


def run_answer_function(python_code: str, params: dict[str, Any] | None = None) -> AnswerRunResult:
    raw, _ = _run_callable(python_code, "solve", params or {})
    if not isinstance(raw, dict):
        raise SolutionRuntimeError("solve(params) must return a dict.")
    answer_md = raw.get("answer_md")
    final_answer = raw.get("final_answer")
    if not isinstance(answer_md, str) or not answer_md.strip():
        raise SolutionRuntimeError("solve return must include non-empty string answer_md.")
    if not isinstance(final_answer, str) or not final_answer.strip():
        raise SolutionRuntimeError("solve return must include non-empty string final_answer.")
    return AnswerRunResult(answer_md=answer_md.strip(), final_answer=final_answer.strip())


def run_distractor_function(python_code: str, params: dict[str, Any] | None = None) -> DistractorRunResult:
    raw, _ = _run_callable(python_code, "distractor", params or {})
    if not isinstance(raw, dict):
        raise SolutionRuntimeError("distractor(params) must return a dict.")
    distractor_md = raw.get("distractor_md")
    rationale = raw.get("rationale")
    if not isinstance(distractor_md, str) or not distractor_md.strip():
        raise SolutionRuntimeError("distractor return must include non-empty string distractor_md.")
    if not isinstance(rationale, str) or not rationale.strip():
        raise SolutionRuntimeError("distractor return must include non-empty string rationale.")
    return DistractorRunResult(distractor_md=distractor_md.strip(), rationale=rationale.strip())


def _normalize_choice_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def _numeric_sort_key(value: str) -> float | None:
    cleaned = re.sub(r",", "", value or "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _strip_disallowed_bold(text: str) -> str:
    out = text or ""
    out = re.sub(r"\*\*(.*?)\*\*", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"__(.*?)__", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"</?strong>", "", out, flags=re.IGNORECASE)
    out = re.sub(r"</?b>", "", out, flags=re.IGNORECASE)
    return out


def run_mc_harness(
    answer_python_code: str,
    distractor_python_codes: list[tuple[str, str]],
    params: dict[str, Any] | None = None,
) -> HarnessRunResult:
    params = params or {}
    answer = run_answer_function(answer_python_code, params)
    rows: list[dict[str, Any]] = [
        {
            "source_id": "answer",
            "content_md": _strip_disallowed_bold(answer.final_answer),
            "is_correct": True,
            "rationale": "correct answer",
        }
    ]
    for source_id, code in distractor_python_codes:
        d = run_distractor_function(code, params)
        rows.append(
            {
                "source_id": source_id,
                "content_md": _strip_disallowed_bold(d.distractor_md),
                "is_correct": False,
                "rationale": _strip_disallowed_bold(d.rationale),
            }
        )

    seen: dict[str, str] = {}
    collisions: list[str] = []
    for row in rows:
        canonical = _normalize_choice_text(str(row["content_md"]))
        if canonical in seen:
            collisions.append(
                f"Duplicate MC option between '{seen[canonical]}' and '{row['source_id']}': {row['content_md']}"
            )
        else:
            seen[canonical] = str(row["source_id"])

    def _sort_tuple(row: dict[str, Any]) -> tuple[int, float, str, str]:
        numeric = _numeric_sort_key(str(row["content_md"]))
        if numeric is None:
            return (1, 0.0, _normalize_choice_text(str(row["content_md"])), str(row["source_id"]))
        return (0, numeric, _normalize_choice_text(str(row["content_md"])), str(row["source_id"]))

    rows.sort(key=_sort_tuple)
    labels = ["A", "B", "C", "D", "E"]
    choices: list[MCChoice] = []
    for idx, row in enumerate(rows):
        choices.append(
            MCChoice(
                label=labels[idx] if idx < len(labels) else "?",
                content_md=str(row["content_md"]),
                is_correct=bool(row["is_correct"]),
                rationale=str(row["rationale"]),
            )
        )
    return HarnessRunResult(choices=choices, collisions=collisions)
