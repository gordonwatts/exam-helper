from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import sympy as sp
from pint import UnitRegistry


ureg = UnitRegistry()


class SolutionRuntimeError(RuntimeError):
    pass


@dataclass
class SolutionRunResult:
    computed_output_md: str
    choices_yaml: str | None = None
    raw: dict[str, Any] | None = None


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


def run_solution_code(
    python_code: str,
    params: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> SolutionRunResult:
    if not python_code.strip():
        raise SolutionRuntimeError("Solution code is empty.")
    ns: dict[str, Any] = {}
    try:
        exec(python_code, _safe_globals(), ns)
    except Exception as ex:
        raise SolutionRuntimeError(f"Solution compile error: {ex}") from ex

    solve = ns.get("solve")
    if not callable(solve):
        raise SolutionRuntimeError("Solution code must define callable solve(params, context).")

    safe_params = params or {}
    if not isinstance(safe_params, dict):
        raise SolutionRuntimeError("Solution params must be a mapping.")
    try:
        raw = solve(safe_params, context or {})
    except Exception as ex:
        raise SolutionRuntimeError(f"Solution runtime error: {ex}") from ex
    if not isinstance(raw, dict):
        raise SolutionRuntimeError("solve(params, context) must return a dict.")

    computed_output_md = raw.get("computed_output_md")
    if not isinstance(computed_output_md, str) or not computed_output_md.strip():
        raise SolutionRuntimeError("solve return must include non-empty string computed_output_md.")
    choices_yaml = raw.get("choices_yaml")
    if choices_yaml is not None and not isinstance(choices_yaml, str):
        raise SolutionRuntimeError("choices_yaml must be a string when provided.")
    return SolutionRunResult(
        computed_output_md=computed_output_md.strip(),
        choices_yaml=choices_yaml.strip() if isinstance(choices_yaml, str) else None,
        raw=raw,
    )
