from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any

import sympy as sp
from pint import UnitRegistry

from exam_helper.models import MCChoice
from exam_helper.normalization import normalize_python_code_string_literals

ureg = UnitRegistry()


class SolutionRuntimeError(RuntimeError):
    pass


@dataclass
class AnswerRunResult:
    calculated_variables_md: str
    answer_md: str
    final_answer: str
    warnings: list[str] = field(default_factory=list)


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


def render_template_from_values(template: str, values: dict[str, Any]) -> str:
    """Replace `{{name}}` placeholders in a template using mapping values.

    Keys are applied from longest to shortest so nested placeholder names do
    not partially overwrite each other during substitution.
    """
    rendered = template or ""
    items = sorted(
        (values or {}).items(), key=lambda item: len(str(item[0])), reverse=True
    )
    for key, value in items:
        rendered = re.sub(
            r"\{\{\s*" + re.escape(str(key)) + r"\s*\}\}",
            str(value),
            rendered,
        )
    return rendered


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


def _format_value(value: Any) -> str:
    if isinstance(value, sp.Basic):
        return sp.sstr(value)
    return str(value)


def _line_targets(node: ast.AST) -> list[str]:
    """Return assignment target names declared by a single AST statement."""
    targets: list[str] = []

    def visit(target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            targets.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                visit(item)

    if isinstance(node, ast.Assign):
        for target in node.targets:
            visit(target)
    elif isinstance(node, ast.AnnAssign):
        visit(node.target)
    elif isinstance(node, ast.AugAssign):
        visit(node.target)
    return targets


def evaluate_answer_formula(
    formula_md: str, params: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[str], list[str], str | None]:
    """Evaluate a multiline answer formula and capture its computed state.

    The formula is treated as a sequence of plain Python lines. Each line is
    executed or evaluated in order with the question parameters exposed as
    locals. The return value includes:
    - the final local namespace
    - a human-readable log of computed variables
    - any warnings encountered while evaluating lines
    - the first fatal error message, if evaluation stopped early
    """
    source = normalize_python_code_string_literals(formula_md or "")
    if not source.strip():
        raise SolutionRuntimeError("Formula text is empty.")
    safe_params = params or {}
    if not isinstance(safe_params, dict):
        raise SolutionRuntimeError("Solution params must be a mapping.")

    locals_ns: dict[str, Any] = dict(safe_params)
    locals_ns["params"] = safe_params
    warnings: list[str] = []
    rendered_lines: list[str] = []
    last_value: Any = None
    explicit_answer_defined = False
    fatal_error: str | None = None
    globals_ns = _safe_globals()

    for lineno, raw_line in enumerate(source.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            expr = ast.parse(line, mode="eval")
        except SyntaxError:
            try:
                stmt = ast.parse(line, mode="exec")
            except SyntaxError as ex:
                fatal_error = f"Line {lineno}: {ex.msg}"
                warnings.append(fatal_error)
                break

            node = stmt.body[0] if stmt.body else None
            try:
                exec(compile(stmt, "<answer_formula>", "exec"), globals_ns, locals_ns)
            except Exception as ex:
                fatal_error = f"Line {lineno}: {ex}"
                warnings.append(fatal_error)
                break

            targets = _line_targets(node) if node is not None else []
            if targets:
                for target in targets:
                    value = locals_ns.get(target)
                    rendered_lines.append(f"{target} = {_format_value(value)}")
                    last_value = value
                    if target == "answer":
                        if not explicit_answer_defined:
                            warnings.append(
                                "Formula defines answer directly; it will not be overwritten."
                            )
                        explicit_answer_defined = True
            else:
                rendered_lines.append(line)
            continue

        try:
            value = eval(
                compile(expr, "<answer_formula>", "eval"), globals_ns, locals_ns
            )
        except Exception as ex:
            fatal_error = f"Line {lineno}: {ex}"
            warnings.append(fatal_error)
            break

        rendered_lines.append(f"result = {_format_value(value)}")
        last_value = value

    if "answer" not in locals_ns and last_value is not None:
        locals_ns["answer"] = last_value
        rendered_lines.append(f"answer = {_format_value(last_value)}")

    return locals_ns, rendered_lines, warnings, fatal_error


def run_answer_formula(
    formula_md: str,
    params: dict[str, Any] | None = None,
    *,
    answer_text_md: str = "",
    strict: bool = True,
) -> AnswerRunResult:
    """Run an answer formula and render the final answer text.

    When `strict` is true, syntax/runtime failures or a missing final `answer`
    value raise `SolutionRuntimeError`. When false, warnings and partial output
    are returned so the editor can preview in-progress formulas.
    """
    locals_ns, rendered_lines, warnings, fatal_error = _evaluate_answer_formula(
        formula_md, params
    )
    answer_value = locals_ns.get("answer")
    if answer_value is None and strict:
        raise SolutionRuntimeError("Formula must produce an answer value.")
    if fatal_error and strict:
        raise SolutionRuntimeError(fatal_error)
    answer_md = render_template_from_values(answer_text_md, locals_ns)
    return AnswerRunResult(
        calculated_variables_md="\n".join(rendered_lines).strip(),
        answer_md=answer_md.strip(),
        final_answer=(
            _format_value(answer_value).strip() if answer_value is not None else ""
        ),
        warnings=warnings,
    )


def run_answer_function(
    python_code: str, params: dict[str, Any] | None = None
) -> AnswerRunResult:
    return run_answer_formula(python_code, params, strict=True)


_evaluate_answer_formula = evaluate_answer_formula


def _run_callable(
    python_code: str, fn_name: str, params: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    if not python_code.strip():
        raise SolutionRuntimeError("Python code is empty.")
    ns: dict[str, Any] = {}
    try:
        safe_code = normalize_python_code_string_literals(python_code)
        exec(safe_code, _safe_globals(), ns)
    except Exception as ex:
        raise SolutionRuntimeError(f"Solution compile error: {ex}") from ex
    fn = ns.get(fn_name)
    if not callable(fn):
        raise SolutionRuntimeError(
            f"Solution code must define callable {fn_name}(params)."
        )
    safe_params = params or {}
    if not isinstance(safe_params, dict):
        raise SolutionRuntimeError("Solution params must be a mapping.")
    try:
        raw = fn(safe_params)
    except Exception as ex:
        raise SolutionRuntimeError(f"Solution runtime error: {ex}") from ex
    return raw, ns


def run_distractor_function(
    python_code: str, params: dict[str, Any] | None = None
) -> DistractorRunResult:
    def _looks_explanatory(text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        cue_words = (
            "mistakenly",
            "incorrect",
            "wrong",
            "misread",
            "misreads",
            "uses",
            "treats",
            "forgets",
            "because",
            "applies",
            "directly",
        )
        return any(w in t for w in cue_words) or len(t) > 90

    def _looks_answer_like(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        if re.search(r"\d", t):
            return True
        if re.search(r"[=+\-/*^]", t):
            return True
        return len(t) <= 40 and not _looks_explanatory(t)

    raw, _ = _run_callable(python_code, "distractor", params or {})
    if not isinstance(raw, dict):
        raise SolutionRuntimeError("distractor(params) must return a dict.")
    distractor_md = raw.get("distractor_md")
    rationale = raw.get("rationale")
    if not isinstance(distractor_md, str) or not distractor_md.strip():
        raise SolutionRuntimeError(
            "distractor return must include non-empty string distractor_md."
        )
    if not isinstance(rationale, str) or not rationale.strip():
        raise SolutionRuntimeError(
            "distractor return must include non-empty string rationale."
        )
    # Repair common model error where answer/rationale are swapped.
    if _looks_explanatory(distractor_md) and _looks_answer_like(rationale):
        distractor_md, rationale = rationale, distractor_md
    return DistractorRunResult(
        distractor_md=distractor_md.strip(), rationale=rationale.strip()
    )


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
    answer_formula_md: str,
    distractor_python_codes: list[tuple[str, str]],
    params: dict[str, Any] | None = None,
) -> HarnessRunResult:
    params = params or {}
    answer = run_answer_formula(answer_formula_md, params, strict=True)
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
            return (
                1,
                0.0,
                _normalize_choice_text(str(row["content_md"])),
                str(row["source_id"]),
            )
        return (
            0,
            numeric,
            _normalize_choice_text(str(row["content_md"])),
            str(row["source_id"]),
        )

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
