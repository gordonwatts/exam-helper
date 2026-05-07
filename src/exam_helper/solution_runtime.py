from __future__ import annotations

import ast
import math
import re
from decimal import Decimal, ROUND_HALF_UP
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
    row_previews: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    correct_answer_md: str = ""


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
            _format_value(value),
            rendered,
        )
    return rendered


def _safe_globals() -> dict[str, Any]:
    """Return the names formulas may resolve without importing anything."""
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


def _expression_name_ids(node: ast.AST) -> set[str]:
    """Collect all loaded variable names used inside a parsed expression."""
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _validate_formula_expression(expr: str, allowed_names: set[str]) -> None:
    """Reject expressions that reference names outside the evaluation scope."""
    expr_ast = ast.parse(expr, mode="eval")
    unknown = _expression_name_ids(expr_ast) - allowed_names
    if unknown:
        raise SolutionRuntimeError(
            "Unknown name(s) in formula: " + ", ".join(sorted(unknown))
        )


def _evaluate_formula_expression(
    expr: str, locals_ns: dict[str, Any], globals_ns: dict[str, Any]
) -> Any:
    """Evaluate a single formula expression with SymPy-aware parsing."""
    try:
        return ast.literal_eval(expr)
    except Exception:
        pass

    builtin_names = set()
    builtins_obj = globals_ns.get("__builtins__")
    if isinstance(builtins_obj, dict):
        builtin_names = set(builtins_obj.keys())
    allowed_names = builtin_names | {
        name for name in {**globals_ns, **locals_ns}.keys() if not name.startswith("__")
    }
    _validate_formula_expression(expr, allowed_names)
    namespace = {
        key: value for key, value in globals_ns.items() if not key.startswith("__")
    }
    if isinstance(builtins_obj, dict):
        namespace.update(builtins_obj)
    namespace.update(locals_ns)
    return sp.sympify(expr, locals=namespace, evaluate=True)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        numeric_value = _decimal_from_value(value)
        if numeric_value is not None:
            return _format_decimal_with_sig_figs(numeric_value)
        return str(value)
    if isinstance(value, sp.Basic):
        numeric_value = _decimal_from_value(value)
        if numeric_value is not None:
            return _format_decimal_with_sig_figs(numeric_value)
        return sp.sstr(value)
    return str(value)


def _decimal_from_value(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return Decimal(str(value))
    if isinstance(value, sp.Basic):
        if not value.is_number or value.is_infinite:
            return None
        if value.is_Integer:
            return Decimal(int(value))
        try:
            return Decimal(str(sp.N(value, 15)))
        except Exception:
            return None
    return None


def _format_decimal_with_sig_figs(value: Decimal, sig_figs: int = 3) -> str:
    if value.is_zero():
        return "0"

    sign = "-" if value.is_signed() else ""
    abs_value = value.copy_abs()

    if abs_value == abs_value.to_integral_value():
        return f"{sign}{format(abs_value.to_integral_value(), 'f')}"

    exponent = abs_value.adjusted()

    def _format_scientific(decimal_value: Decimal) -> str:
        scientific_exponent = decimal_value.copy_abs().adjusted()
        mantissa = decimal_value.scaleb(-scientific_exponent)
        mantissa_places = max(sig_figs - 1, 0)
        quant = Decimal(1).scaleb(-mantissa_places)
        mantissa = mantissa.quantize(quant, rounding=ROUND_HALF_UP)
        if mantissa == Decimal(10):
            mantissa = Decimal(1).quantize(quant, rounding=ROUND_HALF_UP)
            scientific_exponent += 1
        return f"{sign}{format(mantissa, 'f')}e{scientific_exponent}"

    if exponent >= 3 or exponent < -3:
        return _format_scientific(abs_value)

    decimals = max(sig_figs - exponent - 1, 0)
    quant = Decimal(1).scaleb(-decimals)
    rounded = abs_value.quantize(quant, rounding=ROUND_HALF_UP)
    if rounded.adjusted() >= 3 or rounded.adjusted() < -3:
        return _format_scientific(rounded)

    return f"{sign}{format(rounded, 'f')}"


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
    warnings: list[str] = []
    rendered_lines: list[str] = []
    last_value: Any = None
    explicit_answer_defined = False
    fatal_error: str | None = None
    globals_ns = _safe_globals()
    steps: list[tuple[str, str | None, Any]] = []

    for lineno, raw_line in enumerate(source.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            stmt = ast.parse(line, mode="exec")
        except SyntaxError as ex:
            fatal_error = f"Line {lineno}: {ex.msg}"
            warnings.append(fatal_error)
            break

        if len(stmt.body) != 1:
            fatal_error = f"Line {lineno}: only one statement is allowed per line."
            warnings.append(fatal_error)
            break

        node = stmt.body[0]
        try:
            if isinstance(node, ast.Expr):
                value = _evaluate_formula_expression(
                    ast.unparse(node.value), locals_ns, globals_ns
                )
                steps.append(("expr", None, value))
                last_value = value
                continue

            if isinstance(node, ast.Assign):
                value = _evaluate_formula_expression(
                    ast.unparse(node.value), locals_ns, globals_ns
                )
                targets = _line_targets(node)
                if not targets:
                    steps.append(("text", None, line))
                    continue
                assigned_values = [value] * len(targets)
                if len(targets) > 1 and isinstance(value, (tuple, list)):
                    if len(value) != len(targets):
                        raise SolutionRuntimeError(
                            f"Line {lineno}: unpacking count does not match targets."
                        )
                    assigned_values = list(value)
                for target, assigned_value in zip(targets, assigned_values):
                    locals_ns[target] = assigned_value
                    steps.append(("assign", target, assigned_value))
                    last_value = assigned_value
                    if target == "answer":
                        if not explicit_answer_defined:
                            warnings.append(
                                "Formula defines answer directly; it will not be overwritten."
                            )
                        explicit_answer_defined = True
                continue

            if isinstance(node, ast.AnnAssign):
                if node.value is None:
                    raise SolutionRuntimeError(
                        f"Line {lineno}: annotated assignments require a value."
                    )
                target_names = _line_targets(node)
                if not target_names:
                    raise SolutionRuntimeError(
                        f"Line {lineno}: could not resolve assignment target."
                    )
                value = _evaluate_formula_expression(
                    ast.unparse(node.value), locals_ns, globals_ns
                )
                for target in target_names:
                    locals_ns[target] = value
                    steps.append(("assign", target, value))
                    last_value = value
                    if target == "answer":
                        if not explicit_answer_defined:
                            warnings.append(
                                "Formula defines answer directly; it will not be overwritten."
                            )
                        explicit_answer_defined = True
                continue

            if isinstance(node, ast.AugAssign):
                target_names = _line_targets(node)
                if len(target_names) != 1:
                    raise SolutionRuntimeError(
                        f"Line {lineno}: augmented assignment must target one name."
                    )
                target = target_names[0]
                if target not in locals_ns:
                    raise SolutionRuntimeError(
                        f"Line {lineno}: unknown name '{target}' in augmented assignment."
                    )
                rhs_value = _evaluate_formula_expression(
                    ast.unparse(node.value), locals_ns, globals_ns
                )
                current = locals_ns[target]
                if isinstance(node.op, ast.Add):
                    value = current + rhs_value
                elif isinstance(node.op, ast.Sub):
                    value = current - rhs_value
                elif isinstance(node.op, ast.Mult):
                    value = current * rhs_value
                elif isinstance(node.op, ast.Div):
                    value = current / rhs_value
                elif isinstance(node.op, ast.FloorDiv):
                    value = current // rhs_value
                elif isinstance(node.op, ast.Mod):
                    value = current % rhs_value
                elif isinstance(node.op, ast.Pow):
                    value = current**rhs_value
                else:
                    raise SolutionRuntimeError(
                        f"Line {lineno}: unsupported augmented assignment operator."
                    )
                locals_ns[target] = value
                steps.append(("assign", target, value))
                last_value = value
                if target == "answer":
                    if not explicit_answer_defined:
                        warnings.append(
                            "Formula defines answer directly; it will not be overwritten."
                        )
                    explicit_answer_defined = True
                continue

            raise SolutionRuntimeError(
                f"Line {lineno}: unsupported statement type {type(node).__name__}."
            )
        except Exception as ex:
            fatal_error = f"Line {lineno}: {ex}"
            warnings.append(fatal_error)
            break

    for kind, target, value in steps:
        if kind == "expr":
            rendered_lines.append(f"result = {_format_value(value)}")
        elif kind == "assign" and target is not None:
            rendered_lines.append(f"{target} = {_format_value(value)}")
        elif kind == "text" and target is None:
            rendered_lines.append(str(value))

    if fatal_error is None and last_value is not None and not explicit_answer_defined:
        locals_ns["answer"] = last_value
        if steps and steps[-1][0] == "expr":
            rendered_lines[-1] = f"answer = {_format_value(last_value)}"
        elif rendered_lines:
            rendered_lines.append(f"answer = {_format_value(last_value)}")
        else:
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
    if fatal_error and strict:
        raise SolutionRuntimeError(fatal_error)
    if answer_value is None and strict:
        raise SolutionRuntimeError("Formula must produce an answer value.")
    answer_md = render_template_from_values(answer_text_md, locals_ns)
    return AnswerRunResult(
        calculated_variables_md="\n".join(rendered_lines).strip(),
        answer_md=answer_md.strip(),
        final_answer=(
            _format_value(answer_value).strip() if answer_value is not None else ""
        ),
        warnings=warnings,
    )


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
    cleaned = re.sub(r",", "", value or "").strip()
    match = re.match(
        r"^([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(?:\b|$)", cleaned
    )
    if not match:
        return None
    try:
        return float(match.group(1))
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
            "warning": "",
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
                "warning": "",
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
    return HarnessRunResult(
        choices=choices,
        collisions=collisions,
        row_previews=rows,
        correct_answer_md=answer.final_answer,
    )


def run_mc_formula_harness(
    answer_formula_md: str,
    mc_answer_specs: list[Any],
    params: dict[str, Any] | None = None,
    *,
    strict: bool = True,
) -> HarnessRunResult:
    """Evaluate formula-based MC rows and build preview choices.

    The first choice is derived from the main answer formula. Remaining rows
    are evaluated from `mc_answer_specs` in input order and are treated as
    distractors. When `strict` is false, blank or invalid rows are skipped and
    reported in the warnings list instead of aborting the whole preview.
    """
    params = params or {}
    warnings: list[str] = []
    row_previews: list[dict[str, Any]] = []

    answer_locals, _, answer_warnings, answer_fatal_error = evaluate_answer_formula(
        answer_formula_md, params
    )
    answer_value = answer_locals.get("answer")
    if answer_fatal_error:
        message = f"answer: {answer_fatal_error}"
        warnings.append(message)
        if strict:
            raise SolutionRuntimeError(message)
    if answer_value is None:
        message = "answer: formula did not produce an answer."
        warnings.append(message)
        if strict:
            raise SolutionRuntimeError(message)

    rows: list[MCChoice] = []
    if answer_value is not None:
        rendered_answer = _strip_disallowed_bold(_format_value(answer_value))
        row_previews.append(
            {
                "source_id": "answer",
                "content_md": rendered_answer,
                "is_correct": True,
                "rationale": "correct answer",
                "warning": " | ".join(answer_warnings).strip(),
            }
        )
        rows.append(
            MCChoice(
                label="?",
                content_md=rendered_answer,
                is_correct=True,
                rationale="correct answer",
            )
        )

    seed_locals = dict(params)
    seed_locals.update(answer_locals)

    def _evaluate_row(
        *, source_id: str, formula_md: str, rationale_md: str, is_correct: bool
    ) -> MCChoice | None:
        formula = (formula_md or "").strip()
        if not formula:
            message = f"{source_id}: formula is blank."
            if strict:
                raise SolutionRuntimeError(message)
            warnings.append(message)
            row_previews.append(
                {
                    "source_id": source_id,
                    "content_md": "",
                    "is_correct": is_correct,
                    "rationale": rationale_md.strip(),
                    "warning": message,
                }
            )
            return None

        locals_ns, _, row_warnings, fatal_error = evaluate_answer_formula(
            formula, seed_locals
        )
        answer_md = _format_value(locals_ns.get("answer", "")).strip()
        if fatal_error:
            message = f"{source_id}: {fatal_error}"
            if strict:
                raise SolutionRuntimeError(message)
            warnings.append(message)
            row_previews.append(
                {
                    "source_id": source_id,
                    "content_md": "",
                    "is_correct": is_correct,
                    "rationale": rationale_md.strip(),
                    "warning": message,
                }
            )
            return None
        if not answer_md:
            message = f"{source_id}: formula did not produce an answer."
            if strict:
                raise SolutionRuntimeError(message)
            warnings.append(message)
            row_previews.append(
                {
                    "source_id": source_id,
                    "content_md": "",
                    "is_correct": is_correct,
                    "rationale": rationale_md.strip(),
                    "warning": message,
                }
            )
            return None
        rendered = _strip_disallowed_bold(answer_md)
        row_previews.append(
            {
                "source_id": source_id,
                "content_md": rendered,
                "is_correct": is_correct,
                "rationale": rationale_md.strip(),
                "warning": " | ".join(row_warnings).strip(),
            }
        )
        return MCChoice(
            label="?",
            content_md=rendered,
            is_correct=is_correct,
            rationale=rationale_md.strip() or ("correct answer" if is_correct else ""),
        )

    for idx, spec in enumerate(mc_answer_specs, start=1):
        formula_md = (
            spec.get("formula_md", "")
            if isinstance(spec, dict)
            else getattr(spec, "formula_md", "")
        )
        rationale_md = (
            spec.get("rationale_md", "")
            if isinstance(spec, dict)
            else getattr(spec, "rationale_md", "")
        )
        row = _evaluate_row(
            source_id=f"choice_{idx}",
            formula_md=formula_md,
            rationale_md=rationale_md,
            is_correct=False,
        )
        if row is not None:
            rows.append(row)

    seen: dict[str, str] = {}
    collisions: list[str] = []
    for row in row_previews:
        content_md = str(row.get("content_md", ""))
        if not content_md.strip():
            continue
        canonical = _normalize_choice_text(content_md)
        source_id = str(row.get("source_id", ""))
        if canonical in seen:
            collisions.append(
                f"Duplicate MC option between '{seen[canonical]}' and '{source_id}': {content_md}"
            )
        else:
            seen[canonical] = source_id

    def _sort_tuple(row: MCChoice) -> tuple[int, float, str, str]:
        numeric = _numeric_sort_key(str(row.content_md))
        if numeric is None:
            return (
                1,
                0.0,
                _normalize_choice_text(str(row.content_md)),
                "answer" if row.is_correct else str(row.content_md),
            )
        return (
            0,
            numeric,
            _normalize_choice_text(str(row.content_md)),
            "answer" if row.is_correct else str(row.content_md),
        )

    rows.sort(key=_sort_tuple)
    labels = ["A", "B", "C", "D", "E"]
    choices: list[MCChoice] = []
    for idx, row in enumerate(rows):
        choices.append(
            MCChoice(
                label=labels[idx] if idx < len(labels) else "?",
                content_md=row.content_md,
                is_correct=row.is_correct,
                rationale=row.rationale,
            )
        )
    return HarnessRunResult(
        choices=choices,
        collisions=collisions,
        row_previews=row_previews,
        warnings=warnings,
        correct_answer_md=(
            _format_value(answer_value).strip() if answer_value is not None else ""
        ),
    )
