from __future__ import annotations

import warnings

import pytest

from exam_helper.solution_runtime import (
    SolutionRuntimeError,
    evaluate_answer_formula,
    run_answer_formula,
    run_mc_formula_harness,
    run_distractor_function,
    run_mc_harness,
)


def test_evaluate_answer_formula_returns_raw_evaluation_state() -> None:
    locals_ns, rendered_lines, warnings, fatal_error = evaluate_answer_formula(
        "x = v\ny = x + 2\nanswer = y", {"v": 5}
    )
    assert locals_ns["x"] == 5
    assert locals_ns["y"] == 7
    assert locals_ns["answer"] == 7
    assert rendered_lines == ["x = 5", "y = 7", "answer = 7"]
    assert warnings == ["Formula defines answer directly; it will not be overwritten."]
    assert fatal_error is None


def test_evaluate_answer_formula_reports_fatal_errors() -> None:
    locals_ns, rendered_lines, warnings, fatal_error = evaluate_answer_formula(
        "x = 1\nanswer = missing_name", {}
    )
    assert locals_ns["x"] == 1
    assert "answer" not in locals_ns
    assert rendered_lines == ["x = 1"]
    assert warnings
    assert fatal_error is not None


def test_answer_formula_success() -> None:
    code = "v = float(v)\nanswer = v"
    result = run_answer_formula(
        code, {"v": 2.5}, answer_text_md="v={{v}} m/s", strict=True
    )
    assert result.calculated_variables_md == "v = 2.5\nanswer = 2.5"
    assert result.answer_md == "v=2.5 m/s"
    assert result.final_answer == "2.5"


def test_answer_formula_requires_nonempty_formula() -> None:
    with pytest.raises(SolutionRuntimeError, match="Formula text is empty"):
        run_answer_formula("", {})


def test_answer_formula_tracks_last_expression() -> None:
    result = run_answer_formula("x = 1\nx + 2", {}, strict=True)
    assert result.calculated_variables_md == "x = 1\nanswer = 3"
    assert result.final_answer == "3"


def test_answer_formula_does_not_backfill_answer_after_a_fatal_error() -> None:
    locals_ns, rendered_lines, warnings, fatal_error = evaluate_answer_formula(
        "x = 1\nanswer = missing_name", {}
    )
    assert locals_ns["x"] == 1
    assert "answer" not in locals_ns
    assert rendered_lines == ["x = 1"]
    assert warnings
    assert fatal_error is not None


def test_answer_formula_warns_when_answer_is_defined_directly() -> None:
    result = run_answer_formula("answer = 5", {}, strict=True)
    assert result.final_answer == "5"
    assert result.warnings


def test_distractor_function_success() -> None:
    code = (
        "def distractor(params):\n"
        "    return {'distractor_md': '3.0 m/s', 'rationale': 'forgot sign'}\n"
    )
    result = run_distractor_function(code, {})
    assert result.distractor_md == "3.0 m/s"
    assert result.rationale == "forgot sign"


def test_distractor_function_repairs_swapped_answer_and_rationale() -> None:
    code = (
        "def distractor(params):\n"
        "    return {\n"
        "      'distractor_md': 'Mistakenly applies inverse scaling between p and T.',\n"
        "      'rationale': '47.3 C'\n"
        "    }\n"
    )
    result = run_distractor_function(code, {})
    assert result.distractor_md == "47.3 C"
    assert "mistakenly applies inverse scaling" in result.rationale.lower()


def test_harness_detects_collisions() -> None:
    answer_code = "answer = '2 m/s'"
    distractor_code = (
        "def distractor(params):\n"
        "    return {'distractor_md': '2 m/s', 'rationale': 'duplicate'}\n"
    )
    out = run_mc_harness(
        answer_code,
        [
            ("d1", distractor_code),
            ("d2", distractor_code),
            ("d3", distractor_code),
            ("d4", distractor_code),
        ],
        {},
    )
    assert out.collisions


def test_harness_sorts_numeric_then_text_tiebreak_by_source() -> None:
    answer_code = "answer = '10 m/s'"
    d1 = "def distractor(params):\n    return {'distractor_md': '2 m/s', 'rationale': 'r'}\n"
    d2 = "def distractor(params):\n    return {'distractor_md': '20 m/s', 'rationale': 'r'}\n"
    d3 = "def distractor(params):\n    return {'distractor_md': 'alpha', 'rationale': 'r'}\n"
    d4 = "def distractor(params):\n    return {'distractor_md': 'beta', 'rationale': 'r'}\n"
    out = run_mc_harness(
        answer_code, [("d1", d1), ("d2", d2), ("d3", d3), ("d4", d4)], {}
    )
    assert [c.label for c in out.choices] == ["A", "B", "C", "D", "E"]
    assert [c.content_md for c in out.choices] == [
        "2 m/s",
        "10 m/s",
        "20 m/s",
        "alpha",
        "beta",
    ]


def test_mc_formula_harness_uses_answer_formula_for_the_correct_choice() -> None:
    out = run_mc_formula_harness(
        "answer = 2",
        [
            {"formula_md": "answer = 3", "rationale_md": "off by one"},
            {"formula_md": "answer = 4", "rationale_md": "off by two"},
            {"formula_md": "answer = 5", "rationale_md": "off by three"},
            {"formula_md": "answer = 6", "rationale_md": "off by four"},
        ],
        {},
        strict=True,
    )
    assert out.correct_answer_md == "2"
    assert [c.content_md for c in out.choices] == ["2", "3", "4", "5", "6"]
    assert out.row_previews[0]["content_md"] == "2"
    assert out.row_previews[1]["content_md"] == "3"


def test_answer_formula_latex_sequences_do_not_emit_invalid_escape_warnings() -> None:
    code = "answer = 1"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_answer_formula(code, {}, answer_text_md=r"$\theta$")
    assert result.answer_md == r"$\theta$"
    assert result.final_answer == "1"
    assert not [w for w in caught if "invalid escape sequence" in str(w.message)]


def test_answer_formula_preserves_existing_double_escaped_latex() -> None:
    out = run_answer_formula("answer = 1", {}, answer_text_md=r"$\\theta$")
    assert out.answer_md == r"$\\theta$"
    assert out.final_answer == "1"
