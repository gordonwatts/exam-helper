from __future__ import annotations

import pytest

from exam_helper.solution_runtime import (
    SolutionRuntimeError,
    run_answer_function,
    run_distractor_function,
    run_mc_harness,
)


def test_answer_function_success() -> None:
    code = (
        "def solve(params):\n"
        "    v = float(params.get('v', 1.0))\n"
        "    return {'answer_md': f'v={v:.1f} m/s', 'final_answer': f'{v:.1f} m/s'}\n"
    )
    result = run_answer_function(code, {"v": 2.5})
    assert result.answer_md == "v=2.5 m/s"
    assert result.final_answer == "2.5 m/s"


def test_answer_function_requires_solve() -> None:
    with pytest.raises(SolutionRuntimeError, match="must define callable solve"):
        run_answer_function("x = 1", {})


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
    answer_code = (
        "def solve(params):\n"
        "    return {'answer_md': 'Answer is 2 m/s', 'final_answer': '2 m/s'}\n"
    )
    distractor_code = (
        "def distractor(params):\n"
        "    return {'distractor_md': '2 m/s', 'rationale': 'duplicate'}\n"
    )
    out = run_mc_harness(
        answer_code,
        [("d1", distractor_code), ("d2", distractor_code), ("d3", distractor_code), ("d4", distractor_code)],
        {},
    )
    assert out.collisions


def test_harness_sorts_numeric_then_text_tiebreak_by_source() -> None:
    answer_code = (
        "def solve(params):\n"
        "    return {'answer_md': 'Ans', 'final_answer': '10 m/s'}\n"
    )
    d1 = "def distractor(params):\n    return {'distractor_md': '2 m/s', 'rationale': 'r'}\n"
    d2 = "def distractor(params):\n    return {'distractor_md': '20 m/s', 'rationale': 'r'}\n"
    d3 = "def distractor(params):\n    return {'distractor_md': 'alpha', 'rationale': 'r'}\n"
    d4 = "def distractor(params):\n    return {'distractor_md': 'beta', 'rationale': 'r'}\n"
    out = run_mc_harness(answer_code, [("d1", d1), ("d2", d2), ("d3", d3), ("d4", d4)], {})
    assert [c.label for c in out.choices] == ["A", "B", "C", "D", "E"]
    assert [c.content_md for c in out.choices] == ["2 m/s", "10 m/s", "20 m/s", "alpha", "beta"]
