from __future__ import annotations

import pytest

from exam_helper.solution_runtime import SolutionRuntimeError, run_solution_code


def test_solution_runtime_success_free_response() -> None:
    code = (
        "def solve(params, context):\n"
        "    v = float(params.get('v', 0))\n"
        "    return {'final_answer_text': f'Final answer: {v:.1f} m/s'}\n"
    )
    result = run_solution_code(code, {"v": 12}, {})
    assert "Final answer: 12.0" in result.final_answer_text
    assert result.choices_yaml is None


def test_solution_runtime_success_mc_yaml() -> None:
    code = (
        "def solve(params, context):\n"
        "    return {\n"
        "      'final_answer_text': 'Final answer: 2 m/s^2',\n"
        "      'choices_yaml': '- label: A\\n  content_md: 1\\n  is_correct: false\\n"
        "- label: B\\n  content_md: 2\\n  is_correct: true\\n"
        "- label: C\\n  content_md: 3\\n  is_correct: false\\n"
        "- label: D\\n  content_md: 4\\n  is_correct: false\\n"
        "- label: E\\n  content_md: 5\\n  is_correct: false\\n'\n"
        "    }\n"
    )
    result = run_solution_code(code, {}, {})
    assert "Final answer" in result.final_answer_text
    assert result.choices_yaml is not None


def test_solution_runtime_requires_solve() -> None:
    with pytest.raises(SolutionRuntimeError, match="must define callable solve"):
        run_solution_code("x = 1", {}, {})


def test_solution_runtime_requires_non_empty_final_answer_text() -> None:
    code = (
        "def solve(params, context):\n"
        "    return {'final_answer_text': ''}\n"
    )
    with pytest.raises(SolutionRuntimeError, match="final_answer_text"):
        run_solution_code(code, {}, {})
