from __future__ import annotations

import pytest

from exam_helper.solution_runtime import SolutionRuntimeError, run_solution_code


def test_solution_runtime_success_free_response() -> None:
    code = (
        "def solve(params, context):\n"
        "    v = float(params.get('v', 0))\n"
        "    return {'computed_output_md': f'Final answer: {v:.1f} m/s'}\n"
    )
    result = run_solution_code(code, {"v": 12}, {})
    assert "Final answer: 12.0" in result.computed_output_md
    assert result.choices_yaml is None


def test_solution_runtime_success_mc_yaml() -> None:
    code = (
        "def solve(params, context):\n"
        "    return {\n"
        "      'computed_output_md': 'Final answer: 2 m/s^2',\n"
        "      'choices_yaml': '- label: A\\n  content_md: 1\\n  is_correct: false\\n"
        "- label: B\\n  content_md: 2\\n  is_correct: true\\n"
        "- label: C\\n  content_md: 3\\n  is_correct: false\\n"
        "- label: D\\n  content_md: 4\\n  is_correct: false\\n"
        "- label: E\\n  content_md: 5\\n  is_correct: false\\n'\n"
        "    }\n"
    )
    result = run_solution_code(code, {}, {})
    assert "Final answer" in result.computed_output_md
    assert result.choices_yaml is not None


def test_solution_runtime_requires_solve() -> None:
    with pytest.raises(SolutionRuntimeError, match="must define callable solve"):
        run_solution_code("x = 1", {}, {})


def test_solution_runtime_requires_non_empty_computed_output() -> None:
    code = (
        "def solve(params, context):\n"
        "    return {'computed_output_md': ''}\n"
    )
    with pytest.raises(SolutionRuntimeError, match="computed_output_md"):
        run_solution_code(code, {}, {})
