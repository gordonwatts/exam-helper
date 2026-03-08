from __future__ import annotations

from exam_helper.normalization import (
    normalize_markdown_math_delimiters,
    normalize_python_code_string_literals,
)


def test_normalize_markdown_math_delimiters_paren_and_bracket() -> None:
    raw = r"Inline \\(x^2\\) and block \\[E=mc^2\\]."
    out = normalize_markdown_math_delimiters(raw)
    assert out == "Inline $x^2$ and block $$E=mc^2$$."


def test_normalize_markdown_math_delimiters_idempotent_for_dollars() -> None:
    raw = "Already $x$ and $$y$$"
    assert normalize_markdown_math_delimiters(raw) == raw


def test_normalize_python_code_string_literals_preserves_latex_backslashes() -> None:
    code = (
        "def solve(params):\n"
        "    return {'answer_md': '$\\theta$', 'final_answer': '$\\lambda$'}\n"
    )
    out = normalize_python_code_string_literals(code)
    assert "\\\\theta" in out
    assert "\\\\lambda" in out


def test_normalize_python_code_string_literals_keeps_escaped_sequences() -> None:
    code = (
        "def solve(params):\n"
        "    return {'answer_md': '$\\\\theta$', 'final_answer': '$\\\\lambda$'}\n"
    )
    assert normalize_python_code_string_literals(code) == code


def test_normalize_python_code_string_literals_does_not_modify_non_strings() -> None:
    code = "def solve(params):\n    value = 2 + 3\n    return {'answer_md': str(value), 'final_answer': str(value)}\n"
    assert normalize_python_code_string_literals(code) == code
