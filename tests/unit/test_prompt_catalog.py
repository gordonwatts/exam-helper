from __future__ import annotations

from exam_helper.models import AIPromptConfig, Question
from exam_helper.prompt_catalog import PromptCatalog


def test_prompt_catalog_builds_rewrite_prompt() -> None:
    catalog = PromptCatalog.from_package_yaml()
    q = Question(id="q1", title="", prompt_md="Find v")
    bundle = catalog.compose(action="rewrite_parameterize", question=q)
    assert "strict JSON object" in bundle.system_prompt
    assert "Current prompt" in bundle.user_prompt


def test_prompt_catalog_applies_overall_override() -> None:
    catalog = PromptCatalog.from_package_yaml()
    q = Question(id="q1", title="T", prompt_md="P")
    bundle = catalog.compose(
        action="generate_answer_function",
        question=q,
        prompts_override=AIPromptConfig(overall="Always keep SI units."),
    )
    assert "Always keep SI units." in bundle.system_prompt


def test_prompt_catalog_includes_distractor_yaml_context() -> None:
    catalog = PromptCatalog.from_package_yaml()
    q = Question(id="q1", title="T", prompt_md="P")
    q.solution.answer_python_code = "def solve(params): return {'answer_md':'1','final_answer':'1'}"
    q.solution.distractor_python_code = []
    bundle = catalog.compose(action="generate_distractor_functions", question=q)
    assert "Current distractor code blocks:" in bundle.user_prompt
