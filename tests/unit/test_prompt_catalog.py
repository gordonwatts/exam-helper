from __future__ import annotations

from exam_helper.models import AIPromptConfig, Question
from exam_helper.prompt_catalog import PromptCatalog


def test_prompt_catalog_builds_rewrite_prompt() -> None:
    catalog = PromptCatalog.from_package_yaml()
    q = Question(id="q1", title="", prompt_md="Find v")
    q.solution.question_template_md = "A cart moves at {{v}} m/s."
    q.solution.parameters = {"v": 12}
    bundle = catalog.compose(action="rewrite_parameterize", question=q)
    assert "strict JSON object" in bundle.system_prompt
    assert "Rendered Question (Markdown):" in bundle.user_prompt
    assert "Template Parameters (YAML):" in bundle.user_prompt
    assert "```markdown" in bundle.user_prompt
    assert "```yaml" in bundle.user_prompt
    assert "Current prompt" not in bundle.user_prompt


def test_prompt_catalog_applies_overall_and_rewrite_override() -> None:
    catalog = PromptCatalog.from_package_yaml()
    q = Question(id="q1", title="T", prompt_md="P")
    bundle = catalog.compose(
        action="rewrite_parameterize",
        question=q,
        prompts_override=AIPromptConfig(
            overall="Always keep SI units.",
            prompt_review="Prefer minimal wording edits.",
        ),
    )
    assert "Always keep SI units." in bundle.system_prompt
    assert "Prefer minimal wording edits." in bundle.system_prompt


def test_prompt_catalog_applies_solution_and_mc_override_to_answer_generation() -> None:
    catalog = PromptCatalog.from_package_yaml()
    q = Question(id="q1", title="T", prompt_md="P")
    bundle = catalog.compose(
        action="generate_answer_function",
        question=q,
        prompts_override=AIPromptConfig(solution_and_mc="Keep units explicit in final_answer."),
    )
    assert "Keep units explicit in final_answer." in bundle.system_prompt


def test_prompt_catalog_omits_empty_old_code_sections() -> None:
    catalog = PromptCatalog.from_package_yaml()
    q = Question(id="q1", title="T", prompt_md="P")
    q.solution.answer_python_code = "def solve(params): return {'answer_md':'1','final_answer':'1'}"
    q.solution.distractor_python_code = []
    bundle = catalog.compose(action="generate_distractor_functions", question=q)
    assert "Distractor Functions (Python):" not in bundle.user_prompt
    assert "Answer Function (Python):" in bundle.user_prompt
