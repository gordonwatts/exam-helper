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
    assert "under 10 words" in bundle.system_prompt
    assert "<Law or Theorem Name>: ..." in bundle.system_prompt
    assert "should not restate the full problem statement" in bundle.system_prompt
    assert "Rendered Question (Markdown):" in bundle.user_prompt
    assert "Template Parameters (YAML):" in bundle.user_prompt
    assert "```markdown" in bundle.user_prompt
    assert "```yaml" in bundle.user_prompt
    assert "Current prompt" not in bundle.user_prompt


def test_prompt_catalog_builds_answer_formula_prompt() -> None:
    catalog = PromptCatalog.from_package_yaml()
    q = Question(id="q1", title="T", prompt_md="P")
    q.solution.answer_formula_md = "answer = 1"
    q.solution.answer_guidance = "Use the final value."
    bundle = catalog.compose(action="generate_answer_formula", question=q)
    assert "Answer Formula (SymPy):" in bundle.user_prompt
    assert "Answer Text (Markdown):" in bundle.user_prompt


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
        action="generate_answer_formula",
        question=q,
        prompts_override=AIPromptConfig(
            solution_and_mc="Keep units explicit in final_answer."
        ),
    )
    assert "Keep units explicit in final_answer." in bundle.system_prompt


def test_prompt_catalog_omits_empty_old_code_sections() -> None:
    catalog = PromptCatalog.from_package_yaml()
    q = Question(id="q1", title="T", prompt_md="P")
    q.solution.answer_formula_md = "answer = 1"
    q.solution.distractor_python_code = []
    bundle = catalog.compose(action="generate_distractor_functions", question=q)
    assert "Distractor Functions (Python):" not in bundle.user_prompt
    assert "MC Distractor Guidance (Markdown):" not in bundle.user_prompt
    assert "Answer Formula (SymPy):" in bundle.user_prompt


def test_prompt_catalog_includes_mc_distractor_guidance_when_present() -> None:
    catalog = PromptCatalog.from_package_yaml()
    q = Question(id="q1", title="T", prompt_md="P")
    q.solution.answer_formula_md = "answer = 1"
    q.mc_options_guidance = "Prefer unit-conversion mistakes over algebra mistakes."
    bundle = catalog.compose(action="generate_distractor_functions", question=q)
    assert "MC Distractor Guidance (Markdown):" in bundle.user_prompt
    assert "Prefer unit-conversion mistakes" in bundle.user_prompt
