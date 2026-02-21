from __future__ import annotations

from exam_helper.models import AIPromptConfig, Question
from exam_helper.prompt_catalog import PromptCatalog


def test_prompt_catalog_uses_base_prompts_when_overrides_empty() -> None:
    catalog = PromptCatalog.from_package_yaml()
    q = Question(id="q1", title="T", prompt_md="P")
    bundle = catalog.compose(
        action="draft_solution",
        question=q,
        prompts_override=AIPromptConfig(overall="   ", solution_and_mc=""),
    )
    assert "write brief but complete worked solutions" in bundle.system_prompt.lower()


def test_prompt_catalog_applies_overall_and_scoped_overrides() -> None:
    catalog = PromptCatalog.from_package_yaml()
    q = Question(id="q1", title="T", prompt_md="P")
    bundle = catalog.compose(
        action="improve_prompt",
        question=q,
        prompts_override=AIPromptConfig(
            overall="Always keep SI units.",
            prompt_review="Make wording beginner-friendly.",
        ),
    )
    assert "Always keep SI units." in bundle.system_prompt
    assert "Make wording beginner-friendly." in bundle.system_prompt


def test_figure_placeholders_show_ids_and_captions() -> None:
    q = Question.model_validate(
        {
            "id": "qf",
            "prompt_md": "P",
            "figures": [
                {
                    "id": "fig_1",
                    "mime_type": "image/png",
                    "data_base64": "YQ==",
                    "sha256": "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
                    "caption": "Free-body diagram",
                }
            ],
        }
    )
    placeholders = PromptCatalog.figure_placeholders(q)
    assert placeholders == ["<figure fig_1: Free-body diagram>"]


def test_draft_solution_prompt_includes_existing_solution_text() -> None:
    catalog = PromptCatalog.from_package_yaml()
    q = Question.model_validate(
        {
            "id": "q_sol",
            "title": "Kinematics",
            "prompt_md": "Find velocity.",
            "solution": {"worked_solution_md": "Use v = v0 + at; Final answer: 12 m/s"},
        }
    )
    bundle = catalog.compose(action="draft_solution", question=q)
    assert "Current worked solution draft" in bundle.user_prompt
    assert "Final answer: 12 m/s" in bundle.user_prompt


def test_distractors_prompt_includes_existing_choices_yaml() -> None:
    catalog = PromptCatalog.from_package_yaml()
    q = Question.model_validate(
        {
            "id": "q_mc",
            "prompt_md": "Find acceleration.",
            "question_type": "multiple_choice",
            "mc_options_guidance": "Avoid using the previous correct answer as a distractor.",
            "choices": [
                {"label": "A", "content_md": "1 m/s^2", "is_correct": False},
                {"label": "B", "content_md": "2 m/s^2", "is_correct": True},
            ],
        }
    )
    bundle = catalog.compose(action="distractors", question=q, solution_md="Final answer: 2 m/s^2")
    assert "Author guidance for distractors:" in bundle.user_prompt
    assert "Current MC options draft (may be blank; revise or replace as needed):" in bundle.user_prompt
    assert "Avoid using the previous correct answer as a distractor." in bundle.user_prompt
    assert "label: A" in bundle.user_prompt
    assert "content_md: 2 m/s^2" in bundle.user_prompt
