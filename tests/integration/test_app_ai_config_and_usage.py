from __future__ import annotations

from fastapi.testclient import TestClient

from exam_helper.ai_service import AIService
from exam_helper.app import create_app
from exam_helper.models import AIUsageTotals, MCChoice
from exam_helper.repository import ProjectRepository


def test_home_shows_model_and_usage(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics", openai_model="gpt-5.2")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "model:" in resp.text
    assert "gpt-5.2" in resp.text
    assert "total" in resp.text


def test_home_has_title_markdown_render_hooks(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)
    client.post(
        "/questions/save",
        data={
            "question_id": "q_title",
            "title": "Speed is $v+1$",
            "question_type": "free_response",
            "prompt_md": "P",
            "choices_yaml": "[]",
            "solution_md": "",
            "figures_json": "[]",
            "points": 5,
        },
    )
    resp = client.get("/")
    assert resp.status_code == 200
    assert 'class="title-markdown"' in resp.text
    assert "data-title-markdown=" in resp.text
    assert "function renderTitleMarkdownCells()" in resp.text
    assert "renderTitleMarkdownCells();" in resp.text


def test_question_editor_has_title_preview_hooks(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)
    client.post(
        "/questions/save",
        data={
            "question_id": "q_edit",
            "title": "Speed is $v+1$",
            "question_type": "free_response",
            "prompt_md": "P",
            "choices_yaml": "[]",
            "solution_md": "",
            "figures_json": "[]",
            "points": 5,
        },
    )
    resp = client.get("/questions/q_edit/edit")
    assert resp.status_code == 200
    assert 'id="title_preview"' in resp.text
    assert "function renderTitlePreview()" in resp.text
    assert "renderTitlePreview();" in resp.text


def test_project_settings_persist_model_and_prompts(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)
    resp = client.post(
        "/project/settings",
        data={
            "openai_model": "gpt-5.2-pro",
            "prompt_overall": "Keep units explicit.",
            "prompt_solution_and_mc": "Use one numeric check.",
            "prompt_prompt_review": "Simplify wording.",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    updated = repo.load_project()
    assert updated.ai.model == "gpt-5.2-pro"
    assert updated.ai.prompts.overall == "Keep units explicit."
    assert updated.ai.prompts.solution_and_mc == "Use one numeric check."
    assert updated.ai.prompts.prompt_review == "Simplify wording."


def test_usage_totals_accumulate_and_reset(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    client.post(
        "/questions/save",
        data={
            "question_id": "q1",
            "title": "T",
            "question_type": "free_response",
            "prompt_md": "P",
            "choices_yaml": "[]",
            "solution_md": "",
            "figures_json": "[]",
            "points": 5,
        },
    )

    class _AI:
        def improve_prompt(self, question):
            return AIService.AIResult(
                text="Improved",
                usage=AIUsageTotals(input_tokens=10, output_tokens=4, total_tokens=14, total_cost_usd=0.01),
            )

        def draft_solution(self, question):
            return AIService.AIResult(
                text="Sol",
                usage=AIUsageTotals(input_tokens=20, output_tokens=8, total_tokens=28, total_cost_usd=0.02),
            )

        def suggest_title(self, question):
            return "Title"

        def generate_mc_options_from_solution(self, question, solution_md):
            return (
                [
                    MCChoice(label="A", content_md="a", is_correct=True),
                    MCChoice(label="B", content_md="b", is_correct=False),
                    MCChoice(label="C", content_md="c", is_correct=False),
                    MCChoice(label="D", content_md="d", is_correct=False),
                    MCChoice(label="E", content_md="e", is_correct=False),
                ],
                AIUsageTotals(input_tokens=30, output_tokens=10, total_tokens=40, total_cost_usd=0.03),
            )

        def preview_prompt(self, action, question, solution_md=""):
            return {
                "action": action,
                "system_prompt": "S",
                "user_prompt": "U",
                "figure_placeholders": [],
            }

    app.state.ai = _AI()
    assert client.post("/questions/q1/ai/improve-prompt").status_code == 200
    assert client.post("/questions/q1/ai/draft-solution").status_code == 200

    project = repo.load_project()
    assert project.ai.usage.input_tokens == 30
    assert project.ai.usage.output_tokens == 12
    assert project.ai.usage.total_tokens == 42
    assert abs(project.ai.usage.total_cost_usd - 0.03) < 1e-9

    reset = client.post("/project/usage/reset", follow_redirects=False)
    assert reset.status_code == 303
    project_after = repo.load_project()
    assert project_after.ai.usage.total_tokens == 0
    assert project_after.ai.usage.total_cost_usd == 0.0


def test_prompt_preview_endpoint_returns_composed_payload(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    client.post(
        "/questions/save",
        data={
            "question_id": "q2",
            "title": "T",
            "question_type": "free_response",
            "prompt_md": "P",
            "choices_yaml": "[]",
            "solution_md": "",
            "figures_json": "[]",
            "points": 5,
        },
    )

    class _PreviewAI:
        def preview_prompt(self, action, question, solution_md=""):
            return {
                "action": action,
                "system_prompt": "System",
                "user_prompt": "User",
                "figure_placeholders": ["<figure fig_1>"],
            }

    app.state.ai = _PreviewAI()
    resp = client.post("/questions/q2/ai/preview/draft-solution")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["system_prompt"] == "System"
    assert payload["figure_placeholders"] == ["<figure fig_1>"]


def test_openai_models_endpoint_returns_list(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)

    class _AIModels:
        def list_models(self):
            return ["gpt-5.2", "gpt-5.2-mini"]

    app.state.ai = _AIModels()
    resp = client.get("/openai/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["models"] == ["gpt-5.2", "gpt-5.2-mini"]
