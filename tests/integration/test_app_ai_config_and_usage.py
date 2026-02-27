from __future__ import annotations

from fastapi.testclient import TestClient

from exam_helper.ai_service import AIService
from exam_helper.app import create_app
from exam_helper.models import AIUsageTotals
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


def test_question_editor_has_new_workflow_hooks(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)
    client.post(
        "/questions/save",
        data={
            "question_id": "q_edit",
            "title": "T",
            "question_type": "free_response",
            "prompt_md": "P",
            "question_template_md": "P",
            "choices_yaml": "[]",
            "typed_solution_md": "",
            "distractor_functions_text": "",
            "figures_json": "[]",
            "points": 5,
        },
    )
    resp = client.get("/questions/q_edit/edit")
    assert resp.status_code == 200
    assert 'id="btn_rewrite"' in resp.text
    assert 'id="btn_generate_answer"' in resp.text
    assert 'id="btn_generate_typed_solution"' in resp.text


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
            "question_template_md": "P",
            "choices_yaml": "[]",
            "distractor_functions_text": "",
            "typed_solution_md": "",
            "figures_json": "[]",
            "points": 5,
        },
    )

    class _AI:
        def rewrite_parameterize(self, question):
            return AIService.RewriteResult(
                question_template_md="Find v={{v}}",
                parameters={"v": 12},
                title="",
                usage=AIUsageTotals(input_tokens=10, output_tokens=4, total_tokens=14, total_cost_usd=0.01),
            )

    app.state.ai = _AI()
    assert client.post("/questions/q1/ai/rewrite-and-parameterize").status_code == 200

    project = repo.load_project()
    assert project.ai.usage.total_tokens == 14
    assert abs(project.ai.usage.total_cost_usd - 0.01) < 1e-9

    reset = client.post("/project/usage/reset", follow_redirects=False)
    assert reset.status_code == 303
    project_after = repo.load_project()
    assert project_after.ai.usage.total_tokens == 0


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
            "question_template_md": "P",
            "choices_yaml": "[]",
            "distractor_functions_text": "",
            "typed_solution_md": "",
            "figures_json": "[]",
            "points": 5,
        },
    )

    class _PreviewAI:
        def preview_prompt(self, action, question):
            return {
                "action": action,
                "system_prompt": "System",
                "user_prompt": "User",
                "figure_placeholders": ["<figure fig_1>"],
            }

    app.state.ai = _PreviewAI()
    resp = client.post("/questions/q2/ai/preview/generate-answer-function")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["system_prompt"] == "System"
