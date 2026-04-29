from __future__ import annotations

from fastapi.testclient import TestClient

from exam_helper.ai_service import AIService
from exam_helper.app import create_app
from exam_helper.models import AIUsageTotals, ChatTurn
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


def test_question_editor_has_chat_workflow_hooks(tmp_path) -> None:
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
    assert '<details class="card chat-panel"' in resp.text
    assert "<summary>Chat Assistant</summary>" in resp.text
    assert 'id="chat_thread"' in resp.text
    assert 'id="chat_history_keep_count"' in resp.text
    assert 'id="chat_message"' in resp.text
    assert 'id="btn_send_chat"' in resp.text
    assert "formula-preview--compact" in resp.text
    assert "OpenAI chat is enabled." not in resp.text
    assert "Configure an OpenAI key to use chat." not in resp.text
    assert "No API key configured" in resp.text
    assert 'title="Shift+Enter to send"' in resp.text
    assert 'title="No API key configured."' in resp.text
    assert "setChatBusy(true)" in resp.text
    assert 'event.key === "Enter" && event.shiftKey' in resp.text


def test_question_editor_embeds_full_chat_history_and_window_control(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)
    client.post(
        "/questions/save",
        data={
            "question_id": "q_history",
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
    seeded = repo.get_question("q_history")
    seeded.solution.chat_history = [
        ChatTurn(user_message=f"user {idx}", assistant_message=f"assistant {idx}")
        for idx in range(1, 7)
    ]
    repo.save_question(seeded)

    resp = client.get("/questions/q_history/edit")
    assert resp.status_code == 200
    assert 'id="chat_history_keep_count"' in resp.text
    assert 'value="5"' in resp.text
    assert "Sending the last" in resp.text
    assert "Messages above this line are not sent to the LLM." in resp.text
    assert "user 1" in resp.text
    assert "user 6" in resp.text


def test_question_editor_rendering_is_stable(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)
    client.post(
        "/questions/save",
        data={
            "question_id": "q_chat",
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
    resp = client.get("/questions/q_chat/edit")
    assert resp.status_code == 200
    assert 'id="chat_thread"' in resp.text


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
                usage=AIUsageTotals(
                    input_tokens=10,
                    output_tokens=4,
                    total_tokens=14,
                    total_cost_usd=0.01,
                ),
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
    resp = client.post("/questions/q2/ai/preview/generate-answer-formula")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["system_prompt"] == "System"
