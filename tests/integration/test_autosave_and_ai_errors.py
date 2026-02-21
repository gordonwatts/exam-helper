from __future__ import annotations

import json

from fastapi.testclient import TestClient

from exam_helper.app import create_app
from exam_helper.repository import ProjectRepository


def test_autosave_persists_question(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)

    payload = {
        "title": "",
        "question_type": "free_response",
        "prompt_md": "Find v(t)",
        "choices_yaml": "[]",
        "solution_md": "Use kinematics",
        "checker_code": "",
        "figures_json": "[]",
        "points": 5,
    }
    resp = client.post("/questions/q_auto/autosave", json=payload)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    saved = repo.get_question("q_auto")
    assert saved.prompt_md == "Find v(t)"
    assert saved.title == ""


def test_ai_mc_endpoint_returns_422_on_parse_error(tmp_path) -> None:
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
            "checker_code": "",
            "figures_json": "[]",
            "points": 5,
        },
    )

    class _BadAI:
        def generate_mc_options(self, question):
            raise ValueError("bad json")

    app.state.ai = _BadAI()
    resp = client.post("/questions/q1/ai/distractors")
    assert resp.status_code == 422
    assert resp.json()["ok"] is False
