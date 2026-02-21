from __future__ import annotations

import json

from fastapi.testclient import TestClient

from exam_helper.app import create_app
from exam_helper.repository import ProjectRepository


def test_ai_draft_solution_returns_ai_text_verbatim(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    client.post(
        "/questions/save",
        data={
            "question_id": "q0",
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

    class _AI:
        def draft_solution(self, question):
            return "Problem (verbatim): P\nFinal answer: 1"

    app.state.ai = _AI()
    resp = client.post("/questions/q0/ai/draft-solution")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "Problem (verbatim): P" in data["solution_md"]
    assert "Final answer: 1" in data["solution_md"]


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
        def draft_solution(self, question):
            return "Problem (verbatim): P\nFinal answer: 1"

        def generate_mc_options_from_solution(self, question, solution_md):
            raise ValueError("bad json")

    app.state.ai = _BadAI()
    resp = client.post("/questions/q1/ai/distractors")
    assert resp.status_code == 422
    assert resp.json()["ok"] is False


def test_ai_mc_endpoint_returns_choices_and_solution(tmp_path) -> None:
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
            "checker_code": "",
            "figures_json": "[]",
            "points": 5,
        },
    )

    class _GoodAI:
        def draft_solution(self, question):
            return "Problem (verbatim): P\nFinal answer: 1"

        def generate_mc_options_from_solution(self, question, solution_md):
            from exam_helper.models import MCChoice

            assert "Problem (verbatim): P" in solution_md
            assert "Final answer: 1" in solution_md
            return [
                MCChoice(label="A", content_md="a", is_correct=False),
                MCChoice(label="B", content_md="b", is_correct=True),
                MCChoice(label="C", content_md="c", is_correct=False),
                MCChoice(label="D", content_md="d", is_correct=False),
                MCChoice(label="E", content_md="e", is_correct=False),
            ]

    app.state.ai = _GoodAI()
    resp = client.post("/questions/q2/ai/distractors")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "choices_yaml" in data
    assert data["solution_was_generated"] is True
    assert "solution_md" in data
    assert "Problem (verbatim): P" in data["solution_md"]
    assert "Final answer: 1" in data["solution_md"]


def test_ai_mc_endpoint_keeps_existing_solution(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    client.post(
        "/questions/save",
        data={
            "question_id": "q3",
            "title": "T",
            "question_type": "free_response",
            "prompt_md": "P",
            "choices_yaml": "[]",
            "solution_md": "Problem (verbatim): P\nFinal answer: 42",
            "checker_code": "",
            "figures_json": "[]",
            "points": 5,
        },
    )

    class _GoodAI2:
        def draft_solution(self, question):
            raise AssertionError("draft_solution should not be called when solution exists")

        def generate_mc_options_from_solution(self, question, solution_md):
            from exam_helper.models import MCChoice

            assert "Final answer: 42" in solution_md
            return [
                MCChoice(label="A", content_md="a", is_correct=False),
                MCChoice(label="B", content_md="b", is_correct=True),
                MCChoice(label="C", content_md="c", is_correct=False),
                MCChoice(label="D", content_md="d", is_correct=False),
                MCChoice(label="E", content_md="e", is_correct=False),
            ]

    app.state.ai = _GoodAI2()
    resp = client.post("/questions/q3/ai/distractors")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["solution_was_generated"] is False
    assert "solution_md" not in data
