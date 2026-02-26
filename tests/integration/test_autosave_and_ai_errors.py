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
        "mc_options_guidance": "Avoid reusing prior answer.",
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
    assert saved.mc_options_guidance == "Avoid reusing prior answer."
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
            "solution_python_code": (
                "def solve(params, context):\n"
                "    return {'computed_output_md': 'x', 'choices_yaml': 'not: [valid'}\n"
            ),
            "solution_parameters_yaml": "{}",
            "checker_code": "",
            "figures_json": "[]",
            "points": 5,
        },
    )
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
            "solution_python_code": (
                "def solve(params, context):\n"
                "    return {\n"
                "        'computed_output_md': 'Final answer: 1',\n"
                "        'choices_yaml': \"\"\"- label: A\n"
                "  content_md: a\n"
                "  is_correct: false\n"
                "- label: B\n"
                "  content_md: b\n"
                "  is_correct: true\n"
                "- label: C\n"
                "  content_md: c\n"
                "  is_correct: false\n"
                "- label: D\n"
                "  content_md: d\n"
                "  is_correct: false\n"
                "- label: E\n"
                "  content_md: e\n"
                "  is_correct: false\n"
                "\"\"\",\n"
                "    }\n"
            ),
            "solution_parameters_yaml": "{}",
            "checker_code": "",
            "figures_json": "[]",
            "points": 5,
        },
    )
    resp = client.post("/questions/q2/ai/distractors")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "choices_yaml" in data
    assert data["solution_was_generated"] is False
    assert "solution_md" in data
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
            "solution_md": "Final answer: 42",
            "solution_python_code": (
                "def solve(params, context):\n"
                "    return {\n"
                "        'computed_output_md': 'Final answer: 42',\n"
                "        'choices_yaml': \"\"\"- label: A\n"
                "  content_md: a\n"
                "  is_correct: false\n"
                "- label: B\n"
                "  content_md: b\n"
                "  is_correct: true\n"
                "- label: C\n"
                "  content_md: c\n"
                "  is_correct: false\n"
                "- label: D\n"
                "  content_md: d\n"
                "  is_correct: false\n"
                "- label: E\n"
                "  content_md: e\n"
                "  is_correct: false\n"
                "\"\"\",\n"
                "    }\n"
            ),
            "solution_parameters_yaml": "{}",
            "checker_code": "",
            "figures_json": "[]",
            "points": 5,
        },
    )
    resp = client.post("/questions/q3/ai/distractors")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["solution_was_generated"] is False
    assert "solution_md" in data
