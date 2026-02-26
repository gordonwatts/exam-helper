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


def test_ai_draft_solution_falls_back_when_structured_generation_fails(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    client.post(
        "/questions/save",
        data={
            "question_id": "q_struct_fail",
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
        def draft_solution_with_code(self, question, error_feedback=""):
            raise ValueError("model returned invalid JSON")

        def draft_solution(self, question):
            return "Fallback solution text"

    app.state.ai = _AI()
    resp = client.post("/questions/q_struct_fail/ai/draft-solution")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["solution_md"] == "Fallback solution text"
    assert "warning" in data
    assert "Structured solution generation failed" in data["warning"]


def test_ai_draft_solution_keeps_structured_code_when_non_mc_choices_yaml_invalid(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    client.post(
        "/questions/save",
        data={
            "question_id": "q_struct_partial",
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
        class _Draft:
            worked_solution_md = "Use lens equation."
            python_code = (
                "def solve(params, context):\n"
                "    return {'final_answer_text': 'Jacob is farsighted and has an eye with a power of +2 D',"
                " 'choices_yaml': 'A'}\n"
            )
            parameters = {"p": 1}
            from exam_helper.models import AIUsageTotals

            usage = AIUsageTotals()

        def draft_solution_with_code(self, question, error_feedback=""):
            return self._Draft()

        def draft_solution(self, question):
            raise AssertionError("Should not fall back when structured result is otherwise valid.")

    app.state.ai = _AI()
    resp = client.post("/questions/q_struct_partial/ai/draft-solution")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "solution_python_code" in data
    assert "def solve" in data["solution_python_code"]
    assert data["final_answer_text"].startswith("Jacob is farsighted")
    assert "warning" in data
    assert "invalid choices_yaml" in data["warning"]


def test_ai_draft_solution_mc_accepts_ae_mapping_choices_yaml(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    client.post(
        "/questions/save",
        data={
            "question_id": "q_mc_map",
            "title": "T",
            "question_type": "multiple_choice",
            "prompt_md": "P",
            "choices_yaml": (
                "- label: A\n  content_md: a\n  is_correct: false\n"
                "- label: B\n  content_md: b\n  is_correct: true\n"
            ),
            "solution_md": "",
            "checker_code": "",
            "figures_json": "[]",
            "points": 5,
        },
    )

    class _AI:
        class _Draft:
            worked_solution_md = "Use lens equation."
            python_code = (
                "def solve(params, context):\n"
                "    return {\n"
                "      'final_answer_text': 'Final answer: +2 D',\n"
                "      'choices_yaml': \"\"\"A:\n"
                "  content_md: a\n"
                "  is_correct: false\n"
                "  rationale: forgot conversion\n"
                "B:\n"
                "  content_md: b\n"
                "  is_correct: true\n"
                "  rationale: correct\n"
                "C:\n"
                "  content_md: c\n"
                "  is_correct: false\n"
                "  rationale: sign error\n"
                "D:\n"
                "  content_md: d\n"
                "  is_correct: false\n"
                "  rationale: dropped term\n"
                "E:\n"
                "  content_md: e\n"
                "  is_correct: false\n"
                "  rationale: wrong units\n"
                "\"\"\",\n"
                "    }\n"
            )
            parameters = {"p": 1}
            from exam_helper.models import AIUsageTotals

            usage = AIUsageTotals()

        def draft_solution_with_code(self, question, error_feedback=""):
            return self._Draft()

        def draft_solution(self, question):
            raise AssertionError("Should not fall back for A-E mapping choices YAML.")

    app.state.ai = _AI()
    resp = client.post("/questions/q_mc_map/ai/draft-solution")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "solution_python_code" in data and "def solve" in data["solution_python_code"]
    assert "choices_yaml" in data
    assert "label: A" in data["choices_yaml"]
    assert 'rationale: "forgot conversion"' in data["choices_yaml"]


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
                "    return {'final_answer_text': 'x', 'choices_yaml': 'not: [valid'}\n"
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
                "        'final_answer_text': 'Final answer: 1',\n"
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
                "        'final_answer_text': 'Final answer: 42',\n"
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


def test_solution_run_normalizes_and_quotes_rationale_yaml(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    client.post(
        "/questions/save",
        data={
            "question_id": "q4",
            "title": "T",
            "question_type": "free_response",
            "prompt_md": "P",
            "choices_yaml": "[]",
            "solution_md": "S",
            "solution_python_code": (
                "def solve(params, context):\n"
                "    return {\n"
                "        'final_answer_text': 'Final answer: 42',\n"
                "        'choices_yaml': \"\"\"- label: A\n"
                "  content_md: a\n"
                "  is_correct: false\n"
                "  rationale: forgot factor 1000\n"
                "- label: B\n"
                "  content_md: b\n"
                "  is_correct: true\n"
                "  rationale: correct\n"
                "- label: C\n"
                "  content_md: c\n"
                "  is_correct: false\n"
                "  rationale: sign error\n"
                "- label: D\n"
                "  content_md: d\n"
                "  is_correct: false\n"
                "  rationale: dropped term\n"
                "- label: E\n"
                "  content_md: e\n"
                "  is_correct: false\n"
                "  rationale: wrong unit\n"
                "\"\"\",\n"
                "    }\n"
            ),
            "solution_parameters_yaml": "{}",
            "checker_code": "",
            "figures_json": "[]",
            "points": 5,
        },
    )

    resp = client.post("/questions/q4/solution-code/run")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert 'rationale: "forgot factor 1000"' in data["choices_yaml"]
