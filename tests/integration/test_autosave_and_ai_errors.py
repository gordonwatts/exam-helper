from __future__ import annotations

from fastapi.testclient import TestClient

from exam_helper.ai_service import AIService
from exam_helper.app import create_app
from exam_helper.models import AIUsageTotals, DistractorFunction
from exam_helper.repository import ProjectRepository


def _seed_question(client: TestClient, qid: str, qtype: str = "free_response") -> None:
    client.post(
        "/questions/save",
        data={
            "question_id": qid,
            "title": "",
            "question_type": qtype,
            "prompt_md": "old prompt",
            "question_template_md": "old template",
            "solution_parameters_yaml": "{}",
            "answer_python_code": "",
            "distractor_functions_yaml": "[]",
            "choices_yaml": "[]",
            "typed_solution_md": "",
            "typed_solution_status": "missing",
            "figures_json": "[]",
            "points": 5,
        },
    )


def test_autosave_marks_typed_solution_stale_on_parameter_change(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)
    _seed_question(client, "q_auto")

    client.post(
        "/questions/q_auto/autosave",
        json={
            "title": "T",
            "question_type": "free_response",
            "prompt_md": "P",
            "question_template_md": "v={{v}}",
            "solution_parameters_yaml": "{v: 10}",
            "answer_python_code": "def solve(params):\n    return {'answer_md':'10','final_answer':'10'}\n",
            "distractor_functions_yaml": "[]",
            "choices_yaml": "[]",
            "typed_solution_md": "Draft",
            "typed_solution_status": "fresh",
            "figures_json": "[]",
            "points": 5,
        },
    )
    resp = client.post(
        "/questions/q_auto/autosave",
        json={
            "title": "T",
            "question_type": "free_response",
            "prompt_md": "P",
            "question_template_md": "v={{v}}",
            "solution_parameters_yaml": "{v: 11}",
            "answer_python_code": "def solve(params):\n    return {'answer_md':'11','final_answer':'11'}\n",
            "distractor_functions_yaml": "[]",
            "choices_yaml": "[]",
            "typed_solution_md": "Draft",
            "typed_solution_status": "fresh",
            "figures_json": "[]",
            "points": 5,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["typed_solution_status"] == "stale"


def test_ai_rewrite_and_parameterize_updates_template_and_params(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    _seed_question(client, "q_rewrite")

    class _AI:
        def rewrite_parameterize(self, question):
            return AIService.RewriteResult(
                question_template_md="A cart has speed {{v}} m/s.",
                parameters={"v": 3.5},
                title="Cart speed",
                usage=AIUsageTotals(),
            )

    app.state.ai = _AI()
    resp = client.post("/questions/q_rewrite/ai/rewrite-and-parameterize")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "3.5" in data["prompt_md"]


def test_harness_run_returns_422_for_collisions(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)
    _seed_question(client, "q_mc", qtype="multiple_choice")
    client.post(
        "/questions/q_mc/autosave",
        json={
            "title": "T",
            "question_type": "multiple_choice",
            "prompt_md": "P",
            "question_template_md": "P",
            "solution_parameters_yaml": "{}",
            "answer_python_code": "def solve(params):\n    return {'answer_md':'2','final_answer':'2'}\n",
            "distractor_functions_yaml": (
                "- id: d1\n  python_code: |\n    def distractor(params):\n      return {'distractor_md':'2','rationale':'dup'}\n"
                "- id: d2\n  python_code: |\n    def distractor(params):\n      return {'distractor_md':'3','rationale':'r'}\n"
                "- id: d3\n  python_code: |\n    def distractor(params):\n      return {'distractor_md':'4','rationale':'r'}\n"
                "- id: d4\n  python_code: |\n    def distractor(params):\n      return {'distractor_md':'5','rationale':'r'}\n"
            ),
            "choices_yaml": "[]",
            "typed_solution_md": "",
            "typed_solution_status": "missing",
            "figures_json": "[]",
            "points": 5,
        },
    )
    resp = client.post("/questions/q_mc/harness/run")
    assert resp.status_code == 422
    assert resp.json()["ok"] is False
    assert resp.json()["collisions"]


def test_generate_mc_distractors_retries_and_fails_with_actionable_error(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    _seed_question(client, "q_retry", qtype="multiple_choice")
    client.post(
        "/questions/q_retry/autosave",
        json={
            "title": "T",
            "question_type": "multiple_choice",
            "prompt_md": "P",
            "question_template_md": "P",
            "solution_parameters_yaml": "{}",
            "answer_python_code": "def solve(params):\n    return {'answer_md':'2','final_answer':'2'}\n",
            "distractor_functions_yaml": "[]",
            "choices_yaml": "[]",
            "typed_solution_md": "",
            "typed_solution_status": "missing",
            "figures_json": "[]",
            "points": 5,
        },
    )

    class _AI:
        def generate_distractor_functions(self, question):
            return AIService.DistractorFunctionsResult(
                distractors=[
                    DistractorFunction(id="d1", python_code="def distractor(params):\n    return {'distractor_md':'2','rationale':'dup'}"),
                    DistractorFunction(id="d2", python_code="def distractor(params):\n    return {'distractor_md':'2','rationale':'dup'}"),
                    DistractorFunction(id="d3", python_code="def distractor(params):\n    return {'distractor_md':'2','rationale':'dup'}"),
                    DistractorFunction(id="d4", python_code="def distractor(params):\n    return {'distractor_md':'2','rationale':'dup'}"),
                ],
                usage=AIUsageTotals(),
            )

    app.state.ai = _AI()
    resp = client.post("/questions/q_retry/ai/generate-mc-distractors")
    assert resp.status_code == 422
    body = resp.json()
    assert "after 3 attempts" in body["error"]
    assert body["collisions"]


def test_generate_typed_solution_sets_status_fresh(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    _seed_question(client, "q_typed")

    class _AI:
        def generate_typed_solution(self, question):
            return AIService.AIResult(text="Typed explanation", usage=AIUsageTotals())

    app.state.ai = _AI()
    resp = client.post("/questions/q_typed/ai/generate-typed-solution")
    assert resp.status_code == 200
    body = resp.json()
    assert body["typed_solution_md"] == "Typed explanation"
    assert body["typed_solution_status"] == "fresh"
