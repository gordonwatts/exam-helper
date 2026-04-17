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
            "answer_formula_md": "",
            "distractor_functions_text": "",
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
            "answer_formula_md": "answer = 10",
            "distractor_functions_text": "",
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
            "answer_formula_md": "answer = 11",
            "distractor_functions_text": "",
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
    assert "3.5" in data["rendered_prompt_md"]


def test_ai_rewrite_and_parameterize_does_not_fallback_title_from_prompt(
    tmp_path,
) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    _seed_question(client, "q_rewrite_no_title")

    class _AI:
        def rewrite_parameterize(self, question):
            return AIService.RewriteResult(
                question_template_md="You shine photons with frequency {{f}}.",
                parameters={"f": 5.0},
                title="",
                usage=AIUsageTotals(),
            )

    app.state.ai = _AI()
    resp = client.post("/questions/q_rewrite_no_title/ai/rewrite-and-parameterize")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["title"] == ""


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
            "answer_formula_md": "answer = 2",
            "mc_answer_specs_json": (
                "["
                '{"formula_md":"answer = 2","rationale_md":"dup"},'
                '{"formula_md":"answer = 3","rationale_md":"r"},'
                '{"formula_md":"answer = 4","rationale_md":"r"},'
                '{"formula_md":"answer = 5","rationale_md":"r"}'
                "]"
            ),
            "distractor_functions_text": "",
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


def test_autosave_updates_mc_formula_preview(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)
    _seed_question(client, "q_mc_formula", qtype="multiple_choice")

    resp = client.post(
        "/questions/q_mc_formula/autosave",
        json={
            "title": "T",
            "question_type": "multiple_choice",
            "question_template_md": "P",
            "solution_parameters_yaml": "{}",
            "answer_formula_md": "answer = 2",
            "answer_guidance": "Use the result {{answer}}.",
            "mc_options_guidance": "",
            "mc_answer_specs_json": (
                "["
                '{"formula_md":"answer = 3","rationale_md":"off by one"},'
                '{"formula_md":"answer = 4","rationale_md":"off by two"},'
                '{"formula_md":"answer = 5","rationale_md":"off by three"},'
                '{"formula_md":"","rationale_md":"off by four"}'
                "]"
            ),
            "distractor_functions_text": "",
            "choices_yaml": "[]",
            "typed_solution_md": "",
            "typed_solution_status": "missing",
            "figures_json": "[]",
            "points": 5,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert [choice["label"] for choice in body["mc_preview_choices"]] == [
        "A",
        "B",
        "C",
        "D",
    ]
    assert body["mc_preview_choices"][0]["content_md"] == "2"
    assert "Formula defines answer directly" in body["warning"]
    assert "choice_4" not in body["warning"]
    assert "choice_4" in body["mc_preview_warning"]
    assert body["mc_preview_rows"]
    saved = repo.get_question("q_mc_formula")
    assert saved.solution.mc_answer_specs[0].formula_md == "answer = 3"
    assert [c.content_md for c in saved.choices][0] == "2"


def test_generate_mc_distractors_retries_and_returns_partial_unique_set(
    tmp_path,
) -> None:
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
            "answer_formula_md": "answer = 2",
            "distractor_functions_text": "",
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
                    DistractorFunction(
                        id="d1",
                        python_code="def distractor(params):\n    return {'distractor_md':'2','rationale':'dup'}",
                    ),
                    DistractorFunction(
                        id="d2",
                        python_code="def distractor(params):\n    return {'distractor_md':'2','rationale':'dup'}",
                    ),
                    DistractorFunction(
                        id="d3",
                        python_code="def distractor(params):\n    return {'distractor_md':'2','rationale':'dup'}",
                    ),
                    DistractorFunction(
                        id="d4",
                        python_code="def distractor(params):\n    return {'distractor_md':'2','rationale':'dup'}",
                    ),
                ],
                usage=AIUsageTotals(),
            )

    app.state.ai = _AI()
    resp = client.post("/questions/q_retry/ai/generate-mc-distractors")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "warning" in body
    assert "full unique MC set" in body["warning"]
    assert body["collisions"]
    choices = body["choices_yaml"]
    assert "label: A" in choices


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


def test_generate_answer_formula_retries_with_runtime_feedback(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    _seed_question(client, "q_answer_retry")
    client.post(
        "/questions/q_answer_retry/autosave",
        json={
            "title": "T",
            "question_type": "free_response",
            "question_template_md": "P",
            "solution_parameters_yaml": "{v: 5}",
            "answer_guidance": "",
            "answer_formula_md": "",
            "distractor_functions_text": "",
            "choices_yaml": "[]",
            "typed_solution_md": "",
            "typed_solution_status": "missing",
            "figures_json": "[]",
            "points": 5,
        },
    )

    class _AI:
        def __init__(self):
            self.calls = 0

        def generate_answer_formula(self, question, error_feedback=""):
            self.calls += 1
            if self.calls == 1:
                return AIService.AnswerFormulaResult(
                    answer_formula_md="answer = x",
                    usage=AIUsageTotals(),
                )
            return AIService.AnswerFormulaResult(
                answer_formula_md="x = 1\nanswer = x",
                usage=AIUsageTotals(),
            )

    fake = _AI()
    app.state.ai = fake
    resp = client.post("/questions/q_answer_retry/ai/generate-answer-formula")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "answer" in data["answer_formula_md"]
    assert fake.calls == 2


def test_autosave_keeps_last_good_answer_preview_when_formula_has_warning(
    tmp_path,
) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)
    _seed_question(client, "q_warning")

    first = client.post(
        "/questions/q_warning/autosave",
        json={
            "title": "T",
            "question_type": "free_response",
            "question_template_md": "P",
            "solution_parameters_yaml": "{v: 5}",
            "answer_formula_md": "answer = v",
            "answer_guidance": "{{answer}}",
            "distractor_functions_text": "",
            "choices_yaml": "[]",
            "typed_solution_md": "",
            "typed_solution_status": "missing",
            "figures_json": "[]",
            "points": 5,
        },
    )
    assert first.status_code == 200
    assert first.json()["rendered_answer_md"] == "5"

    second = client.post(
        "/questions/q_warning/autosave",
        json={
            "title": "T",
            "question_type": "free_response",
            "question_template_md": "P",
            "solution_parameters_yaml": "{v: 5}",
            "answer_formula_md": "answer = missing_name",
            "answer_guidance": "{{answer}}",
            "distractor_functions_text": "",
            "choices_yaml": "[]",
            "typed_solution_md": "",
            "typed_solution_status": "missing",
            "figures_json": "[]",
            "points": 5,
        },
    )
    assert second.status_code == 200
    body = second.json()
    assert body["ok"] is True
    assert body["warning"]
    assert body["rendered_answer_md"] == "5"
    saved = repo.get_question("q_warning")
    assert saved.solution.last_computed_answer_md == "5"


def test_harness_run_preserves_latex_backslashes_in_answer_output(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)
    _seed_question(client, "q_latex")

    client.post(
        "/questions/q_latex/autosave",
        json={
            "title": "T",
            "question_type": "free_response",
            "question_template_md": r"Use \\(\\theta\\)",
            "solution_parameters_yaml": "{}",
            "answer_guidance": r"$\theta$",
            "answer_formula_md": "answer = 1",
            "distractor_functions_text": "",
            "choices_yaml": "[]",
            "typed_solution_md": "",
            "typed_solution_status": "missing",
            "figures_json": "[]",
            "points": 5,
        },
    )

    resp = client.post("/questions/q_latex/harness/run")
    assert resp.status_code == 200
    data = resp.json()
    assert data["computed_answer_md"] == r"$\theta$"
    assert data["final_answer_text"] == "1"

    saved = repo.get_question("q_latex")
    assert saved.solution.last_computed_answer_md == r"$\theta$"


def test_autosave_updates_mc_options_guidance(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)
    _seed_question(client, "q_mc_guidance", qtype="multiple_choice")

    resp = client.post(
        "/questions/q_mc_guidance/autosave",
        json={
            "title": "T",
            "question_type": "multiple_choice",
            "mc_options_guidance": "Use realistic sign mistakes only.",
            "question_template_md": "P",
            "solution_parameters_yaml": "{}",
            "answer_formula_md": "",
            "distractor_functions_text": "",
            "choices_yaml": "[]",
            "typed_solution_md": "",
            "typed_solution_status": "missing",
            "figures_json": "[]",
            "points": 5,
        },
    )

    assert resp.status_code == 200
    saved = repo.get_question("q_mc_guidance")
    assert saved.mc_options_guidance == "Use realistic sign mistakes only."
