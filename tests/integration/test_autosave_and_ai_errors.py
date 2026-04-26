from __future__ import annotations

import json

from fastapi.testclient import TestClient

from exam_helper.ai_service import AIService
from exam_helper.app import create_app
from exam_helper.models import (
    AIUsageTotals,
    ChatTurn,
    DistractorFunction,
    MCAnswerSpec,
    Question,
)
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


def _editor_state_for_question(question: Question) -> dict[str, object]:
    distractor_functions_text = ""
    if question.solution.distractor_python_code:
        distractor_functions_text = (
            "\n---\n".join(
                [
                    f"# distractor: {d.id}\n{(d.python_code or '').strip()}"
                    for d in question.solution.distractor_python_code
                ]
            ).strip()
            + "\n"
        )
    return {
        "title": question.title,
        "question_type": question.question_type.value,
        "mc_options_guidance": question.mc_options_guidance,
        "question_template_md": question.solution.question_template_md,
        "solution_parameters_yaml": (
            "{}"
            if not question.solution.parameters
            else json.dumps(question.solution.parameters)
        ),
        "answer_formula_md": question.solution.answer_formula_md,
        "answer_guidance": question.solution.answer_guidance,
        "distractor_functions_text": distractor_functions_text,
        "mc_answer_specs_json": json.dumps(
            [spec.model_dump(mode="json") for spec in question.solution.mc_answer_specs]
        ),
        "choices_yaml": "[]",
        "typed_solution_md": question.solution.typed_solution_md,
        "typed_solution_status": question.solution.typed_solution_status,
        "figures_json": json.dumps(
            [fig.model_dump(mode="json") for fig in question.figures]
        ),
        "chat_history_json": json.dumps(
            [turn.model_dump(mode="json") for turn in question.solution.chat_history]
        ),
        "points": question.points,
    }


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
    assert body["mc_preview_rationale"].startswith("A. 2 -")
    assert body["mc_preview_rows"]
    saved = repo.get_question("q_mc_formula")
    assert saved.solution.mc_answer_specs[0].formula_md == "answer = 3"
    assert [c.content_md for c in saved.choices][0] == "2"


def test_harness_run_keeps_correct_answer_out_of_distractor_rows(tmp_path) -> None:
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)
    _seed_question(client, "q_mc_harness_rows", qtype="multiple_choice")
    client.post(
        "/questions/q_mc_harness_rows/autosave",
        json={
            "title": "T",
            "question_type": "multiple_choice",
            "question_template_md": "P",
            "solution_parameters_yaml": '{"offset": 10}',
            "answer_formula_md": "base = 2\nanswer = base + offset",
            "mc_answer_specs_json": (
                "["
                '{"formula_md":"answer = base + 1","rationale_md":"r1"},'
                '{"formula_md":"answer = base + 2","rationale_md":"r2"},'
                '{"formula_md":"answer = base + 3","rationale_md":"r3"},'
                '{"formula_md":"answer = base + 4","rationale_md":"r4"}'
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

    resp = client.post("/questions/q_mc_harness_rows/harness/run")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert [row["preview_md"] for row in body["mc_preview_rows"]] == [
        "3",
        "4",
        "5",
        "6",
    ]
    assert [choice["content_md"] for choice in body["mc_preview_choices"]] == [
        "3",
        "4",
        "5",
        "6",
        "12",
    ]


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


def test_ai_chat_updates_question_and_returns_payload(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    _seed_question(client, "q_chat")

    class _AI:
        def chat_edit_question(self, question, user_message, attached_figure_ids=None):
            assert question.id == "q_chat"
            assert user_message == "Please clean this up."
            assert attached_figure_ids == []
            updated = question.model_copy(deep=True)
            updated.title = "Cleaned title"
            updated.solution.question_template_md = "A cleaner prompt."
            updated.solution.chat_history = []
            return AIService.QuestionEditorResult(
                assistant_message="Cleaned up the prompt.",
                question=updated,
                warnings=["Preserved the typed solution."],
                usage=AIUsageTotals(),
                changed_fields=["title", "question_template_md"],
            )

    app.state.ai = _AI()
    q = repo.get_question("q_chat")
    resp = client.post(
        "/questions/q_chat/ai/chat",
        json={
            "message": "Please clean this up.",
            "attached_figure_ids": [],
            "editor_state": _editor_state_for_question(q),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["assistant_message"] == "Cleaned up the prompt."
    assert "title" in body["changed_fields"]
    assert body["title"] == "Cleaned title"
    assert body["question_template_md"] == "A cleaner prompt."
    assert body["warnings"] == ["Preserved the typed solution."]
    saved = repo.get_question("q_chat")
    assert saved.title == "Cleaned title"
    assert saved.solution.question_template_md == "A cleaner prompt."
    assert len(saved.solution.chat_history) == 1
    assert saved.solution.chat_history[0].user_message == "Please clean this up."


def test_ai_chat_refreshes_rendered_answer_when_guidance_changes(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    _seed_question(client, "q_chat_answer")
    seeded = repo.get_question("q_chat_answer")
    seeded.solution.answer_formula_md = "answer = 9"
    seeded.solution.answer_guidance = "Old answer: {{answer}}"
    seeded.solution.last_computed_answer_md = "Old answer: 9"
    repo.save_question(seeded)

    class _AI:
        def chat_edit_question(self, question, user_message, attached_figure_ids=None):
            updated = question.model_copy(deep=True)
            updated.solution.answer_guidance = "New answer: {{answer}}"
            updated.solution.last_computed_answer_md = "Stale answer: 9"
            return AIService.QuestionEditorResult(
                assistant_message="Updated the answer text.",
                question=updated,
                warnings=[],
                usage=AIUsageTotals(),
                changed_fields=["answer_guidance"],
            )

    app.state.ai = _AI()
    resp = client.post(
        "/questions/q_chat_answer/ai/chat",
        json={
            "message": "Update the answer wording.",
            "attached_figure_ids": [],
            "editor_state": _editor_state_for_question(seeded),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["rendered_answer_md"] == "New answer: 9"
    saved = repo.get_question("q_chat_answer")
    assert saved.solution.answer_guidance == "New answer: {{answer}}"
    assert saved.solution.last_computed_answer_md == "New answer: 9"


def test_ai_chat_uses_live_editor_state_and_persists_last_five_turns(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    _seed_question(client, "q_live_chat")
    seeded = repo.get_question("q_live_chat")
    seeded.solution.chat_history = [
        ChatTurn(user_message=f"user {idx}", assistant_message=f"assistant {idx}")
        for idx in range(1, 6)
    ]
    repo.save_question(seeded)

    class _AI:
        def chat_edit_question(self, question, user_message, attached_figure_ids=None):
            assert question.title == "Unsaved local title"
            assert question.solution.question_template_md == "Unsaved local template"
            assert attached_figure_ids == ["fig_1"]
            updated = question.model_copy(deep=True)
            updated.solution.answer_formula_md = "answer = 12"
            updated.solution.answer_guidance = "{{answer}}"
            updated.solution.last_computed_answer_md = "12"
            return AIService.QuestionEditorResult(
                assistant_message="Stored the answer formula.",
                question=updated,
                warnings=[],
                usage=AIUsageTotals(),
                changed_fields=["answer_formula_md", "last_computed_answer_md"],
            )

    app.state.ai = _AI()
    live_state = _editor_state_for_question(seeded)
    live_state["title"] = "Unsaved local title"
    live_state["question_template_md"] = "Unsaved local template"
    live_state["figures_json"] = json.dumps(
        [
            {
                "id": "fig_1",
                "mime_type": "image/png",
                "data_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2pG1EAAAAASUVORK5CYII=",
                "sha256": "e8516d5fe5dff6dd2cc7f7269a4df3b741b83472ae1fbf2cbfb2aa8d0cd91015",
                "caption": "chat attachment",
            }
        ]
    )
    live_state["chat_history_json"] = json.dumps(
        [turn.model_dump(mode="json") for turn in seeded.solution.chat_history]
    )

    resp = client.post(
        "/questions/q_live_chat/ai/chat",
        json={
            "message": "Compute the answer.",
            "attached_figure_ids": ["fig_1"],
            "editor_state": live_state,
        },
    )

    assert resp.status_code == 200
    saved = repo.get_question("q_live_chat")
    assert saved.title == "Unsaved local title"
    assert saved.solution.question_template_md == "Unsaved local template"
    assert saved.solution.last_computed_answer_md == "12"
    assert len(saved.solution.chat_history) == 5
    assert saved.solution.chat_history[-1].user_message == "Compute the answer."
    assert saved.solution.chat_history[-1].attached_figure_ids == ["fig_1"]
    assert saved.solution.chat_history[0].user_message == "user 2"


def test_autosave_after_chat_keeps_chat_applied_solution_fields(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key="k")
    client = TestClient(app)
    _seed_question(client, "q_chat_autosave", qtype="multiple_choice")

    class _AI:
        def chat_edit_question(self, question, user_message, attached_figure_ids=None):
            updated = question.model_copy(deep=True)
            updated.solution.answer_formula_md = "answer = 9"
            updated.solution.answer_guidance = "{{answer}}"
            updated.solution.mc_answer_specs = [
                MCAnswerSpec(formula_md=f"answer = {idx}", rationale_md=f"r{idx}")
                for idx in range(1, 5)
            ]
            return AIService.QuestionEditorResult(
                assistant_message="Updated answer and distractors.",
                question=updated,
                warnings=[],
                usage=AIUsageTotals(),
                changed_fields=["answer_formula_md", "mc_answer_specs_json"],
            )

    app.state.ai = _AI()
    original = repo.get_question("q_chat_autosave")
    chat_resp = client.post(
        "/questions/q_chat_autosave/ai/chat",
        json={
            "message": "Set the answer and distractors.",
            "attached_figure_ids": [],
            "editor_state": _editor_state_for_question(original),
        },
    )
    assert chat_resp.status_code == 200
    body = chat_resp.json()

    autosave_resp = client.post(
        "/questions/q_chat_autosave/autosave",
        json={
            "title": body["title"],
            "question_type": body["question_type"],
            "mc_options_guidance": body["mc_options_guidance"],
            "question_template_md": body["question_template_md"],
            "solution_parameters_yaml": body["solution_parameters_yaml"],
            "answer_formula_md": body["answer_formula_md"],
            "answer_guidance": body["answer_guidance"],
            "distractor_functions_text": body["distractor_functions_text"],
            "mc_answer_specs_json": body["mc_answer_specs_json"],
            "choices_yaml": body["choices_yaml"],
            "typed_solution_md": body["typed_solution_md"],
            "typed_solution_status": body["typed_solution_status"],
            "figures_json": body["figures_json"],
            "chat_history_json": body["chat_history_json"],
            "points": body["points"],
        },
    )
    assert autosave_resp.status_code == 200
    saved = repo.get_question("q_chat_autosave")
    assert saved.solution.answer_formula_md == "answer = 9"
    assert len(saved.solution.mc_answer_specs) == 4
