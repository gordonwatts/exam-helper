from __future__ import annotations

from fastapi.testclient import TestClient

from exam_helper.app import create_app
from exam_helper.repository import ProjectRepository


def test_export_docx_route_uses_sanitized_project_name(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("EM & Waves  Final!!!", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)

    fake_docx = b"PK\x03\x04fake-docx"

    def _fake_export(project_root, include_solutions=False):
        assert project_root == tmp_path
        assert include_solutions is False
        return fake_docx, []

    app.dependency_overrides = {}
    import exam_helper.app as app_module

    original = app_module.render_project_docx_bytes
    app_module.render_project_docx_bytes = _fake_export
    try:
        response = client.post("/export/docx", data={})
    finally:
        app_module.render_project_docx_bytes = original

    assert response.status_code == 200
    assert response.content == fake_docx
    assert (
        response.headers["content-disposition"]
        == 'attachment; filename="em-waves-final.docx"'
    )


def test_export_docx_route_include_solutions_and_warning_headers(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("***", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)

    fake_docx = b"PK\x03\x04fake-docx"
    captured = {"include": None}

    def _fake_export(project_root, include_solutions=False):
        captured["include"] = include_solutions
        return fake_docx, [
            "Pandoc not available; DOCX was exported with plain-text math fallback."
        ]

    import exam_helper.app as app_module

    original = app_module.render_project_docx_bytes
    app_module.render_project_docx_bytes = _fake_export
    try:
        response = client.post("/export/docx", data={"include_solutions": "1"})
    finally:
        app_module.render_project_docx_bytes = original

    assert response.status_code == 200
    assert captured["include"] is True
    assert response.headers["content-disposition"] == 'attachment; filename="exam.docx"'
    assert "x-exam-helper-export-warnings" in response.headers
    assert "set-cookie" in response.headers


def test_export_single_question_docx_route_uses_question_id_filename(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)

    save_resp = client.post(
        "/questions/save",
        data={
            "question_id": "single-1",
            "title": "Only One",
            "question_type": "free_response",
            "question_template_md": "A prompt.",
            "solution_parameters_yaml": "{}",
            "answer_formula_md": "",
            "answer_guidance": "",
            "mc_options_guidance": "",
            "distractor_functions_text": "",
            "choices_yaml": "[]",
            "typed_solution_md": "",
            "typed_solution_status": "fresh",
            "figures_json": "[]",
            "points": 5,
        },
        follow_redirects=False,
    )
    assert save_resp.status_code == 303

    fake_docx = b"PK\x03\x04fake-docx"
    captured = {"project_root": None, "question_id": None, "include": None}

    def _fake_export(project_root, question_id, include_solutions=False):
        captured["project_root"] = project_root
        captured["question_id"] = question_id
        captured["include"] = include_solutions
        return fake_docx, ["single question warning"]

    import exam_helper.app as app_module

    original = app_module.render_question_docx_bytes
    app_module.render_question_docx_bytes = _fake_export
    try:
        response = client.post(
            "/questions/single-1/export/docx", data={"include_solutions": "1"}
        )
    finally:
        app_module.render_question_docx_bytes = original

    assert response.status_code == 200
    assert captured["project_root"] == tmp_path
    assert captured["question_id"] == "single-1"
    assert captured["include"] is True
    assert response.content == fake_docx
    assert (
        response.headers["content-disposition"]
        == 'attachment; filename="single-1.docx"'
    )
    assert "x-exam-helper-export-warnings" in response.headers
    assert "set-cookie" in response.headers
