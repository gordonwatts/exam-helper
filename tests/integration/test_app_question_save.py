from __future__ import annotations

import base64
import hashlib
import yaml

from fastapi.testclient import TestClient

from exam_helper.app import create_app
from exam_helper.repository import ProjectRepository


def test_create_question_with_embedded_figure(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)

    raw = b"image-bytes"
    b64 = base64.b64encode(raw).decode("ascii")
    digest = hashlib.sha256(raw).hexdigest()
    fig = (
        '[{"id":"fig_1","mime_type":"image/png","data_base64":"'
        + b64
        + '","sha256":"'
        + digest
        + '","caption":"test"}]'
    )

    resp = client.post(
        "/questions/save",
        data={
            "question_id": "q1",
            "title": "Title",
            "topic": "mech",
            "course_level": "intro",
            "tags": "kinematics",
            "question_type": "free_response",
            "prompt_md": "Prompt",
            "choices_yaml": "[]",
            "solution_md": "Solve",
            "figures_json": fig,
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    saved = repo.get_question("q1")
    assert len(saved.figures) == 1


def test_validate_figure_endpoint_returns_hash_and_size(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)

    raw = b"png-bytes"
    b64 = base64.b64encode(raw).decode("ascii")
    digest = hashlib.sha256(raw).hexdigest()

    resp = client.post("/figures/validate", data={"data_base64": b64})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["sha256"] == digest
    assert payload["size"] == len(raw)


def test_embedded_figure_route_serves_question_figure_bytes(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)

    raw = b"figure-bytes"
    b64 = base64.b64encode(raw).decode("ascii")
    digest = hashlib.sha256(raw).hexdigest()
    fig = (
        '[{"id":"fig_1","mime_type":"image/png","data_base64":"'
        + b64
        + '","sha256":"'
        + digest
        + '","caption":""}]'
    )
    client.post(
        "/questions/save",
        data={
            "question_id": "q1",
            "title": "Title",
            "question_type": "free_response",
            "question_template_md": "![]({{figure_ref}})",
            "solution_parameters_yaml": "{figure_ref: fig_1}",
            "choices_yaml": "[]",
            "distractor_functions_text": "",
            "typed_solution_md": "",
            "figures_json": fig,
            "points": 5,
        },
        follow_redirects=False,
    )

    resp = client.get("/questions/q1/fig_1")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")
    assert resp.content == raw


def test_new_question_page_contains_figure_upload_controls(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)

    resp = client.get("/questions/new")
    assert resp.status_code == 200
    html = resp.text
    assert 'id="figures_json"' in html
    assert 'id="figures_preview"' in html
    assert 'id="btn_add_figure"' in html
    assert 'id="figure_file_input"' in html


def test_save_clears_legacy_checker_data(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    question_path = tmp_path / "questions" / "legacy.yaml"
    question_path.write_text(
        yaml.safe_dump(
            {
                "id": "legacy",
                "title": "Old",
                "question_type": "free_response",
                "prompt_md": "Prompt",
                "choices": [],
                "checker": {
                    "python_code": "def grade(student_answer, context): return {}"
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)
    resp = client.post(
        "/questions/save",
        data={
            "question_id": "legacy",
            "title": "Updated",
            "question_type": "free_response",
            "prompt_md": "Prompt",
            "choices_yaml": "[]",
            "solution_md": "",
            "figures_json": "[]",
            "points": 5,
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    raw = yaml.safe_load(question_path.read_text(encoding="utf-8"))
    assert "checker" not in raw


def test_soft_delete_hides_question_but_keeps_yaml_on_disk(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)

    save = client.post(
        "/questions/save",
        data={
            "question_id": "q1",
            "title": "Delete Me",
            "question_type": "free_response",
            "question_template_md": "Prompt",
            "choices_yaml": "[]",
            "distractor_functions_text": "",
            "typed_solution_md": "",
            "figures_json": "[]",
            "points": 5,
        },
        follow_redirects=False,
    )
    assert save.status_code == 303

    delete = client.post("/questions/q1/delete", follow_redirects=False)
    assert delete.status_code == 303
    assert delete.headers["location"] == "/"

    home = client.get("/")
    assert home.status_code == 200
    assert "/questions/q1/edit" not in home.text

    question_file = tmp_path / "questions" / "q1.yaml"
    assert question_file.exists()
    assert repo.get_question("q1").is_deleted is True


def test_new_question_id_skips_soft_deleted_ids(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)

    save = client.post(
        "/questions/save",
        data={
            "question_id": "q1",
            "title": "Delete Me",
            "question_type": "free_response",
            "question_template_md": "Prompt",
            "choices_yaml": "[]",
            "distractor_functions_text": "",
            "typed_solution_md": "",
            "figures_json": "[]",
            "points": 5,
        },
        follow_redirects=False,
    )
    assert save.status_code == 303
    delete = client.post("/questions/q1/delete", follow_redirects=False)
    assert delete.status_code == 303

    new_page = client.get("/questions/new")
    assert new_page.status_code == 200
    assert 'id="question_id" value="q2"' in new_page.text


def test_save_persists_dollar_math_delimiters(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)

    resp = client.post(
        "/questions/save",
        data={
            "question_id": "q_math",
            "title": "Math",
            "question_type": "free_response",
            "question_template_md": r"Compute \\(x\\) and \\[x^2\\]",
            "choices_yaml": "[]",
            "distractor_functions_text": "",
            "typed_solution_md": r"Use \\(x\\) first.",
            "figures_json": "[]",
            "points": 5,
        },
        follow_redirects=False,
    )

    assert resp.status_code == 303
    saved = repo.get_question("q_math")
    assert saved.solution.question_template_md == "Compute $x$ and $x^2$"
    assert saved.solution.typed_solution_md == "Use $x$ first."


def test_autosave_persists_dollar_math_delimiters(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)

    client.post(
        "/questions/save",
        data={
            "question_id": "q_auto_math",
            "title": "",
            "question_type": "free_response",
            "question_template_md": "",
            "choices_yaml": "[]",
            "distractor_functions_text": "",
            "typed_solution_md": "",
            "figures_json": "[]",
            "points": 5,
        },
        follow_redirects=False,
    )

    resp = client.post(
        "/questions/q_auto_math/autosave",
        json={
            "title": "T",
            "question_type": "free_response",
            "question_template_md": r"Given \\(v\\)",
            "solution_parameters_yaml": "{}",
            "answer_guidance": "",
            "answer_python_code": "",
            "distractor_functions_text": "",
            "choices_yaml": "[]",
            "typed_solution_md": r"So \\(v=1\\).",
            "typed_solution_status": "fresh",
            "figures_json": "[]",
            "points": 5,
        },
    )

    assert resp.status_code == 200
    saved = repo.get_question("q_auto_math")
    assert saved.solution.question_template_md == "Given $v$"
    assert saved.solution.typed_solution_md == "So $v=1$."


def test_save_persists_mc_options_guidance(tmp_path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    app = create_app(tmp_path, openai_key=None)
    client = TestClient(app)

    resp = client.post(
        "/questions/save",
        data={
            "question_id": "q_mc_guidance",
            "title": "MC Guidance",
            "question_type": "multiple_choice",
            "mc_options_guidance": "Focus on missed unit conversions.",
            "question_template_md": "Prompt",
            "choices_yaml": "[]",
            "distractor_functions_text": "",
            "typed_solution_md": "",
            "figures_json": "[]",
            "points": 5,
        },
        follow_redirects=False,
    )

    assert resp.status_code == 303
    saved = repo.get_question("q_mc_guidance")
    assert saved.mc_options_guidance == "Focus on missed unit conversions."