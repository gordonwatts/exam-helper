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
                "checker": {"python_code": "def grade(student_answer, context): return {}"},
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
