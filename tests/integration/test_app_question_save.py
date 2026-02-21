from __future__ import annotations

import base64
import hashlib

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
            "checker_code": "def grade(student_answer, context): return {'verdict': 'correct'}",
            "figures_json": fig,
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    saved = repo.get_question("q1")
    assert len(saved.figures) == 1
