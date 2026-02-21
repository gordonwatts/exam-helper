from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from exam_helper.models import FigureData, Question, QuestionType
from exam_helper.repository import ProjectRepository


def test_figure_hash_round_trip() -> None:
    raw = b"abc123"
    b64 = base64.b64encode(raw).decode("ascii")
    digest = hashlib.sha256(raw).hexdigest()
    fig = FigureData(
        id="f1", mime_type="image/png", data_base64=b64, sha256=digest, caption="cap"
    )
    assert fig.sha256 == digest


def test_repo_save_and_load_question(tmp_path: Path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    q = Question(
        id="q1",
        title="Kinematics",
        prompt_md="Find acceleration.",
        question_type=QuestionType.free_response,
    )
    repo.save_question(q)
    loaded = repo.get_question("q1")
    assert loaded.title == "Kinematics"
