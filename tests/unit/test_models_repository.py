from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import yaml

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
        mc_options_guidance="Avoid sign-error distractors.",
        question_type=QuestionType.free_response,
    )
    repo.save_question(q)
    loaded = repo.get_question("q1")
    assert loaded.title == "Kinematics"
    assert loaded.mc_options_guidance == "Avoid sign-error distractors."
    assert loaded.points == 5


def test_question_defaults_allow_empty_title() -> None:
    q = Question(id="q2", prompt_md="p")
    assert q.title == ""
    assert q.points == 5


def test_project_defaults_include_ai_config(tmp_path: Path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics")
    project = repo.load_project()
    assert project.ai.model == "gpt-5.2"
    assert project.ai.prompts.overall == ""
    assert project.ai.usage.total_tokens == 0


def test_project_load_back_compat_without_ai_block(tmp_path: Path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.ensure_layout()
    (tmp_path / "project.yaml").write_text("name: Exam\ncourse: Physics\n", encoding="utf-8")
    project = repo.load_project()
    assert project.ai.model == "gpt-5.2"
    assert project.ai.usage.total_cost_usd == 0.0


def test_init_project_accepts_custom_model(tmp_path: Path) -> None:
    repo = ProjectRepository(tmp_path)
    repo.init_project("Exam", "Physics", openai_model="gpt-5.2-custom")
    data = yaml.safe_load((tmp_path / "project.yaml").read_text(encoding="utf-8"))
    assert data["ai"]["model"] == "gpt-5.2-custom"
