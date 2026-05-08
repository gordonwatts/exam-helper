from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from exam_helper import cli
from exam_helper.models import DEFAULT_OPENAI_MODEL

runner = CliRunner()


def test_serve_help_mentions_project_path_and_openai_env() -> None:
    result = runner.invoke(cli.cli_app, ["serve", "--help"])

    assert result.exit_code == 0
    out = result.stdout
    assert "project.yaml" in out
    assert "Path to the exam project directory." in out
    assert "EXAM_HELPER_OPENAI_KEY in ~/.env" in out
    assert "EXAM_HELPER_OPENAI_KEY in the project .env path" in out
    assert "--openai-key" in out


def test_serve_command_accepts_positional_path(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_app(*, project_root: Path, openai_key: str | None):
        captured["project_root"] = project_root
        captured["openai_key"] = openai_key
        return object()

    def fake_run(app, host: str, port: int, log_level: str) -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr(cli, "create_app", fake_create_app)
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    monkeypatch.setattr(
        cli, "resolve_openai_api_key", lambda key, project_root=None: "key"
    )

    result = runner.invoke(cli.cli_app, ["serve", "my-project", "--port", "9000"])

    assert result.exit_code == 0
    assert captured["project_root"] == Path("my-project")
    assert captured["openai_key"] == "key"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9000
    assert captured["log_level"] == "info"


def test_serve_command_defaults_path_to_current_directory(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_app(*, project_root: Path, openai_key: str | None):
        captured["project_root"] = project_root
        captured["openai_key"] = openai_key
        return object()

    def fake_run(app, host: str, port: int, log_level: str) -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr(cli, "create_app", fake_create_app)
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    monkeypatch.setattr(
        cli, "resolve_openai_api_key", lambda key, project_root=None: "key"
    )

    result = runner.invoke(cli.cli_app, ["serve"])

    assert result.exit_code == 0
    assert captured["project_root"] == Path(".")
    assert captured["openai_key"] == "key"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000
    assert captured["log_level"] == "info"


def test_init_command_defaults_openai_model_to_gpt_54(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRepo:
        def __init__(self, root: Path):
            captured["root"] = root

        def init_project(self, *, name: str, course: str, openai_model: str) -> None:
            captured["name"] = name
            captured["course"] = course
            captured["openai_model"] = openai_model

    monkeypatch.setattr(cli, "ProjectRepository", FakeRepo)

    result = runner.invoke(cli.cli_app, ["init", "my-project"])

    assert result.exit_code == 0
    assert captured["root"] == Path("my-project")
    assert captured["name"] == "Example Exam Project"
    assert captured["course"] == "Calculus-based Intro Physics"
    assert captured["openai_model"] == DEFAULT_OPENAI_MODEL


def test_export_docx_command_defaults_include_solutions(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_export_project_to_docx(
        *, project_root: Path, output_path: Path, include_solutions: bool
    ) -> list[str]:
        captured["project_root"] = project_root
        captured["output_path"] = output_path
        captured["include_solutions"] = include_solutions
        return []

    monkeypatch.setattr(cli, "export_project_to_docx", fake_export_project_to_docx)

    output = tmp_path / "exam.docx"
    result = runner.invoke(
        cli.cli_app, ["export", "docx", "my-project", "--output", str(output)]
    )

    assert result.exit_code == 0
    assert captured["project_root"] == Path("my-project")
    assert captured["output_path"] == output
    assert captured["include_solutions"] is False
