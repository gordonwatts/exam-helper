from __future__ import annotations

from pathlib import Path

import pytest

from exam_helper import cli
from exam_helper.models import DEFAULT_OPENAI_MODEL


def test_serve_parser_accepts_positional_path() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["serve", "my-project"])

    assert args.path == "my-project"


def test_serve_parser_defaults_path_to_current_directory() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["serve"])

    assert args.path == "."


def test_serve_help_mentions_project_path_and_openai_env(capsys) -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "-h"])

    out = capsys.readouterr().out
    assert "project.yaml" in out
    assert "Path to the exam project directory." in out
    assert "EXAM_HELPER_OPENAI_KEY in ~/.env" in out
    assert "EXAM_HELPER_OPENAI_KEY in the project .env path" in out
    assert "--openai-key OPENAI_KEY" in out


def test_cmd_serve_uses_path_for_project_root(monkeypatch) -> None:
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

    parser = cli.build_parser()
    args = parser.parse_args(["serve", "my-project", "--port", "9000"])

    assert cli.cmd_serve(args) == 0
    assert captured["project_root"] == Path("my-project")
    assert captured["openai_key"] == "key"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9000
    assert captured["log_level"] == "info"


def test_init_parser_defaults_openai_model_to_gpt_54() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["init", "my-project"])

    assert args.openai_model == DEFAULT_OPENAI_MODEL
