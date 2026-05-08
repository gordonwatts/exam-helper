from __future__ import annotations

from pathlib import Path

from exam_helper.config import resolve_openai_api_key


def test_cli_key_wins(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".env").write_text(
        "EXAM_HELPER_OPENAI_KEY=from_env\n", encoding="utf-8"
    )
    assert resolve_openai_api_key("from_cli") == "from_cli"


def test_home_env_used_when_cli_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".env").write_text(
        "EXAM_HELPER_OPENAI_KEY=from_env\n", encoding="utf-8"
    )
    assert resolve_openai_api_key(None) == "from_env"


def test_local_env_overrides_home_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    home = tmp_path / "home"
    home.mkdir()
    project_root = tmp_path / "project" / "nested"
    project_root.mkdir(parents=True)
    (home / ".env").write_text("EXAM_HELPER_OPENAI_KEY=from_home\n", encoding="utf-8")
    (project_root.parent / ".env").write_text(
        "EXAM_HELPER_OPENAI_KEY=from_local\n", encoding="utf-8"
    )

    assert resolve_openai_api_key(None, project_root) == "from_local"
