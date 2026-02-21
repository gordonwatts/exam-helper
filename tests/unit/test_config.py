from __future__ import annotations

from pathlib import Path

from exam_helper.config import resolve_openai_api_key


def test_cli_key_wins(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".env").write_text("EXAM_HELPER_OPENAI_KEY=from_env\n", encoding="utf-8")
    assert resolve_openai_api_key("from_cli") == "from_cli"


def test_home_env_used_when_cli_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".env").write_text("EXAM_HELPER_OPENAI_KEY=from_env\n", encoding="utf-8")
    assert resolve_openai_api_key(None) == "from_env"
