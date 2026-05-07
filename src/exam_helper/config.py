from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    openai_api_key: str | None = Field(default=None, alias="EXAM_HELPER_OPENAI_KEY")


def _load_env_file(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        return {}
    raw = dotenv_values(env_path)
    return {k: v for k, v in raw.items() if isinstance(v, str)}


def load_home_env() -> dict[str, str]:
    return _load_env_file(Path.home() / ".env")


def load_local_env(project_root: Path | None) -> dict[str, str]:
    base_path = Path.cwd() if project_root is None else Path(project_root)
    if base_path.is_file():
        base_path = base_path.parent
    resolved = base_path.resolve(strict=False)
    env: dict[str, str] = {}
    for ancestor in list(resolved.parents)[::-1] + [resolved]:
        env.update(_load_env_file(ancestor / ".env"))
    return env


def resolve_openai_api_key(
    cli_key: str | None, project_root: Path | None = None
) -> str | None:
    if cli_key:
        return cli_key
    home_env = load_home_env()
    local_env = load_local_env(project_root)
    key_name = "EXAM_HELPER_OPENAI_KEY"
    if key_name in local_env:
        return local_env[key_name]
    if key_name in home_env:
        return home_env[key_name]
    return None
