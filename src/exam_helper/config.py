from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    openai_api_key: str | None = Field(default=None, alias="EXAM_HELPER_OPENAI_KEY")


def load_home_env() -> dict[str, str]:
    env_path = Path.home() / ".env"
    if not env_path.exists():
        return {}
    raw = dotenv_values(env_path)
    return {k: v for k, v in raw.items() if isinstance(v, str)}


def resolve_openai_api_key(cli_key: str | None) -> str | None:
    if cli_key:
        return cli_key
    home_env = load_home_env()
    return home_env.get("EXAM_HELPER_OPENAI_KEY")
