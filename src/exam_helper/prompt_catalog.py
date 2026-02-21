from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from string import Formatter

import yaml

from exam_helper.models import AIPromptConfig, Question


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    user_prompt: str


class PromptCatalog:
    def __init__(self, actions: dict[str, dict[str, str]]):
        self._actions = actions

    @classmethod
    def from_package_yaml(cls) -> "PromptCatalog":
        data = files("exam_helper").joinpath("prompt_templates.yaml").read_text(encoding="utf-8")
        raw = yaml.safe_load(data) or {}
        actions = raw.get("actions")
        if not isinstance(actions, dict):
            raise ValueError("prompt_templates.yaml must define 'actions' mapping")
        required = {"suggest_title", "improve_prompt", "draft_solution", "distractors"}
        missing = required.difference(actions.keys())
        if missing:
            raise ValueError(f"prompt_templates.yaml missing actions: {sorted(missing)}")
        for action_name, payload in actions.items():
            if not isinstance(payload, dict):
                raise ValueError(f"Action '{action_name}' must be a mapping")
            if not isinstance(payload.get("system_prompt"), str):
                raise ValueError(f"Action '{action_name}' missing system_prompt")
            if not isinstance(payload.get("user_prompt_template"), str):
                raise ValueError(f"Action '{action_name}' missing user_prompt_template")
        return cls(actions=actions)

    def compose(
        self,
        *,
        action: str,
        question: Question,
        prompts_override: AIPromptConfig | None = None,
        solution_md: str = "",
    ) -> PromptBundle:
        payload = self._actions.get(action)
        if payload is None:
            raise ValueError(f"Unknown prompt action: {action}")
        system_parts = [payload["system_prompt"].strip()]
        if prompts_override:
            if prompts_override.overall.strip():
                system_parts.append(prompts_override.overall.strip())
            scoped = self._scoped_override(action=action, prompts_override=prompts_override)
            if scoped:
                system_parts.append(scoped)
        system_prompt = "\n\n".join(p for p in system_parts if p)
        values = {
            "title": question.title or "",
            "prompt_md": question.prompt_md or "",
            "solution_md": solution_md or "",
        }
        user_template = payload["user_prompt_template"]
        user_prompt = self._safe_format(user_template, values)
        return PromptBundle(system_prompt=system_prompt, user_prompt=user_prompt)

    @staticmethod
    def figure_placeholders(question: Question) -> list[str]:
        out: list[str] = []
        for fig in question.figures:
            caption = (fig.caption or fig.id or "").strip()
            if caption:
                out.append(f"<figure {fig.id}: {caption}>")
            else:
                out.append(f"<figure {fig.id}>")
        return out

    @staticmethod
    def _scoped_override(action: str, prompts_override: AIPromptConfig) -> str:
        if action in {"draft_solution", "distractors"}:
            return prompts_override.solution_and_mc.strip()
        if action == "improve_prompt":
            return prompts_override.prompt_review.strip()
        return ""

    @staticmethod
    def _safe_format(template: str, values: dict[str, str]) -> str:
        formatter = Formatter()
        allowed = {"title", "prompt_md", "solution_md"}
        for _, field_name, _, _ in formatter.parse(template):
            if field_name and field_name not in allowed:
                raise ValueError(f"Unsupported template key: {field_name}")
        return template.format(**values)
