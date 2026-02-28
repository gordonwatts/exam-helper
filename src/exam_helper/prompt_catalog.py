from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from string import Formatter
import re

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
        required = {
            "rewrite_parameterize",
            "generate_answer_function",
            "generate_distractor_functions",
            "generate_typed_solution",
        }
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
    ) -> PromptBundle:
        payload = self._actions.get(action)
        if payload is None:
            raise ValueError(f"Unknown prompt action: {action}")
        system_parts = [payload["system_prompt"].strip()]
        if prompts_override:
            if prompts_override.overall.strip():
                system_parts.append(prompts_override.overall.strip())
            action_override = self._action_override_text(action=action, prompts_override=prompts_override)
            if action_override:
                system_parts.append(action_override)
        system_prompt = "\n\n".join(p for p in system_parts if p)
        rendered_question_md = self._render_template_from_parameters(
            question.solution.question_template_md or "",
            question.solution.parameters or {},
        )
        values = {
            "title": question.title or "",
            "prompt_md": rendered_question_md,
            "question_type": question.question_type.value,
            "question_template_md": question.solution.question_template_md or "",
            "solution_parameters_yaml": yaml.safe_dump(
                question.solution.parameters or {}, sort_keys=False
            ).strip(),
            "answer_guidance": question.solution.answer_guidance or "",
            "answer_python_code": question.solution.answer_python_code or "",
            "distractor_functions_text": (
                "\n---\n".join(
                    [
                        f"# distractor: {d.id}\n{(d.python_code or '').strip()}"
                        for d in question.solution.distractor_python_code
                    ]
                ).strip()
            ),
            "typed_solution_md": question.solution.typed_solution_md or "",
            "last_computed_answer_md": question.solution.last_computed_answer_md or "",
        }
        values["context_sections"] = self._build_context_sections(action=action, values=values)
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
    def _safe_format(template: str, values: dict[str, str]) -> str:
        formatter = Formatter()
        allowed = set(values.keys())
        for _, field_name, _, _ in formatter.parse(template):
            if field_name and field_name not in allowed:
                raise ValueError(f"Unsupported template key: {field_name}")
        return template.format(**values)

    @staticmethod
    def _action_override_text(action: str, prompts_override: AIPromptConfig) -> str:
        if action == "rewrite_parameterize":
            return prompts_override.prompt_review.strip()
        return prompts_override.solution_and_mc.strip()

    @classmethod
    def _build_context_sections(cls, *, action: str, values: dict[str, str]) -> str:
        section_specs: dict[str, list[tuple[str, str, str | None]]] = {
            "rewrite_parameterize": [
                ("Question Type", "question_type", None),
                ("Title", "title", None),
                ("Rendered Question (Markdown)", "prompt_md", "markdown"),
                ("Question Template (Markdown)", "question_template_md", "markdown"),
                ("Template Parameters (YAML)", "solution_parameters_yaml", "yaml"),
            ],
            "generate_answer_function": [
                ("Question Type", "question_type", None),
                ("Title", "title", None),
                ("Question Template (Markdown)", "question_template_md", "markdown"),
                ("Template Parameters (YAML)", "solution_parameters_yaml", "yaml"),
                ("Answer Guidance (Markdown)", "answer_guidance", "markdown"),
                ("Answer Function (Python)", "answer_python_code", "python"),
            ],
            "generate_distractor_functions": [
                ("Question Type", "question_type", None),
                ("Title", "title", None),
                ("Question Template (Markdown)", "question_template_md", "markdown"),
                ("Template Parameters (YAML)", "solution_parameters_yaml", "yaml"),
                ("Answer Function (Python)", "answer_python_code", "python"),
                ("Distractor Functions (Python)", "distractor_functions_text", "python"),
            ],
            "generate_typed_solution": [
                ("Question Type", "question_type", None),
                ("Title", "title", None),
                ("Question Template (Markdown)", "question_template_md", "markdown"),
                ("Template Parameters (YAML)", "solution_parameters_yaml", "yaml"),
                ("Answer Function (Python)", "answer_python_code", "python"),
                ("Typed Solution (Markdown)", "typed_solution_md", "markdown"),
                ("Latest Computed Answer (Markdown)", "last_computed_answer_md", "markdown"),
            ],
        }
        if action not in section_specs:
            raise ValueError(f"Unknown prompt action: {action}")
        parts: list[str] = []
        for heading, key, code_lang in section_specs[action]:
            section = cls._format_section(
                heading=heading,
                content=values.get(key, ""),
                code_lang=code_lang,
            )
            if section:
                parts.append(section)
        return "\n\n".join(parts)

    @staticmethod
    def _format_section(*, heading: str, content: str, code_lang: str | None = None) -> str:
        text = (content or "").strip()
        if not text:
            return ""
        if code_lang:
            body = f"```{code_lang}\n{text}\n```"
        else:
            body = text
        return f"{heading}:\n{body}"

    @staticmethod
    def _render_template_from_parameters(template: str, params: dict[str, object]) -> str:
        rendered = template or ""
        for key, value in (params or {}).items():
            rendered = re.sub(r"\{\{\s*" + re.escape(str(key)) + r"\s*\}\}", str(value), rendered)
        return rendered
