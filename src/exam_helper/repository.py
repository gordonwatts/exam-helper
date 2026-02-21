from __future__ import annotations

from pathlib import Path

import yaml

from exam_helper.models import AIUsageTotals, ProjectConfig, Question


class ProjectRepository:
    def __init__(self, root: Path):
        self.root = root
        self.questions_dir = self.root / "questions"
        self.project_file = self.root / "project.yaml"

    def ensure_layout(self) -> None:
        self.questions_dir.mkdir(parents=True, exist_ok=True)

    def init_project(self, name: str, course: str, openai_model: str = "gpt-5.2") -> None:
        self.ensure_layout()
        cfg = ProjectConfig(name=name, course=course)
        cfg.ai.model = openai_model
        self.project_file.write_text(
            yaml.safe_dump(cfg.model_dump(), sort_keys=False), encoding="utf-8"
        )

    def load_project(self) -> ProjectConfig:
        raw = yaml.safe_load(self.project_file.read_text(encoding="utf-8"))
        return ProjectConfig.model_validate(raw)

    def save_project(self, project: ProjectConfig) -> None:
        self.ensure_layout()
        self.project_file.write_text(
            yaml.safe_dump(project.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )

    def list_questions(self) -> list[Question]:
        items: list[Question] = []
        for q_file in sorted(self.questions_dir.glob("*.yaml")):
            raw = yaml.safe_load(q_file.read_text(encoding="utf-8"))
            items.append(Question.model_validate(raw))
        return items

    def get_question(self, question_id: str) -> Question:
        q_file = self.questions_dir / f"{question_id}.yaml"
        raw = yaml.safe_load(q_file.read_text(encoding="utf-8"))
        return Question.model_validate(raw)

    def save_question(self, question: Question) -> None:
        self.ensure_layout()
        q_file = self.questions_dir / f"{question.id}.yaml"
        q_file.write_text(
            yaml.safe_dump(question.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )

    def validate_all(self) -> list[str]:
        errors: list[str] = []
        if not self.project_file.exists():
            errors.append("Missing project.yaml")
            return errors
        try:
            self.load_project()
        except Exception as ex:
            errors.append(f"project.yaml invalid: {ex}")
        for q_file in sorted(self.questions_dir.glob("*.yaml")):
            try:
                raw = yaml.safe_load(q_file.read_text(encoding="utf-8"))
                Question.model_validate(raw)
            except Exception as ex:
                errors.append(f"{q_file.name}: {ex}")
        return errors

    def add_ai_usage(self, delta: AIUsageTotals) -> None:
        project = self.load_project()
        project.ai.usage.input_tokens += max(0, int(delta.input_tokens))
        project.ai.usage.output_tokens += max(0, int(delta.output_tokens))
        project.ai.usage.total_tokens += max(0, int(delta.total_tokens))
        project.ai.usage.total_cost_usd += max(0.0, float(delta.total_cost_usd))
        self.save_project(project)

    def reset_ai_usage(self) -> None:
        project = self.load_project()
        project.ai.usage = AIUsageTotals()
        self.save_project(project)
