from __future__ import annotations

from enum import Enum
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class QuestionType(str, Enum):
    free_response = "free_response"
    multiple_choice = "multiple_choice"


class FigureData(BaseModel):
    id: str
    mime_type: str
    data_base64: str
    sha256: str
    caption: str | None = None

    @model_validator(mode="after")
    def validate_hash(self) -> "FigureData":
        import base64

        raw = base64.b64decode(self.data_base64.encode("ascii"))
        digest = sha256(raw).hexdigest()
        if digest != self.sha256:
            raise ValueError(f"Figure hash mismatch for {self.id}")
        return self


class MCChoice(BaseModel):
    label: str
    content_md: str
    is_correct: bool = False
    rationale: str | None = None


class DistractorFunction(BaseModel):
    id: str
    python_code: str = ""
    label_hint: str | None = None

    @field_validator("id")
    @classmethod
    def id_safe(cls, v: str) -> str:
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("distractor id must be alphanumeric, underscore, or hyphen")
        return v


class Solution(BaseModel):
    question_template_md: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    answer_guidance: str = ""
    answer_python_code: str = ""
    distractor_python_code: list[DistractorFunction] = Field(default_factory=list)
    typed_solution_md: str = ""
    typed_solution_status: Literal["missing", "fresh", "stale"] = "missing"
    last_computed_answer_md: str = ""


class Question(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str = ""
    topic: str = ""
    course_level: str = "intro"
    tags: list[str] = Field(default_factory=list)
    difficulty: int = 3
    points: int = 5
    question_type: QuestionType = QuestionType.free_response
    mc_options_guidance: str = ""
    figures: list[FigureData] = Field(default_factory=list)
    choices: list[MCChoice] = Field(default_factory=list)
    solution: Solution = Field(default_factory=Solution)

    @field_validator("id")
    @classmethod
    def id_safe(cls, v: str) -> str:
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("id must be alphanumeric, underscore, or hyphen")
        return v

    @model_validator(mode="after")
    def check_mc(self) -> "Question":
        if self.question_type == QuestionType.multiple_choice and self.choices:
            if len(self.choices) != 5:
                raise ValueError("multiple_choice choices must define exactly five options (A-E)")
            if sum(1 for c in self.choices if c.is_correct) != 1:
                raise ValueError("multiple_choice requires exactly one correct choice")
        return self


class AIPromptConfig(BaseModel):
    overall: str = ""
    solution_and_mc: str = ""
    prompt_review: str = ""


class AIUsageTotals(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0


class AIConfig(BaseModel):
    model: str = "gpt-5.2"
    prompts: AIPromptConfig = Field(default_factory=AIPromptConfig)
    usage: AIUsageTotals = Field(default_factory=AIUsageTotals)


class ProjectConfig(BaseModel):
    name: str
    course: str
    ai: AIConfig = Field(default_factory=AIConfig)
