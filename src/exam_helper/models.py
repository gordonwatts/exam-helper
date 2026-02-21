from __future__ import annotations

from enum import Enum
from hashlib import sha256

from pydantic import BaseModel, Field, field_validator, model_validator


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


class Solution(BaseModel):
    worked_solution_md: str = ""
    rubric: list[str] = Field(default_factory=list)


class CheckerSpec(BaseModel):
    python_code: str = ""
    sample_answer: dict = Field(default_factory=dict)


class Question(BaseModel):
    id: str
    title: str = ""
    topic: str = ""
    course_level: str = "intro"
    tags: list[str] = Field(default_factory=list)
    difficulty: int = 3
    points: int = 5
    question_type: QuestionType = QuestionType.free_response
    prompt_md: str = ""
    figures: list[FigureData] = Field(default_factory=list)
    choices: list[MCChoice] = Field(default_factory=list)
    solution: Solution = Field(default_factory=Solution)
    checker: CheckerSpec = Field(default_factory=CheckerSpec)

    @field_validator("id")
    @classmethod
    def id_safe(cls, v: str) -> str:
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("id must be alphanumeric, underscore, or hyphen")
        return v

    @model_validator(mode="after")
    def check_mc(self) -> "Question":
        if self.question_type == QuestionType.multiple_choice:
            if len(self.choices) < 2:
                raise ValueError("multiple_choice requires at least two choices")
            if sum(1 for c in self.choices if c.is_correct) < 1:
                raise ValueError("multiple_choice requires at least one correct choice")
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
