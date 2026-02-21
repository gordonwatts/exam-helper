from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from exam_helper.models import MCChoice, Question


@dataclass
class AIService:
    api_key: str | None
    model: str = "gpt-4.1-mini"

    def _client(self) -> OpenAI:
        if not self.api_key:
            raise ValueError("OpenAI API key is not configured.")
        return OpenAI(api_key=self.api_key)

    def _text(self, system_prompt: str, user_prompt: str) -> str:
        client = self._client()
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = getattr(response, "output_text", "").strip()
        if not text:
            raise ValueError("Empty AI response.")
        return text

    def improve_prompt(self, question: Question) -> str:
        return self._text(
            "You improve physics exam problem statements for clarity and precision.",
            f"Improve this prompt while preserving intent and difficulty:\n{question.prompt_md}",
        )

    def draft_solution(self, question: Question) -> str:
        return self._text(
            "You write concise but complete worked solutions for calculus-based physics exams.",
            f"Write a worked solution for:\nTitle: {question.title}\nPrompt:\n{question.prompt_md}",
        )

    def distractors(self, question: Question, count: int = 3) -> list[MCChoice]:
        raw = self._text(
            (
                "Generate plausible incorrect multiple-choice distractors for intro physics."
                " Return strict JSON array of objects with keys label, content_md, rationale."
            ),
            (
                f"Question prompt:\n{question.prompt_md}\n"
                f"Generate {count} distractors. Do not include correct answer."
            ),
        )
        items = json.loads(raw)
        out: list[MCChoice] = []
        for item in items:
            out.append(
                MCChoice(
                    label=item["label"],
                    content_md=item["content_md"],
                    is_correct=False,
                    rationale=item.get("rationale"),
                )
            )
        return out
