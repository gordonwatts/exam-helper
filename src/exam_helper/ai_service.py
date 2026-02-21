from __future__ import annotations

import json
import re
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

    @staticmethod
    def _figure_content(question: Question) -> list[dict]:
        items: list[dict] = []
        for fig in question.figures:
            caption = fig.caption or fig.id
            items.append({"type": "input_text", "text": f"Figure {fig.id}: {caption}"})
            items.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{fig.mime_type};base64,{fig.data_base64}",
                    "detail": "low",
                }
            )
        return items

    def _text(self, system_prompt: str, user_prompt: str) -> str:
        client = self._client()
        user_content = [{"type": "input_text", "text": user_prompt}]
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        text = getattr(response, "output_text", "").strip()
        if not text:
            raise ValueError("Empty AI response.")
        return text

    def _text_with_question_context(self, system_prompt: str, user_prompt: str, question: Question) -> str:
        client = self._client()
        user_content = [{"type": "input_text", "text": user_prompt}]
        user_content.extend(self._figure_content(question))
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        text = getattr(response, "output_text", "").strip()
        if not text:
            raise ValueError("Empty AI response.")
        return text

    @staticmethod
    def _parse_json_payload(raw: str) -> list[dict]:
        # First try strict JSON.
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Try fenced code block JSON.
        fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, flags=re.DOTALL)
        if fence_match:
            data = json.loads(fence_match.group(1))
            if isinstance(data, list):
                return data

        # Try first list-like JSON substring.
        start = raw.find("[")
        end = raw.rfind("]")
        if start >= 0 and end > start:
            candidate = raw[start : end + 1]
            data = json.loads(candidate)
            if isinstance(data, list):
                return data

        raise ValueError("AI response was not parseable JSON list.")

    def improve_prompt(self, question: Question) -> str:
        return self._text_with_question_context(
            "You improve physics exam problem statements for clarity and precision.",
            f"Improve this prompt while preserving intent and difficulty:\n{question.prompt_md}",
            question,
        )

    def draft_solution(self, question: Question) -> str:
        return self._text_with_question_context(
            "You write concise but complete worked solutions for calculus-based physics exams.",
            f"Write a worked solution for:\nTitle: {question.title}\nPrompt:\n{question.prompt_md}",
            question,
        )

    def suggest_title(self, question: Question) -> str:
        return self._text_with_question_context(
            "You generate short, descriptive titles for physics exam problems.",
            f"Generate a concise title (max 8 words) for:\n{question.prompt_md}",
            question,
        )

    def generate_mc_options(self, question: Question) -> list[MCChoice]:
        raw = self._text_with_question_context(
            (
                "Generate complete multiple-choice options for a physics question. "
                "Return strict JSON array with exactly 5 objects (A-E) with keys: "
                "label, content_md, is_correct, rationale."
            ),
            (
                f"Question prompt:\n{question.prompt_md}\n"
                "Generate exactly options A, B, C, D, E. "
                "You may infer and mark one correct answer using is_correct=true."
            ),
            question,
        )
        items = self._parse_json_payload(raw)
        if len(items) != 5:
            raise ValueError("AI did not return exactly five options (A-E).")
        out: list[MCChoice] = []
        valid_labels = ["A", "B", "C", "D", "E"]
        for item in items:
            label = str(item.get("label", "")).strip().upper()
            if label not in valid_labels:
                raise ValueError("AI options must use labels A-E.")
            out.append(
                MCChoice(
                    label=label,
                    content_md=item["content_md"],
                    is_correct=bool(item.get("is_correct", False)),
                    rationale=item.get("rationale"),
                )
            )
        if sum(1 for c in out if c.is_correct) != 1:
            raise ValueError("AI must mark exactly one correct option.")
        out.sort(key=lambda c: c.label)
        return out
