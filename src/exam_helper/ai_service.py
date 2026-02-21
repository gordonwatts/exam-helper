from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

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

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any]:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
        if fence_match:
            data = json.loads(fence_match.group(1))
            if isinstance(data, dict):
                return data

        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            candidate = raw[start : end + 1]
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data

        raise ValueError("AI response was not parseable JSON object.")

    def improve_prompt(self, question: Question) -> str:
        return self._text_with_question_context(
            "You improve physics exam problem statements for clarity and precision.",
            f"Improve this prompt while preserving intent and difficulty:\n{question.prompt_md}",
            question,
        )

    def draft_solution(self, question: Question) -> str:
        return self._text_with_question_context(
            (
                "You write brief but complete worked solutions for calculus-based physics exams. "
                "Be concise: target 3-5 short lines plus one final answer line unless complexity requires more. "
                "Every mathematical expression, equation, or scientific notation must be written in "
                "LaTeX inline math mode using \\( ... \\). "
                "Do not repeat the problem statement or include any 'Problem (verbatim):' line. "
                "End with a line that starts exactly with 'Final answer: ' and provide the computed final result."
            ),
            (
                "Write a worked solution.\n"
                f"Title: {question.title}\n"
                "Prompt:\n"
                f"{question.prompt_md}"
            ),
            question,
        )

    def suggest_title(self, question: Question) -> str:
        return self._text_with_question_context(
            "You generate short, descriptive titles for physics exam problems.",
            f"Generate a concise title (max 8 words) for:\n{question.prompt_md}",
            question,
        )

    def generate_mc_options(self, question: Question) -> list[MCChoice]:
        return self.generate_mc_options_from_solution(question, question.solution.worked_solution_md)

    def generate_mc_options_from_solution(self, question: Question, solution_md: str) -> list[MCChoice]:
        raw = self._text_with_question_context(
            (
                "Generate complete multiple-choice options from an existing worked solution. "
                "Return strict JSON array with exactly 5 objects (A-E). "
                "Each object must have keys: label, content_md, is_correct, rationale. "
                "Exactly one option must be correct. "
                "Do not return any additional keys or narrative text."
            ),
            (
                f"Question prompt:\n{question.prompt_md}\n"
                f"Worked solution (authoritative, use this to determine final answer):\n{solution_md}\n"
                "Generate exactly options A, B, C, D, E."
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
