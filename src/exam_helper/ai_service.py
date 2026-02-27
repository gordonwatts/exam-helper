from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from exam_helper.models import AIPromptConfig, AIUsageTotals, MCChoice, Question
from exam_helper.prompt_catalog import PromptBundle, PromptCatalog


@dataclass
class AIService:
    api_key: str | None
    model: str = "gpt-5.2"
    prompts_override: AIPromptConfig | None = None
    prompt_catalog: PromptCatalog | None = None

    @dataclass
    class AIResult:
        text: str
        usage: AIUsageTotals

    @dataclass
    class SolutionDraft:
        worked_solution_md: str
        python_code: str
        parameters: dict[str, Any]
        usage: AIUsageTotals
        raw_text: str

    def _client(self) -> OpenAI:
        if not self.api_key:
            raise ValueError("OpenAI API key is not configured.")
        return OpenAI(api_key=self.api_key)

    def _catalog(self) -> PromptCatalog:
        if self.prompt_catalog is None:
            self.prompt_catalog = PromptCatalog.from_package_yaml()
        return self.prompt_catalog

    def list_models(self) -> list[str]:
        client = self._client()
        response = client.models.list()
        ids = sorted(
            {
                str(getattr(item, "id", "")).strip()
                for item in getattr(response, "data", [])
                if str(getattr(item, "id", "")).strip()
            }
        )
        return ids

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

    def compose_prompt(self, action: str, question: Question, solution_md: str = "") -> PromptBundle:
        return self._catalog().compose(
            action=action,
            question=question,
            prompts_override=self.prompts_override,
            solution_md=solution_md,
        )

    def preview_prompt(self, action: str, question: Question, solution_md: str = "") -> dict[str, Any]:
        bundle = self.compose_prompt(action=action, question=question, solution_md=solution_md)
        return {
            "action": action,
            "system_prompt": bundle.system_prompt,
            "user_prompt": bundle.user_prompt,
            "figure_placeholders": self._catalog().figure_placeholders(question),
        }

    def _usage_from_response(self, response: Any) -> AIUsageTotals:
        usage = getattr(response, "usage", None)
        if usage is None:
            return AIUsageTotals()
        if hasattr(usage, "model_dump"):
            data = usage.model_dump()
        elif isinstance(usage, dict):
            data = usage
        else:
            data = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "total_cost": getattr(usage, "total_cost", None),
            }
        input_tokens = self._to_int(data.get("input_tokens") or data.get("prompt_tokens"))
        output_tokens = self._to_int(data.get("output_tokens") or data.get("completion_tokens"))
        total_tokens = self._to_int(data.get("total_tokens"))
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens
        total_cost = self._extract_total_cost_usd(data)
        return AIUsageTotals(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
        )

    @staticmethod
    def _to_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, str):
            cleaned = re.sub(r"[^0-9eE+\-\.]", "", value.strip())
            if not cleaned:
                return 0.0
            value = cleaned
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _extract_total_cost_usd(cls, data: dict[str, Any]) -> float:
        direct_keys = [
            "total_cost_usd",
            "total_cost",
            "cost",
            "estimated_cost",
            "cost_usd",
            "usd_cost",
        ]
        for key in direct_keys:
            value = cls._to_float(data.get(key))
            if value > 0:
                return value

        input_candidates = ["input_cost", "prompt_cost", "input_cost_usd", "prompt_cost_usd"]
        output_candidates = ["output_cost", "completion_cost", "output_cost_usd", "completion_cost_usd"]
        input_cost = max(cls._to_float(data.get(k)) for k in input_candidates)
        output_cost = max(cls._to_float(data.get(k)) for k in output_candidates)
        split_total = input_cost + output_cost
        if split_total > 0:
            return split_total

        details = data.get("cost_details")
        if isinstance(details, dict):
            nested_total = cls._extract_total_cost_usd(details)
            if nested_total > 0:
                return nested_total
        return 0.0

    def _text(self, bundle: PromptBundle) -> AIResult:
        client = self._client()
        user_content = [{"type": "input_text", "text": bundle.user_prompt}]
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": bundle.system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        text = getattr(response, "output_text", "").strip()
        if not text:
            raise ValueError("Empty AI response.")
        return AIService.AIResult(text=text, usage=self._usage_from_response(response))

    def _text_with_question_context(self, bundle: PromptBundle, question: Question) -> AIResult:
        client = self._client()
        user_content = [{"type": "input_text", "text": bundle.user_prompt}]
        user_content.extend(self._figure_content(question))
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": bundle.system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        text = getattr(response, "output_text", "").strip()
        if not text:
            raise ValueError("Empty AI response.")
        return AIService.AIResult(text=text, usage=self._usage_from_response(response))

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

    def improve_prompt(self, question: Question) -> AIResult:
        bundle = self.compose_prompt(action="improve_prompt", question=question)
        return self._text_with_question_context(bundle, question)

    def draft_solution(self, question: Question) -> AIResult:
        bundle = self.compose_prompt(
            action="draft_solution",
            question=question,
            solution_md=question.solution.worked_solution_md,
        )
        return self._text_with_question_context(bundle, question)

    def suggest_title(self, question: Question) -> AIResult:
        bundle = self.compose_prompt(action="suggest_title", question=question)
        return self._text_with_question_context(bundle, question)

    def generate_mc_options(self, question: Question) -> list[MCChoice]:
        choices, _ = self.generate_mc_options_from_solution(question, question.solution.worked_solution_md)
        return choices

    def draft_solution_with_code(
        self,
        question: Question,
        error_feedback: str = "",
    ) -> SolutionDraft:
        bundle = self.compose_prompt(
            action="draft_solution_with_code",
            question=question,
            solution_md=question.solution.worked_solution_md,
        )
        if error_feedback.strip():
            bundle = PromptBundle(
                system_prompt=bundle.system_prompt,
                user_prompt=f"{bundle.user_prompt}\n\nPrevious execution error:\n{error_feedback.strip()}",
            )
        result = self._text_with_question_context(bundle, question)
        try:
            payload = self._parse_json_object(result.text)
        except Exception as ex:
            excerpt = result.text[:800].replace("\r", " ").replace("\n", "\\n")
            raise ValueError(
                f"AI structured solution response was not valid JSON object. Raw excerpt: {excerpt}"
            ) from ex
        worked_solution_md = str(payload.get("worked_solution_md", "")).strip()
        python_code = str(payload.get("python_code", "")).strip()
        parameters_raw = payload.get("parameters") or {}
        if not isinstance(parameters_raw, dict):
            raise ValueError("AI response field 'parameters' must be an object.")
        if not worked_solution_md:
            raise ValueError("AI response field 'worked_solution_md' is required.")
        if not python_code:
            raise ValueError("AI response field 'python_code' is required.")
        return AIService.SolutionDraft(
            worked_solution_md=worked_solution_md,
            python_code=python_code,
            parameters=parameters_raw,
            usage=result.usage,
            raw_text=result.text,
        )

    def sync_parameters_draft(self, question: Question) -> tuple[dict[str, Any], AIUsageTotals]:
        bundle = self.compose_prompt(
            action="sync_parameters",
            question=question,
            solution_md=question.solution.worked_solution_md,
        )
        result = self._text_with_question_context(bundle, question)
        payload = self._parse_json_object(result.text)
        prompt_md = str(payload.get("prompt_md", "")).strip()
        worked_solution_md = str(payload.get("worked_solution_md", "")).strip()
        if not prompt_md:
            raise ValueError("AI response field 'prompt_md' is required.")
        if not worked_solution_md:
            raise ValueError("AI response field 'worked_solution_md' is required.")
        return {"prompt_md": prompt_md, "worked_solution_md": worked_solution_md}, result.usage

    def generate_mc_options_from_solution(
        self, question: Question, solution_md: str
    ) -> tuple[list[MCChoice], AIUsageTotals]:
        bundle = self.compose_prompt(action="distractors", question=question, solution_md=solution_md)
        result = self._text_with_question_context(bundle, question)
        items = self._parse_json_payload(result.text)
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
        return out, result.usage
