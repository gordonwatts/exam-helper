from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from exam_helper.models import AIPromptConfig, AIUsageTotals, DistractorFunction, Question
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
    class RewriteResult:
        question_template_md: str
        parameters: dict[str, Any]
        title: str
        usage: AIUsageTotals

    @dataclass
    class AnswerFunctionResult:
        answer_python_code: str
        usage: AIUsageTotals

    @dataclass
    class DistractorFunctionsResult:
        distractors: list[DistractorFunction]
        usage: AIUsageTotals

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

    def compose_prompt(self, action: str, question: Question) -> PromptBundle:
        return self._catalog().compose(
            action=action,
            question=question,
            prompts_override=self.prompts_override,
        )

    def preview_prompt(self, action: str, question: Question) -> dict[str, Any]:
        bundle = self.compose_prompt(action=action, question=question)
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
        for key in ("total_cost_usd", "total_cost", "cost"):
            value = cls._to_float(data.get(key))
            if value > 0:
                return value
        input_cost = cls._to_float(data.get("input_cost"))
        output_cost = cls._to_float(data.get("output_cost"))
        return max(0.0, input_cost + output_cost)

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

    def rewrite_parameterize(self, question: Question) -> RewriteResult:
        bundle = self.compose_prompt(action="rewrite_parameterize", question=question)
        result = self._text_with_question_context(bundle, question)
        payload = self._parse_json_object(result.text)
        template = str(payload.get("question_template_md", "")).strip()
        title = str(payload.get("title", "")).strip()
        params = payload.get("parameters") or {}
        if not isinstance(params, dict):
            raise ValueError("AI response field 'parameters' must be an object.")
        if not template:
            raise ValueError("AI response field 'question_template_md' is required.")
        return AIService.RewriteResult(
            question_template_md=template,
            parameters=params,
            title=title,
            usage=result.usage,
        )

    def generate_answer_function(self, question: Question) -> AnswerFunctionResult:
        bundle = self.compose_prompt(action="generate_answer_function", question=question)
        result = self._text_with_question_context(bundle, question)
        payload = self._parse_json_object(result.text)
        answer_python_code = str(payload.get("answer_python_code", "")).strip()
        if not answer_python_code:
            raise ValueError("AI response field 'answer_python_code' is required.")
        return AIService.AnswerFunctionResult(
            answer_python_code=answer_python_code,
            usage=result.usage,
        )

    def generate_distractor_functions(self, question: Question) -> DistractorFunctionsResult:
        bundle = self.compose_prompt(action="generate_distractor_functions", question=question)
        result = self._text_with_question_context(bundle, question)
        payload = self._parse_json_object(result.text)
        raw = payload.get("distractors")
        if not isinstance(raw, list) or len(raw) != 4:
            raise ValueError("AI response field 'distractors' must be a list with exactly 4 entries.")
        distractors = [DistractorFunction.model_validate(item) for item in raw]
        return AIService.DistractorFunctionsResult(distractors=distractors, usage=result.usage)

    def generate_typed_solution(self, question: Question) -> AIResult:
        bundle = self.compose_prompt(action="generate_typed_solution", question=question)
        result = self._text_with_question_context(bundle, question)
        payload = self._parse_json_object(result.text)
        text = str(payload.get("typed_solution_md", "")).strip()
        if not text:
            raise ValueError("AI response field 'typed_solution_md' is required.")
        return AIService.AIResult(text=text, usage=result.usage)
