from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
import yaml

from exam_helper.models import (
    AIPromptConfig,
    AIUsageTotals,
    DistractorFunction,
    Question,
)
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
        input_tokens = self._to_int(
            data.get("input_tokens") or data.get("prompt_tokens")
        )
        output_tokens = self._to_int(
            data.get("output_tokens") or data.get("completion_tokens")
        )
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

    def _text_with_question_context(
        self, bundle: PromptBundle, question: Question
    ) -> AIResult:
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
        text = (raw or "").strip()
        if not text:
            raise ValueError("AI response was empty.")

        def _as_dict(candidate: Any) -> dict[str, Any] | None:
            return candidate if isinstance(candidate, dict) else None

        try:
            data = _as_dict(json.loads(text))
            if data is not None:
                return data
        except json.JSONDecodeError:
            pass

        for pattern in (r"```(?:json)?\s*(\{.*?\})\s*```", r"```(?:yaml|yml)\s*(.*?)\s*```"):
            fence_match = re.search(pattern, text, flags=re.DOTALL)
            if not fence_match:
                continue
            fenced = fence_match.group(1)
            try:
                data = _as_dict(json.loads(fenced))
                if data is not None:
                    return data
            except json.JSONDecodeError:
                pass
            try:
                data = _as_dict(yaml.safe_load(fenced))
                if data is not None:
                    return data
            except Exception:
                pass

        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
            try:
                data = _as_dict(json.loads(candidate))
                if data is not None:
                    return data
            except json.JSONDecodeError:
                pass
            try:
                data = _as_dict(yaml.safe_load(candidate))
                if data is not None:
                    return data
            except Exception:
                pass

        try:
            data = _as_dict(yaml.safe_load(text))
            if data is not None:
                return data
        except Exception:
            pass

        sample = re.sub(r"\s+", " ", text)[:500]
        raise ValueError(
            "AI response was not parseable as an object. "
            f"Sample: {sample}"
        )

    @staticmethod
    def _extract_typed_solution_text(raw: str) -> str:
        text = (raw or "").strip()
        if not text:
            return ""
        # Try strict/heuristic JSON object parsing first.
        try:
            payload = AIService._parse_json_object(text)
            value = payload.get("typed_solution_md")
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception:
            pass
        # Try YAML/JSON loading for near-JSON model outputs.
        try:
            loaded = yaml.safe_load(text)
            if isinstance(loaded, dict):
                value = loaded.get("typed_solution_md")
                if isinstance(value, str) and value.strip():
                    return value.strip()
        except Exception:
            pass
        return text

    @staticmethod
    def _coerce_parameters_object(raw_params: Any) -> dict[str, Any]:
        if raw_params is None:
            return {}
        if isinstance(raw_params, dict):
            return raw_params
        if isinstance(raw_params, str):
            text = raw_params.strip()
            if not text:
                return {}
            for loader in (json.loads, yaml.safe_load):
                try:
                    parsed = loader(text)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            sample = re.sub(r"\s+", " ", text)[:200]
            raise ValueError(
                "AI response field 'parameters' was a string but could not be "
                f"parsed as an object. Sample: {sample}"
            )
        if isinstance(raw_params, list):
            out: dict[str, Any] = {}
            for idx, item in enumerate(raw_params):
                if not isinstance(item, dict):
                    raise ValueError(
                        "AI response field 'parameters' list entries must be "
                        "objects with name/key and value."
                    )
                key = item.get("name", item.get("key"))
                if not isinstance(key, str) or not key.strip():
                    raise ValueError(
                        "AI response field 'parameters' list entry is missing a "
                        f"valid name/key at index {idx}."
                    )
                if "value" in item:
                    value = item["value"]
                elif "val" in item:
                    value = item["val"]
                else:
                    raise ValueError(
                        "AI response field 'parameters' list entry is missing a "
                        f"value at index {idx}."
                    )
                out[key] = value
            return out
        raise ValueError(
            "AI response field 'parameters' must be an object or a coercible "
            f"string/list, got {type(raw_params).__name__}."
        )

    @staticmethod
    def _coerce_numeric_scalar(value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return value

        if re.fullmatch(r"[+-]?\d+", text):
            try:
                return int(text)
            except Exception:
                return value

        if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", text):
            try:
                return float(text)
            except Exception:
                return value

        frac_match = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", text)
        if frac_match:
            denom = int(frac_match.group(2))
            if denom != 0:
                return int(frac_match.group(1)) / denom
            return value

        latex_frac_match = re.fullmatch(
            r"\\(?:t?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", text
        )
        if latex_frac_match:
            denom = int(latex_frac_match.group(2))
            if denom != 0:
                return int(latex_frac_match.group(1)) / denom
        return value

    @classmethod
    def _normalize_rewrite_parameters(
        cls,
        params: dict[str, Any],
        question: Question,
        template: str,
    ) -> tuple[dict[str, Any], str]:
        figure_ids = {f.id for f in (question.figures or [])}
        filtered: dict[str, Any] = {}
        rewritten_template = template

        for key, value in (params or {}).items():
            key_clean = str(key).strip()
            if not key_clean:
                continue
            key_lower = key_clean.casefold()
            is_figure_ref_key = (
                key_lower
                in {"figure_ref", "fig_ref", "figure_id", "fig_id", "image_ref", "image_id"}
                or (key_lower.startswith("figure_") and key_lower.endswith("_ref"))
                or (key_lower.startswith("fig_") and key_lower.endswith("_ref"))
            )
            value_str = str(value).strip() if isinstance(value, str) else ""
            looks_like_figure_id = bool(re.fullmatch(r"fig_[A-Za-z0-9_-]+", value_str))
            if is_figure_ref_key and (value_str in figure_ids or looks_like_figure_id):
                if value_str:
                    rewritten_template = re.sub(
                        r"\{\{\s*" + re.escape(key_clean) + r"\s*\}\}",
                        value_str,
                        rewritten_template,
                    )
                continue
            filtered[key_clean] = cls._coerce_numeric_scalar(value)

        return filtered, rewritten_template

    def rewrite_parameterize(self, question: Question) -> RewriteResult:
        bundle = self.compose_prompt(action="rewrite_parameterize", question=question)
        result = self._text_with_question_context(bundle, question)
        payload = self._parse_json_object(result.text)
        template = str(payload.get("question_template_md", "")).strip()
        title = str(payload.get("title", "")).strip()
        params = self._coerce_parameters_object(payload.get("parameters"))
        params, template = self._normalize_rewrite_parameters(
            params=params,
            question=question,
            template=template,
        )
        if not template:
            raise ValueError("AI response field 'question_template_md' is required.")
        return AIService.RewriteResult(
            question_template_md=template,
            parameters=params,
            title=title,
            usage=result.usage,
        )

    def generate_answer_function(
        self,
        question: Question,
        error_feedback: str = "",
    ) -> AnswerFunctionResult:
        bundle = self.compose_prompt(
            action="generate_answer_function", question=question
        )
        if error_feedback.strip():
            bundle = PromptBundle(
                system_prompt=bundle.system_prompt,
                user_prompt=f"{bundle.user_prompt}\n\nPrevious execution error:\n{error_feedback.strip()}",
            )
        result = self._text_with_question_context(bundle, question)
        payload = self._parse_json_object(result.text)
        answer_python_code = str(payload.get("answer_python_code", "")).strip()
        if not answer_python_code:
            raise ValueError("AI response field 'answer_python_code' is required.")
        return AIService.AnswerFunctionResult(
            answer_python_code=answer_python_code,
            usage=result.usage,
        )

    def generate_distractor_functions(
        self, question: Question
    ) -> DistractorFunctionsResult:
        bundle = self.compose_prompt(
            action="generate_distractor_functions", question=question
        )
        result = self._text_with_question_context(bundle, question)
        payload = self._parse_json_object(result.text)
        raw = payload.get("distractors")
        if not isinstance(raw, list) or len(raw) != 4:
            raise ValueError(
                "AI response field 'distractors' must be a list with exactly 4 entries."
            )
        distractors = [DistractorFunction.model_validate(item) for item in raw]
        return AIService.DistractorFunctionsResult(
            distractors=distractors, usage=result.usage
        )

    def generate_typed_solution(self, question: Question) -> AIResult:
        bundle = self.compose_prompt(
            action="generate_typed_solution", question=question
        )
        result = self._text_with_question_context(bundle, question)
        text = self._extract_typed_solution_text(result.text)
        if not text:
            raise ValueError("AI response field 'typed_solution_md' is required.")
        return AIService.AIResult(text=text, usage=result.usage)
