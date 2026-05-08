from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
import yaml
from pydantic import BaseModel, Field, model_validator

from exam_helper.models import (
    AIPromptConfig,
    AIUsageTotals,
    ChatTurn,
    DEFAULT_OPENAI_MODEL,
    DistractorFunction,
    MCAnswerSpec,
    Question,
    QuestionType,
)
from exam_helper.parameter_utils import coerce_numeric_scalar
from exam_helper.prompt_catalog import PromptBundle, PromptCatalog
from exam_helper.solution_runtime import (
    SolutionRuntimeError,
    run_answer_formula,
    run_mc_formula_harness,
    run_mc_harness,
)
from exam_helper.validation import validate_question

logger = logging.getLogger(__name__)


@dataclass
class AIService:
    api_key: str | None
    model: str = DEFAULT_OPENAI_MODEL
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
    class AnswerFormulaResult:
        answer_formula_md: str
        usage: AIUsageTotals

    @dataclass
    class DistractorFunctionsResult:
        distractors: list[DistractorFunction]
        usage: AIUsageTotals

    class QuestionEditorState(BaseModel):
        title: str = ""
        question_type: str = "free_response"
        points: int = 5
        mc_options_guidance: str = ""
        figure_ids: list[str] = Field(default_factory=list)
        figure_summaries: list[str] = Field(default_factory=list)
        question_template_md: str = ""
        solution_parameters_yaml: str = "{}"
        answer_formula_md: str = ""
        answer_guidance: str = ""
        distractor_functions_text: str = ""
        mc_answer_specs_json: str = "[]"
        choices_yaml: str = "[]"
        typed_solution_md: str = ""
        typed_solution_status: str = "missing"
        last_computed_answer_md: str = ""

    class ChatAssistantResponse(BaseModel):
        assistant_message: str
        warnings: list[str] = Field(default_factory=list)

        @model_validator(mode="after")
        def _normalize(self) -> "AIService.ChatAssistantResponse":
            self.assistant_message = self.assistant_message.strip()
            self.warnings = [
                str(item).strip() for item in self.warnings if str(item).strip()
            ]
            return self

    class ChatFinishArgs(BaseModel):
        assistant_message: str
        warnings: list[str] = Field(default_factory=list)

        @model_validator(mode="after")
        def _normalize(self) -> "AIService.ChatFinishArgs":
            self.assistant_message = self.assistant_message.strip()
            self.warnings = [
                str(item).strip() for item in self.warnings if str(item).strip()
            ]
            return self

    @dataclass
    class QuestionEditorResult:
        assistant_message: str
        question: Question
        warnings: list[str]
        usage: AIUsageTotals
        changed_fields: list[str]

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
    def _figure_content(
        question: Question, figure_ids: list[str] | None = None
    ) -> list[dict]:
        items: list[dict] = []
        selected = set(figure_ids or [])
        for fig in question.figures:
            if selected and fig.id not in selected:
                continue
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

    @staticmethod
    def _figure_summaries(question: Question) -> list[str]:
        summaries: list[str] = []
        for fig in question.figures:
            # Use author-provided caption first, then fall back to the figure id.
            # This keeps the prompt stable without a second image-summary AI call.
            caption = (fig.caption or fig.id or "").strip() or fig.id
            summaries.append(f"{fig.id}: {caption} ({fig.mime_type})")
        return summaries

    @classmethod
    def _figure_prompt_text(cls, question: Question) -> str:
        summaries = cls._figure_summaries(question)
        if not summaries:
            return ""
        return "Figures associated with this question:\n" + "\n".join(
            f"- {item}" for item in summaries
        )

    @staticmethod
    def _editor_state_for_question(
        question: Question,
    ) -> "AIService.QuestionEditorState":
        def _dump_parameters(params: dict[str, Any]) -> str:
            return yaml.safe_dump(params or {}, sort_keys=False).strip()

        def _dump_distractors(funcs: list[DistractorFunction]) -> str:
            if not funcs:
                return ""
            chunks: list[str] = []
            for f in funcs:
                chunks.append(f"# distractor: {f.id}\n{(f.python_code or '').strip()}")
            return "\n---\n".join(chunks).strip() + "\n"

        def _dump_mc_answer_specs(specs: list[MCAnswerSpec]) -> str:
            return json.dumps(
                [spec.model_dump(mode="json") for spec in specs], ensure_ascii=False
            )

        def _dump_choices(choices: list) -> str:
            payload = [
                c.model_dump(mode="json", exclude_none=True)
                for c in sorted(choices, key=lambda x: x.label)
            ]
            return yaml.safe_dump(payload, sort_keys=False)

        return AIService.QuestionEditorState(
            title=question.title,
            question_type=question.question_type.value,
            points=question.points,
            mc_options_guidance=question.mc_options_guidance,
            figure_ids=[fig.id for fig in question.figures],
            figure_summaries=AIService._figure_summaries(question),
            question_template_md=question.solution.question_template_md,
            solution_parameters_yaml=_dump_parameters(question.solution.parameters),
            answer_formula_md=question.solution.answer_formula_md,
            answer_guidance=question.solution.answer_guidance,
            distractor_functions_text=_dump_distractors(
                question.solution.distractor_python_code
            ),
            mc_answer_specs_json=_dump_mc_answer_specs(
                question.solution.mc_answer_specs
            ),
            choices_yaml=_dump_choices(question.choices),
            typed_solution_md=question.solution.typed_solution_md,
            typed_solution_status=question.solution.typed_solution_status,
            last_computed_answer_md=question.solution.last_computed_answer_md,
        )

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
    def _truncate_chat_history(turns: list[ChatTurn], keep: int = 5) -> list[ChatTurn]:
        if keep <= 0:
            return [turn.model_copy(deep=True) for turn in turns]
        return [turn.model_copy(deep=True) for turn in turns[-keep:]]

    @staticmethod
    def _response_tool_calls(response: Any) -> list[Any]:
        output = getattr(response, "output", None) or []
        return [
            item
            for item in output
            if getattr(item, "type", None) == "function_call"
            and getattr(item, "name", None)
        ]

    @classmethod
    def _combine_usage(cls, totals: list[AIUsageTotals]) -> AIUsageTotals:
        usage = AIUsageTotals()
        for item in totals:
            usage.input_tokens += int(item.input_tokens or 0)
            usage.output_tokens += int(item.output_tokens or 0)
            usage.total_tokens += int(item.total_tokens or 0)
            usage.total_cost_usd += float(item.total_cost_usd or 0.0)
        return usage

    @staticmethod
    def _dump_choices_yaml(choices: list[Any]) -> str:
        payload = [
            c.model_dump(mode="json", exclude_none=True)
            for c in sorted(choices, key=lambda x: x.label)
        ]
        return yaml.safe_dump(payload, sort_keys=False)

    @classmethod
    def _sync_mc_choices(
        cls, question: Question, changed_fields: set[str], warnings: list[str]
    ) -> None:
        if question.question_type != QuestionType.multiple_choice:
            return
        if not question.solution.answer_formula_md.strip():
            return
        try:
            if question.solution.mc_answer_specs:
                harness = run_mc_formula_harness(
                    answer_formula_md=question.solution.answer_formula_md,
                    mc_answer_specs=question.solution.mc_answer_specs,
                    params=question.solution.parameters,
                    strict=False,
                )
            else:
                funcs = question.solution.distractor_python_code
                if len(funcs) != 4 or any(not d.python_code.strip() for d in funcs):
                    return
                harness = run_mc_harness(
                    answer_formula_md=question.solution.answer_formula_md,
                    distractor_python_codes=[(d.id, d.python_code) for d in funcs],
                    params=question.solution.parameters,
                )
        except SolutionRuntimeError as ex:
            warnings.append(str(ex))
            return
        except Exception as ex:
            warnings.append(str(ex))
            return
        question.choices = harness.choices
        changed_fields.add("choices_yaml")
        if harness.collisions:
            warnings.extend(harness.collisions)

    @staticmethod
    def _tool_schema() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "finish",
                "description": (
                    "Finish the editing task and return the final assistant message. "
                    "Call this exactly once when you are done."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "assistant_message": {"type": "string"},
                        "warnings": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["assistant_message"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_exam_item",
                "description": "Get the current editable exam item state.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "function",
                "name": "set_title",
                "description": "Set the short question title.",
                "parameters": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "set_question_type",
                "description": "Set the question type.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question_type": {
                            "type": "string",
                            "enum": ["free_response", "multiple_choice"],
                        }
                    },
                    "required": ["question_type"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "set_points",
                "description": "Set the point value for the item.",
                "parameters": {
                    "type": "object",
                    "properties": {"points": {"type": "integer"}},
                    "required": ["points"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "set_question_text",
                "description": "Replace the question markdown/template text.",
                "parameters": {
                    "type": "object",
                    "properties": {"question_template_md": {"type": "string"}},
                    "required": ["question_template_md"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "set_parameters",
                "description": "Replace the YAML/JSON parameters mapping.",
                "parameters": {
                    "type": "object",
                    "properties": {"parameters": {}},
                    "required": ["parameters"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "set_answer_guidance",
                "description": "Set private guidance for the deterministic solution code.",
                "parameters": {
                    "type": "object",
                    "properties": {"answer_guidance": {"type": "string"}},
                    "required": ["answer_guidance"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "set_answer_formula",
                "description": "Replace the answer formula text and validate it.",
                "parameters": {
                    "type": "object",
                    "properties": {"answer_formula_md": {"type": "string"}},
                    "required": ["answer_formula_md"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "compute_answer",
                "description": "Run the current deterministic answer formula and store the rendered answer.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "function",
                "name": "set_distractor",
                "description": "Set one multiple-choice distractor by 1-based index using formula and rationale text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "formula_md": {"type": "string"},
                        "rationale_md": {"type": "string"},
                    },
                    "required": ["index", "formula_md"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "delete_distractor",
                "description": "Delete one distractor function by 1-based index.",
                "parameters": {
                    "type": "object",
                    "properties": {"index": {"type": "integer"}},
                    "required": ["index"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "set_mc_options_guidance",
                "description": "Set the multiple-choice author guidance text.",
                "parameters": {
                    "type": "object",
                    "properties": {"mc_options_guidance": {"type": "string"}},
                    "required": ["mc_options_guidance"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "validate_exam_item",
                "description": "Run validation against the current exam item and return any problems.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def _execute_chat_tool(
        self,
        question: Question,
        tool_name: str,
        raw_arguments: str,
        changed_fields: set[str],
        warnings: list[str],
    ) -> dict[str, Any]:
        args = self._parse_json_object(raw_arguments or "{}")
        if tool_name == "get_exam_item":
            state = self._editor_state_for_question(question).model_dump(mode="json")
            state["figure_ids"] = [fig.id for fig in question.figures]
            return {"ok": True, "state": state}
        if tool_name == "set_title":
            question.title = str(args.get("title", ""))
            changed_fields.add("title")
            return {"ok": True}
        if tool_name == "set_question_type":
            question.question_type = QuestionType(str(args.get("question_type")))
            changed_fields.add("question_type")
            self._sync_mc_choices(question, changed_fields, warnings)
            return {"ok": True, "question_type": question.question_type.value}
        if tool_name == "set_points":
            question.points = int(args.get("points", question.points))
            changed_fields.add("points")
            return {"ok": True, "points": question.points}
        if tool_name == "set_question_text":
            question.solution.question_template_md = str(
                args.get("question_template_md", "")
            )
            changed_fields.add("question_template_md")
            return {"ok": True}
        if tool_name == "set_parameters":
            params = self._coerce_parameters_object(args.get("parameters"))
            question.solution.parameters = params
            changed_fields.add("solution_parameters_yaml")
            self._sync_mc_choices(question, changed_fields, warnings)
            return {"ok": True, "parameters": params}
        if tool_name == "set_answer_guidance":
            question.solution.answer_guidance = str(args.get("answer_guidance", ""))
            changed_fields.add("answer_guidance")
            return {"ok": True}
        if tool_name == "set_answer_formula":
            formula = str(args.get("answer_formula_md", "")).strip()
            run_answer_formula(
                formula,
                question.solution.parameters,
                answer_text_md=question.solution.answer_guidance,
                strict=True,
            )
            question.solution.answer_formula_md = formula
            changed_fields.add("answer_formula_md")
            self._sync_mc_choices(question, changed_fields, warnings)
            return {"ok": True}
        if tool_name == "compute_answer":
            result = run_answer_formula(
                question.solution.answer_formula_md,
                question.solution.parameters,
                answer_text_md=question.solution.answer_guidance,
                strict=True,
            )
            question.solution.last_computed_answer_md = result.answer_md
            changed_fields.add("last_computed_answer_md")
            return {
                "ok": True,
                "calculated_variables_md": result.calculated_variables_md,
                "answer_md": result.answer_md,
                "final_answer": result.final_answer,
            }
        if tool_name == "set_distractor":
            index = int(args.get("index", 0))
            if index < 1:
                raise ValueError("Distractor index must be 1 or greater.")
            specs = list(question.solution.mc_answer_specs)
            while len(specs) < index:
                specs.append(MCAnswerSpec())
            specs[index - 1] = MCAnswerSpec(
                formula_md=str(args.get("formula_md", "")).strip(),
                rationale_md=str(args.get("rationale_md", "")).strip(),
            )
            question.solution.mc_answer_specs = specs
            changed_fields.add("mc_answer_specs_json")
            self._sync_mc_choices(question, changed_fields, warnings)
            return {"ok": True, "index": index}
        if tool_name == "delete_distractor":
            index = int(args.get("index", 0))
            if index < 1:
                raise ValueError("Distractor index must be 1 or greater.")
            specs = list(question.solution.mc_answer_specs)
            if index > len(specs):
                return {"ok": True, "deleted": False}
            del specs[index - 1]
            question.solution.mc_answer_specs = specs
            question.choices = []
            changed_fields.update({"mc_answer_specs_json", "choices_yaml"})
            return {"ok": True, "deleted": True}
        if tool_name == "set_mc_options_guidance":
            question.mc_options_guidance = str(args.get("mc_options_guidance", ""))
            changed_fields.add("mc_options_guidance")
            return {"ok": True}
        if tool_name == "validate_exam_item":
            errors = validate_question(question)
            return {"ok": not bool(errors), "errors": errors}
        raise ValueError(f"Unsupported tool call: {tool_name}")

    def chat_edit_question(
        self,
        question: Question,
        user_message: str,
        attached_figure_ids: list[str] | None = None,
        history_keep_count: int = 5,
    ) -> QuestionEditorResult:
        client = self._client()
        working = question.model_copy(deep=True)
        changed_fields: set[str] = set()
        warnings: list[str] = []
        recent_history = self._truncate_chat_history(
            working.solution.chat_history, keep=history_keep_count
        )
        state = self._editor_state_for_question(working)
        state_json = json.dumps(
            state.model_dump(mode="json"), ensure_ascii=False, indent=2
        )
        history_lines = []
        for turn in recent_history:
            history_lines.append(f"User: {turn.user_message}")
            history_lines.append(f"Assistant: {turn.assistant_message}")
        history_text = "\n".join(history_lines) if history_lines else "(none)"
        attached_ids = [
            str(fid).strip() for fid in (attached_figure_ids or []) if str(fid).strip()
        ]
        system_prompt = (
            "You are editing a single exam item for a physics-authoring app. "
            "Use the provided tools to inspect and update the item. "
            "Do not invent fields or return raw patches. "
            "Use deterministic tools for answer and distractor changes. "
            "For rewrite or parameter-extraction requests, call only the minimal tools needed. "
            "Do not call compute_answer unless the author explicitly asks you to calculate, compute, or solve. "
            "When you are done, call the finish tool exactly once with a short factual assistant_message."
        )
        user_content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "Current editor state:\n"
                    f"{state_json}\n\n"
                    "Persisted recent chat history:\n"
                    f"{history_text}\n\n"
                    f"Current author request:\n{user_message.strip()}\n\n"
                    "Figure metadata and summaries are included below. "
                    "Only image bytes explicitly attached to this turn are included below."
                ),
            }
        ]
        figure_prompt_text = self._figure_prompt_text(working)
        if figure_prompt_text:
            user_content.append({"type": "input_text", "text": figure_prompt_text})
        if attached_ids:
            user_content.append(
                {
                    "type": "input_text",
                    "text": "Attached figure ids for this turn: "
                    + ", ".join(attached_ids),
                }
            )
            user_content.extend(self._figure_content(working, attached_ids))
        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {"role": "user", "content": user_content},
            ],
            tools=self._tool_schema(),
        )
        usage_parts = [self._usage_from_response(response)]
        loop_guard = 0
        while True:
            loop_guard += 1
            if loop_guard > 20:
                raise ValueError("AI chat exceeded the tool-call limit.")
            tool_calls = self._response_tool_calls(response)
            if not tool_calls:
                payload = self._parse_json_object(getattr(response, "output_text", ""))
                parsed = AIService.ChatAssistantResponse.model_validate(payload)
                final_warnings = warnings + parsed.warnings
                return AIService.QuestionEditorResult(
                    assistant_message=parsed.assistant_message,
                    question=working,
                    warnings=final_warnings,
                    usage=self._combine_usage(usage_parts),
                    changed_fields=sorted(changed_fields),
                )
            outputs: list[dict[str, Any]] = []
            for call in tool_calls:
                if str(getattr(call, "name", "")) == "finish":
                    payload = self._parse_json_object(
                        str(getattr(call, "arguments", "") or "{}")
                    )
                    parsed = AIService.ChatFinishArgs.model_validate(payload)
                    final_warnings = warnings + parsed.warnings
                    return AIService.QuestionEditorResult(
                        assistant_message=parsed.assistant_message,
                        question=working,
                        warnings=final_warnings,
                        usage=self._combine_usage(usage_parts),
                        changed_fields=sorted(changed_fields),
                    )
                try:
                    result = self._execute_chat_tool(
                        working,
                        str(getattr(call, "name", "")),
                        str(getattr(call, "arguments", "") or "{}"),
                        changed_fields,
                        warnings,
                    )
                except Exception as ex:
                    result = {"ok": False, "error": str(ex)}
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": getattr(call, "call_id", ""),
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )
            response = client.responses.create(
                model=self.model,
                previous_response_id=getattr(response, "id", None),
                input=outputs,
                tools=self._tool_schema(),
            )
            usage_parts.append(self._usage_from_response(response))

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

        for pattern in (
            r"```(?:json)?\s*(\{.*?\})\s*```",
            r"```(?:yaml|yml)\s*(.*?)\s*```",
        ):
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
            "AI response was not parseable as an object. " f"Sample: {sample}"
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
                in {
                    "figure_ref",
                    "fig_ref",
                    "figure_id",
                    "fig_id",
                    "image_ref",
                    "image_id",
                }
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
            filtered[key_clean] = coerce_numeric_scalar(value, strict=True)

        return filtered, rewritten_template

    def rewrite_parameterize(self, question: Question) -> RewriteResult:
        bundle = self.compose_prompt(action="rewrite_parameterize", question=question)
        result = self._text_with_question_context(bundle, question)
        payload = self._parse_json_object(result.text)
        template = str(payload.get("question_template_md", "")).strip()

        title = ""
        title_key = "title"
        for candidate_key in ("title", "question_title", "short_title", "name"):
            candidate_value = payload.get(candidate_key)
            if isinstance(candidate_value, str) and candidate_value.strip():
                title = candidate_value.strip()
                title_key = candidate_key
                break
        if not title and isinstance(payload.get("title"), str):
            title = str(payload.get("title", "")).strip()

        logger.debug(
            "rewrite_parameterize parsed payload keys=%s title_key=%s title_empty=%s",
            sorted(payload.keys()),
            title_key,
            (not bool(title)),
        )
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

    def generate_answer_formula(
        self,
        question: Question,
        error_feedback: str = "",
    ) -> AnswerFormulaResult:
        bundle = self.compose_prompt(
            action="generate_answer_formula", question=question
        )
        if error_feedback.strip():
            bundle = PromptBundle(
                system_prompt=bundle.system_prompt,
                user_prompt=f"{bundle.user_prompt}\n\nPrevious execution error:\n{error_feedback.strip()}",
            )
        result = self._text_with_question_context(bundle, question)
        payload = self._parse_json_object(result.text)
        answer_formula_md = str(payload.get("answer_formula_md", "")).strip()
        if not answer_formula_md:
            raise ValueError("AI response field 'answer_formula_md' is required.")
        return AIService.AnswerFormulaResult(
            answer_formula_md=answer_formula_md,
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
