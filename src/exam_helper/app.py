from __future__ import annotations

import json
import logging
import re
import base64
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from fastapi import Body, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from exam_helper.ai_service import AIService
from exam_helper.export_docx import (
    _PANDOC_MISSING_WARNING,
    render_question_docx_bytes,
    render_project_docx_bytes,
)
from exam_helper.models import (
    AIUsageTotals,
    ChatTurn,
    DEFAULT_OPENAI_MODEL,
    DistractorFunction,
    MCAnswerSpec,
    MCChoice,
    ProjectConfig,
    Question,
    QuestionType,
)
from exam_helper.parameter_utils import coerce_numeric_scalar
from exam_helper.normalization import normalize_markdown_math_delimiters
from exam_helper.repository import ProjectRepository
from exam_helper.solution_runtime import (
    evaluate_answer_formula,
    render_template_from_values,
    run_answer_formula,
    run_mc_formula_harness,
    run_mc_harness,
)
from exam_helper.validation import validate_question


class QuestionEditorState(BaseModel):
    title: str = ""
    question_type: str = "multiple_choice"
    mc_options_guidance: str = ""
    question_template_md: str = ""
    solution_parameters_yaml: str = "{}"
    answer_formula_md: str = ""
    answer_guidance: str = ""
    distractor_functions_text: str = ""
    mc_answer_specs_json: str = "[]"
    choices_yaml: str = "[]"
    typed_solution_md: str = ""
    typed_solution_status: str = "missing"
    figures_json: str = "[]"
    chat_history_json: str = "[]"
    points: int = 5


AutosavePayload = QuestionEditorState


class ChatPayload(BaseModel):
    message: str = ""
    attached_figure_ids: list[str] = Field(default_factory=list)
    editor_state: QuestionEditorState
    history_keep_count: int = Field(default=5, ge=0)


def _sanitize_docx_filename_stem(project_name: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", project_name.lower()).strip("-")
    stem = re.sub(r"-{2,}", "-", stem)
    return stem or "exam"


def create_app(project_root: Path, openai_key: str | None) -> FastAPI:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    app = FastAPI(title="Exam Helper")
    repo = ProjectRepository(project_root)
    project = (
        repo.load_project()
        if repo.project_file.exists()
        else ProjectConfig(name="(uninitialized)", course="")
    )

    def make_ai_service(config: ProjectConfig) -> AIService:
        return AIService(
            api_key=openai_key,
            model=config.ai.model,
            prompts_override=config.ai.prompts,
        )

    def refresh_ai_service() -> None:
        latest = repo.load_project()
        app.state.ai = make_ai_service(latest)

    app.state.project_root = project_root
    app.state.openai_key = openai_key
    app.state.repo = repo
    app.state.ai = make_ai_service(project)
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    def parse_parameters_yaml(raw_yaml: str) -> dict[str, Any]:
        data = yaml.safe_load(raw_yaml or "{}")
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError("Solution parameters YAML must be a mapping.")
        return data

    def dump_parameters_yaml(params: dict[str, Any]) -> str:
        return yaml.safe_dump(params or {}, sort_keys=False).strip()

    def validate_numeric_parameters(params: dict[str, Any]) -> dict[str, Any]:
        validated: dict[str, Any] = {}
        for key, value in (params or {}).items():
            key_clean = str(key).strip()
            if not key_clean:
                continue
            validated[key_clean] = coerce_numeric_scalar(value, strict=True)
        return validated

    def parse_distractor_functions_text(raw_text: str) -> list[DistractorFunction]:
        text = (raw_text or "").strip()
        if not text:
            return []
        # Backward-compatible: accept YAML list payload as well.
        if text.startswith("- ") or text.startswith("["):
            raw = yaml.safe_load(text) or []
            if not isinstance(raw, list):
                raise ValueError(
                    "Distractor functions must be YAML list or plain text blocks."
                )
            return [DistractorFunction.model_validate(item) for item in raw]

        blocks = [b.strip() for b in re.split(r"\n---\n", text) if b.strip()]
        out: list[DistractorFunction] = []
        for idx, block in enumerate(blocks, start=1):
            lines = block.splitlines()
            header = lines[0].strip() if lines else ""
            m = re.match(r"^#\s*distractor\s*:\s*([A-Za-z0-9_-]+)\s*$", header)
            if m:
                did = m.group(1)
                code = "\n".join(lines[1:]).strip()
            else:
                did = f"d{idx}"
                code = block
            out.append(DistractorFunction(id=did, python_code=code))
        return out

    def dump_distractor_functions_text(funcs: list[DistractorFunction]) -> str:
        if not funcs:
            return ""
        chunks: list[str] = []
        for f in funcs:
            chunks.append(f"# distractor: {f.id}\n{(f.python_code or '').strip()}")
        return "\n---\n".join(chunks).strip() + "\n"

    def parse_mc_answer_specs_json(raw_json: str) -> list[MCAnswerSpec]:
        raw = json.loads(raw_json or "[]")
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise ValueError("MC answer specs must parse to a list.")
        return [MCAnswerSpec.model_validate(item) for item in raw]

    def dump_mc_answer_specs_json(specs: list[MCAnswerSpec]) -> str:
        return json.dumps(
            [spec.model_dump(mode="json") for spec in specs], ensure_ascii=False
        )

    def parse_choices_yaml(choices_yaml: str) -> list[MCChoice]:
        def _strip_disallowed_bold(text: str) -> str:
            out = text or ""
            out = re.sub(r"\*\*(.*?)\*\*", r"\1", out, flags=re.DOTALL)
            out = re.sub(r"__(.*?)__", r"\1", out, flags=re.DOTALL)
            out = re.sub(r"</?strong>", "", out, flags=re.IGNORECASE)
            out = re.sub(r"</?b>", "", out, flags=re.IGNORECASE)
            return out

        def _looks_explanatory(text: str) -> bool:
            t = (text or "").strip().lower()
            if not t:
                return False
            cue_words = (
                "mistakenly",
                "incorrect",
                "wrong",
                "misread",
                "misreads",
                "uses",
                "treats",
                "forgets",
                "because",
                "applies",
                "directly",
            )
            return any(w in t for w in cue_words) or len(t) > 90

        def _looks_answer_like(text: str) -> bool:
            t = (text or "").strip()
            if not t:
                return False
            if re.search(r"\d", t):
                return True
            if re.search(r"[=+\-/*^]", t):
                return True
            return len(t) <= 40 and not _looks_explanatory(t)

        raw = yaml.safe_load(choices_yaml or "[]") or []
        if not isinstance(raw, list):
            raise ValueError("choices_yaml YAML must parse to a list.")
        choices = []
        for c in raw:
            item = MCChoice.model_validate(c)
            item.content_md = _strip_disallowed_bold(item.content_md)
            if item.rationale is not None:
                item.rationale = _strip_disallowed_bold(item.rationale)
            if (
                item.rationale is not None
                and _looks_explanatory(item.content_md)
                and _looks_answer_like(item.rationale)
            ):
                item.content_md, item.rationale = item.rationale, item.content_md
            choices.append(item)
        return sorted(choices, key=lambda c: c.label)

    def dump_choices_yaml(choices: list[MCChoice]) -> str:
        payload = [
            c.model_dump(mode="json", exclude_none=True)
            for c in sorted(choices, key=lambda x: x.label)
        ]
        return yaml.safe_dump(payload, sort_keys=False)

    def dedupe_choices(choices: list[MCChoice]) -> list[MCChoice]:
        seen: set[str] = set()
        unique: list[MCChoice] = []
        for c in choices:
            canonical = re.sub(r"\s+", " ", (c.content_md or "").strip()).casefold()
            if canonical in seen:
                continue
            seen.add(canonical)
            unique.append(c)
        labels = ["A", "B", "C", "D", "E"]
        out: list[MCChoice] = []
        for idx, c in enumerate(unique):
            out.append(
                MCChoice(
                    label=labels[idx] if idx < len(labels) else c.label,
                    content_md=c.content_md,
                    is_correct=c.is_correct,
                    rationale=c.rationale,
                )
            )
        return out

    def default_mc_choices_yaml() -> str:
        defaults = [
            {"label": "A", "content_md": "", "is_correct": True, "rationale": ""},
            {"label": "B", "content_md": "", "is_correct": False, "rationale": ""},
            {"label": "C", "content_md": "", "is_correct": False, "rationale": ""},
            {"label": "D", "content_md": "", "is_correct": False, "rationale": ""},
            {"label": "E", "content_md": "", "is_correct": False, "rationale": ""},
        ]
        return yaml.safe_dump(defaults, sort_keys=False)

    def default_mc_answer_specs() -> list[MCAnswerSpec]:
        return [MCAnswerSpec() for _ in range(4)]

    def _suggest_next_question_id() -> str:
        try:
            existing = [q.id for q in repo.list_questions(include_deleted=True)]
        except Exception:
            existing = []
        used = set(existing)
        for i in range(1, 10000):
            candidate = f"q{i}"
            if candidate not in used:
                return candidate
        return "q_new"

    def _compute_mc_preview(
        answer_formula_md: str,
        mc_answer_specs: list[MCAnswerSpec],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        if not answer_formula_md.strip():
            return {
                "correct_answer_md": "",
                "choices": [],
                "rows": [
                    {
                        "label": f"Distractor {idx}",
                        "formula_md": spec.formula_md,
                        "rationale_md": spec.rationale_md,
                        "preview_md": "",
                        "warning": "",
                    }
                    for idx, spec in enumerate(mc_answer_specs, start=1)
                ],
                "preview_answers": "",
                "preview_rationale": "",
                "warning": "",
                "preview_choices": [],
            }
        try:
            result = run_mc_formula_harness(
                answer_formula_md=answer_formula_md,
                mc_answer_specs=mc_answer_specs,
                params=params,
                strict=False,
            )
            rows_by_source = {
                str(row.get("source_id", "")): row for row in result.row_previews
            }
            rows: list[dict[str, Any]] = []
            preview_answers_lines: list[str] = []
            preview_rationale_lines: list[str] = []
            preview_choices = [
                choice.model_dump(mode="json") for choice in result.choices
            ]
            for choice in result.choices:
                label = str(choice.label or "?")
                content = normalize_markdown_math_delimiters(choice.content_md or "")
                preview_answers_lines.append(f"{label}. {content}")
                rationale = normalize_markdown_math_delimiters(choice.rationale or "")
                if rationale.strip():
                    preview_rationale_lines.append(
                        f"{label}. {content} - {rationale}".strip()
                    )
                else:
                    preview_rationale_lines.append(f"{label}. {content}")
            for idx, spec in enumerate(mc_answer_specs, start=1):
                preview = rows_by_source.get(f"choice_{idx}", {})
                content_md = normalize_markdown_math_delimiters(
                    str(preview.get("content_md", ""))
                )
                rows.append(
                    {
                        "label": f"Distractor {idx}",
                        "formula_md": spec.formula_md,
                        "rationale_md": spec.rationale_md,
                        "preview_md": content_md,
                        "warning": str(preview.get("warning", "")),
                    }
                )
            warning = " | ".join(
                [item for item in [*result.warnings, *result.collisions] if item]
            ).strip()
            return {
                "correct_answer_md": normalize_markdown_math_delimiters(
                    result.correct_answer_md
                ),
                "choices": result.choices,
                "rows": rows,
                "preview_answers": normalize_markdown_math_delimiters(
                    "\n".join(preview_answers_lines).strip()
                ),
                "preview_rationale": normalize_markdown_math_delimiters(
                    "\n".join(preview_rationale_lines).strip()
                ),
                "warning": warning,
                "preview_choices": preview_choices,
            }
        except Exception as ex:
            return {
                "correct_answer_md": "",
                "choices": [],
                "rows": [
                    {
                        "label": f"Distractor {idx}",
                        "formula_md": spec.formula_md,
                        "rationale_md": spec.rationale_md,
                        "preview_md": "",
                        "warning": str(ex),
                    }
                    for idx, spec in enumerate(mc_answer_specs, start=1)
                ],
                "preview_answers": "",
                "preview_rationale": "",
                "warning": str(ex),
                "preview_choices": [],
            }

    def _question_form_context(question: Question | None) -> dict[str, Any]:
        mc_answer_specs = (
            question.solution.mc_answer_specs
            if question and question.solution.mc_answer_specs
            else default_mc_answer_specs()
        )
        mc_preview = _compute_mc_preview(
            question.solution.answer_formula_md if question else "",
            mc_answer_specs,
            question.solution.parameters if question else {},
        )
        figures_json = json.dumps(
            [f.model_dump(mode="json") for f in question.figures] if question else []
        )
        solution_parameters_yaml = dump_parameters_yaml(
            question.solution.parameters if question else {}
        )
        distractor_functions_text = dump_distractor_functions_text(
            question.solution.distractor_python_code if question else []
        )
        answer_preview = _compute_answer_preview(question)
        chat_history_json = json.dumps(
            [
                turn.model_dump(mode="json")
                for turn in (question.solution.chat_history if question else [])
            ]
        )
        editor_state = QuestionEditorState(
            title=question.title if question else "",
            points=question.points if question else 5,
            question_type=(
                question.question_type.value
                if question
                else QuestionType.multiple_choice.value
            ),
            mc_options_guidance=question.mc_options_guidance if question else "",
            question_template_md=(
                question.solution.question_template_md
                if question
                else "The length of the rod is {{dummy}} $m$."
            ),
            solution_parameters_yaml=(
                solution_parameters_yaml if question else "dummy: 5"
            ),
            answer_formula_md=question.solution.answer_formula_md if question else "",
            answer_guidance=question.solution.answer_guidance if question else "",
            distractor_functions_text=distractor_functions_text,
            mc_answer_specs_json=dump_mc_answer_specs_json(mc_answer_specs),
            choices_yaml=(
                dump_choices_yaml(question.choices)
                if question and question.choices
                else default_mc_choices_yaml()
            ),
            typed_solution_md=(question.solution.typed_solution_md if question else ""),
            typed_solution_status=(
                question.solution.typed_solution_status if question else "missing"
            ),
            figures_json=figures_json,
            chat_history_json=chat_history_json,
        )
        return {
            "question": question,
            "figures_json": figures_json,
            "solution_parameters_yaml": solution_parameters_yaml,
            "distractor_functions_text": distractor_functions_text,
            "mc_answer_specs_json": dump_mc_answer_specs_json(mc_answer_specs),
            "choices_yaml": editor_state.choices_yaml,
            "mc_answer_rows": mc_preview["rows"],
            "mc_preview_choices": mc_preview["preview_choices"],
            "mc_preview_answers": mc_preview["preview_answers"],
            "mc_preview_rationale": mc_preview["preview_rationale"],
            "mc_preview_warning": mc_preview["warning"],
            "answer_formula_md": (
                question.solution.answer_formula_md if question else ""
            ),
            "answer_guidance": question.solution.answer_guidance if question else "",
            "calculated_variables_md": answer_preview["calculated_variables_md"],
            "answer_formula_warning": answer_preview["warning"],
            "rendered_answer_md": (
                question.solution.last_computed_answer_md
                if question and question.solution.last_computed_answer_md.strip()
                else answer_preview["rendered_answer_md"]
            ),
            "chat_history_json": chat_history_json,
            "chat_history_keep_count_default": 5,
            "ai_enabled": bool(openai_key),
            "question_id_default": (
                question.id if question else _suggest_next_question_id()
            ),
            "editor_state": editor_state.model_dump(mode="json"),
        }

    def _parse_chat_history_json(raw: str) -> list[ChatTurn]:
        loaded = json.loads(raw or "[]")
        if not isinstance(loaded, list):
            raise ValueError("chat_history_json must be a JSON list.")
        return [ChatTurn.model_validate(item) for item in loaded]

    def _build_question_from_editor_values(
        question_id: str,
        existing: Question | None,
        values: dict[str, Any],
    ) -> tuple[Question, dict[str, Any], dict[str, Any]]:
        figures = json.loads(str(values.get("figures_json", "[]")) or "[]")
        solution_parameters = parse_parameters_yaml(
            str(values.get("solution_parameters_yaml", "{}"))
        )
        distractor_funcs = parse_distractor_functions_text(
            str(values.get("distractor_functions_text", ""))
        )
        mc_answer_specs = parse_mc_answer_specs_json(
            str(values.get("mc_answer_specs_json", "[]"))
        )
        chat_history = _parse_chat_history_json(
            str(
                values.get(
                    "chat_history_json",
                    json.dumps(
                        [
                            turn.model_dump(mode="json")
                            for turn in (
                                existing.solution.chat_history if existing else []
                            )
                        ]
                    ),
                )
            )
        )
        question_type_value = str(
            values.get(
                "question_type",
                existing.question_type.value if existing else "multiple_choice",
            )
        )
        mc_preview = _compute_mc_preview(
            str(values.get("answer_formula_md", "")),
            mc_answer_specs,
            solution_parameters,
        )
        if (
            question_type_value == QuestionType.multiple_choice.value
            and mc_answer_specs
        ):
            choices = mc_preview["choices"]
        else:
            choices = parse_choices_yaml(str(values.get("choices_yaml", "[]")))
        question = Question.model_validate(
            {
                "id": question_id,
                "title": str(values.get("title", "")),
                "points": int(values.get("points", 5) or 5),
                "is_deleted": (existing.is_deleted if existing else False),
                "question_type": QuestionType(question_type_value),
                "mc_options_guidance": str(values.get("mc_options_guidance", "")),
                "choices": choices,
                "solution": {
                    "question_template_md": normalize_markdown_math_delimiters(
                        str(values.get("question_template_md", ""))
                    ),
                    "parameters": solution_parameters,
                    "answer_formula_md": str(values.get("answer_formula_md", "")),
                    "answer_guidance": str(values.get("answer_guidance", "")),
                    "mc_answer_specs": mc_answer_specs,
                    "distractor_python_code": distractor_funcs,
                    "typed_solution_md": normalize_markdown_math_delimiters(
                        str(
                            values.get(
                                "typed_solution_md",
                                existing.solution.typed_solution_md if existing else "",
                            )
                        )
                    ),
                    "typed_solution_status": str(
                        values.get("typed_solution_status", "missing")
                    ),
                    "last_computed_answer_md": (
                        normalize_markdown_math_delimiters(
                            existing.solution.last_computed_answer_md
                        )
                        if existing
                        else ""
                    ),
                    "chat_history": chat_history,
                },
                "figures": figures,
            }
        )
        preview = _compute_answer_preview(question)
        if not preview["fatal_error"]:
            question.solution.last_computed_answer_md = (
                preview["rendered_answer_md"]
                or question.solution.last_computed_answer_md
            )
        _mark_typed_solution_stale_if_needed(existing, question)
        mc_preview = _compute_mc_preview(
            question.solution.answer_formula_md,
            question.solution.mc_answer_specs,
            question.solution.parameters,
        )
        return question, preview, mc_preview

    def _editor_values_from_question(question: Question | None) -> dict[str, Any]:
        return {
            "title": question.title if question else "",
            "points": question.points if question else 5,
            "question_type": (
                question.question_type.value
                if question
                else QuestionType.multiple_choice.value
            ),
            "mc_options_guidance": question.mc_options_guidance if question else "",
            "question_template_md": (
                question.solution.question_template_md if question else ""
            ),
            "solution_parameters_yaml": dump_parameters_yaml(
                question.solution.parameters if question else {}
            ),
            "answer_formula_md": (
                question.solution.answer_formula_md if question else ""
            ),
            "answer_guidance": question.solution.answer_guidance if question else "",
            "distractor_functions_text": dump_distractor_functions_text(
                question.solution.distractor_python_code if question else []
            ),
            "mc_answer_specs_json": dump_mc_answer_specs_json(
                question.solution.mc_answer_specs
                if question
                else default_mc_answer_specs()
            ),
            "choices_yaml": (
                dump_choices_yaml(question.choices)
                if question and question.choices
                else default_mc_choices_yaml()
            ),
            "typed_solution_md": (
                question.solution.typed_solution_md if question else ""
            ),
            "typed_solution_status": (
                question.solution.typed_solution_status if question else "missing"
            ),
            "figures_json": json.dumps(
                [f.model_dump(mode="json") for f in question.figures]
                if question
                else []
            ),
            "chat_history_json": json.dumps(
                [
                    turn.model_dump(mode="json")
                    for turn in (question.solution.chat_history if question else [])
                ]
            ),
        }

    def _state_from_question(question: Question | None) -> QuestionEditorState:
        return QuestionEditorState.model_validate(
            _editor_values_from_question(question)
        )

    def _build_question_from_editor_state(
        question_id: str,
        existing: Question | None,
        state: QuestionEditorState,
    ) -> tuple[Question, dict[str, Any], dict[str, Any]]:
        return _build_question_from_editor_values(
            question_id, existing, state.model_dump(mode="json")
        )

    def _editor_response_payload(
        question: Question,
        preview: dict[str, Any],
        mc_preview: dict[str, Any],
        *,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        editor_state = _state_from_question(question)
        payload: dict[str, Any] = {
            "ok": True,
            "editor_state": editor_state.model_dump(mode="json"),
            "typed_solution_status": question.solution.typed_solution_status,
            "calculated_variables_md": preview["calculated_variables_md"],
            "rendered_answer_md": (
                question.solution.last_computed_answer_md
                or preview["rendered_answer_md"]
            ),
            "mc_preview_choices": mc_preview["preview_choices"],
            "mc_preview_answers": mc_preview["preview_answers"],
            "mc_preview_rationale": mc_preview["preview_rationale"],
            "mc_preview_warning": mc_preview["warning"],
            "mc_preview_rows": mc_preview["rows"],
            "warning": preview["warning"],
            "answer_formula_warning": preview["warning"],
        }
        payload.update(editor_state.model_dump(mode="json"))
        if extra:
            payload.update(extra)
        return payload

    def _append_chat_turn(
        question: Question,
        *,
        user_message: str,
        assistant_message: str,
        attached_figure_ids: list[str],
    ) -> None:
        question.solution.chat_history = question.solution.chat_history + [
            ChatTurn(
                user_message=user_message.strip(),
                assistant_message=assistant_message.strip(),
                attached_figure_ids=attached_figure_ids[:],
            )
        ]

    def _compute_answer_preview(question: Question | None) -> dict[str, str]:
        """Build the non-editable answer preview payload for the editor UI."""
        if question is None:
            return {
                "calculated_variables_md": "",
                "rendered_answer_md": "",
                "warning": "",
                "fatal_error": "",
            }
        try:
            locals_ns, rendered_lines, warnings, fatal_error = evaluate_answer_formula(
                question.solution.answer_formula_md,
                question.solution.parameters,
            )
            return {
                "calculated_variables_md": normalize_markdown_math_delimiters(
                    "\n".join(rendered_lines).strip()
                ),
                "rendered_answer_md": normalize_markdown_math_delimiters(
                    render_template_from_values(
                        question.solution.answer_guidance, locals_ns
                    ).strip()
                ),
                "warning": " | ".join(warnings),
                "fatal_error": fatal_error or "",
            }
        except Exception as ex:
            return {
                "calculated_variables_md": "",
                "rendered_answer_md": "",
                "warning": str(ex),
                "fatal_error": str(ex),
            }

    def _mark_typed_solution_stale_if_needed(
        existing: Question | None, candidate: Question
    ) -> None:
        if existing is None:
            return

        def _normalized_code(text: str) -> str:
            return (text or "").replace("\r\n", "\n").replace("\r", "\n").rstrip()

        if (
            existing.solution.parameters != candidate.solution.parameters
            or _normalized_code(existing.solution.answer_formula_md)
            != _normalized_code(candidate.solution.answer_formula_md)
            or _normalized_code(existing.solution.answer_guidance)
            != _normalized_code(candidate.solution.answer_guidance)
            or [
                _normalized_code(spec.formula_md)
                for spec in existing.solution.mc_answer_specs
            ]
            != [
                _normalized_code(spec.formula_md)
                for spec in candidate.solution.mc_answer_specs
            ]
            or [
                _normalized_code(spec.rationale_md)
                for spec in existing.solution.mc_answer_specs
            ]
            != [
                _normalized_code(spec.rationale_md)
                for spec in candidate.solution.mc_answer_specs
            ]
            or [
                _normalized_code(d.python_code)
                for d in existing.solution.distractor_python_code
            ]
            != [
                _normalized_code(d.python_code)
                for d in candidate.solution.distractor_python_code
            ]
        ):
            if candidate.solution.typed_solution_md.strip():
                candidate.solution.typed_solution_status = "stale"
            else:
                candidate.solution.typed_solution_status = "missing"

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        questions = repo.list_questions() if repo.project_file.exists() else []
        project = repo.load_project() if repo.project_file.exists() else None
        export_warning = request.cookies.get("exam_helper_export_warning")
        pandoc_warning = (
            _PANDOC_MISSING_WARNING if shutil.which("pandoc") is None else None
        )
        response = templates.TemplateResponse(
            request,
            "index.html",
            {
                "project_root": str(project_root),
                "project": project,
                "questions": questions,
                "export_warning": export_warning,
                "pandoc_warning": pandoc_warning,
                "ai_model": (project.ai.model if project else DEFAULT_OPENAI_MODEL),
                "ai_usage": (project.ai.usage if project else AIUsageTotals()),
                "ai_prompts": (project.ai.prompts if project else None),
            },
        )
        if export_warning:
            response.delete_cookie("exam_helper_export_warning")
        return response

    @app.get("/questions/new", response_class=HTMLResponse)
    def new_question(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "question_form_v2.html",
            _question_form_context(None),
        )

    @app.get("/questions/{question_id}/edit", response_class=HTMLResponse)
    def edit_question(request: Request, question_id: str) -> HTMLResponse:
        q = repo.get_question(question_id)
        return templates.TemplateResponse(
            request,
            "question_form_v2.html",
            _question_form_context(q),
        )

    @app.post("/questions/{question_id}/delete")
    def delete_question(question_id: str) -> RedirectResponse:
        question = repo.get_question(question_id)
        question.is_deleted = True
        repo.save_question(question)
        return RedirectResponse("/", status_code=303)

    @app.post("/questions/save")
    def save_question(
        question_id: str = Form(...),
        title: str = Form(""),
        points: int = Form(5),
        question_type: str = Form("multiple_choice"),
        mc_options_guidance: str = Form(""),
        question_template_md: str = Form(""),
        mc_answer_specs_json: str = Form("[]"),
        choices_yaml: str = Form("[]"),
        solution_parameters_yaml: str = Form("{}"),
        answer_formula_md: str = Form(""),
        answer_guidance: str = Form(""),
        distractor_functions_text: str = Form(""),
        typed_solution_md: str = Form(""),
        typed_solution_status: str = Form("missing"),
        figures_json: str = Form("[]"),
        chat_history_json: str = Form("[]"),
    ) -> RedirectResponse:
        existing = None
        try:
            existing = repo.get_question(question_id)
        except Exception:
            existing = None
        question, _, _ = _build_question_from_editor_values(
            question_id,
            existing,
            {
                "title": title,
                "points": points,
                "question_type": question_type,
                "mc_options_guidance": mc_options_guidance,
                "question_template_md": question_template_md,
                "solution_parameters_yaml": solution_parameters_yaml,
                "answer_formula_md": answer_formula_md,
                "answer_guidance": answer_guidance,
                "distractor_functions_text": distractor_functions_text,
                "mc_answer_specs_json": mc_answer_specs_json,
                "typed_solution_md": typed_solution_md,
                "typed_solution_status": typed_solution_status,
                "choices_yaml": choices_yaml,
                "figures_json": figures_json,
                "chat_history_json": chat_history_json,
            },
        )
        repo.save_question(question)
        return RedirectResponse("/", status_code=303)

    @app.post("/questions/{question_id}/autosave")
    def autosave_question(
        question_id: str, payload: AutosavePayload = Body(...)
    ) -> JSONResponse:
        try:
            existing = None
            try:
                existing = repo.get_question(question_id)
            except Exception:
                existing = None
            question, preview, mc_preview = _build_question_from_editor_state(
                question_id, existing, payload
            )
            repo.save_question(question)
            return JSONResponse(_editor_response_payload(question, preview, mc_preview))
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.get("/questions/{question_id}/{figure_id}")
    def serve_embedded_figure(question_id: str, figure_id: str) -> Response:
        q = repo.get_question(question_id)
        fig = next((f for f in q.figures if f.id == figure_id), None)
        if fig is None:
            return Response(status_code=404)
        try:
            raw = base64.b64decode((fig.data_base64 or "").encode("ascii"))
        except Exception:
            return Response(status_code=404)
        media_type = fig.mime_type or "application/octet-stream"
        return Response(content=raw, media_type=media_type)

    @app.post("/figures/validate")
    async def validate_figure(request: Request) -> dict:
        data_base64 = ""
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            payload = await request.json()
            if isinstance(payload, dict):
                data_base64 = str(payload.get("data_base64", ""))
        else:
            form = await request.form()
            data_base64 = str(form.get("data_base64", ""))
        if not data_base64:
            return JSONResponse(
                {"ok": False, "error": "Missing data_base64 payload."}, status_code=400
            )
        raw = base64.b64decode(data_base64.encode("ascii"))
        return {"sha256": sha256(raw).hexdigest(), "size": len(raw)}

    @app.post("/figures/summarize", response_model=None)
    def summarize_figure(
        data_base64: str = Form(...), mime_type: str = Form("image/png")
    ) -> Any:
        try:
            normalized_mime, normalized_data = AIService._prepare_figure_summary_input(
                mime_type, data_base64
            )
            result = app.state.ai.summarize_figure(
                mime_type=normalized_mime, data_base64=normalized_data
            )
            repo.add_ai_usage(result.usage)
            return {"summary": result.text}
        except Exception as ex:
            logger.exception(
                "figure.summarize failed mime_type=%s size=%s",
                mime_type,
                len(data_base64 or ""),
            )
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/validate")
    def validate_question_endpoint(question_id: str) -> dict:
        q = repo.get_question(question_id)
        errors = validate_question(q)
        return {"question_id": question_id, "errors": errors, "ok": not errors}

    @app.post("/questions/{question_id}/ai/chat")
    def ai_chat(question_id: str, payload: ChatPayload = Body(...)) -> dict:
        try:
            question_for_log = None
            existing = None
            try:
                existing = repo.get_question(question_id)
            except Exception:
                existing = None
            live_question, _, _ = _build_question_from_editor_state(
                question_id, existing, payload.editor_state
            )
            question_for_log = live_question
            result = app.state.ai.chat_edit_question(
                live_question,
                payload.message,
                attached_figure_ids=payload.attached_figure_ids,
                history_keep_count=payload.history_keep_count,
            )
            question_for_log = result.question
            repo.add_ai_usage(result.usage)
            question = result.question
            _append_chat_turn(
                question,
                user_message=payload.message,
                assistant_message=result.assistant_message,
                attached_figure_ids=payload.attached_figure_ids,
            )
            preview = _compute_answer_preview(question)
            if not preview["fatal_error"]:
                question.solution.last_computed_answer_md = preview[
                    "rendered_answer_md"
                ]
            mc_preview = _compute_mc_preview(
                question.solution.answer_formula_md,
                question.solution.mc_answer_specs,
                question.solution.parameters,
            )
            repo.save_question(question)
            return _editor_response_payload(
                question,
                preview,
                mc_preview,
                extra={
                    "assistant_message": result.assistant_message,
                    "warnings": result.warnings,
                    "changed_fields": result.changed_fields + ["chat_history_json"],
                },
            )
        except Exception as ex:
            logger.exception(
                "harness.run failed question_id=%s qtype=%s template_len=%s params_keys=%s figures=%s",
                question_id,
                (
                    question_for_log.question_type.value
                    if question_for_log is not None
                    else "unknown"
                ),
                (
                    len((question_for_log.solution.question_template_md or "").strip())
                    if question_for_log is not None
                    else 0
                ),
                (
                    sorted((question_for_log.solution.parameters or {}).keys())
                    if question_for_log is not None
                    else []
                ),
                (
                    [f.id for f in (question_for_log.figures or [])]
                    if question_for_log is not None
                    else []
                ),
            )
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/ai/rewrite-and-parameterize")
    def ai_rewrite_and_parameterize(question_id: str) -> dict:
        try:
            q = repo.get_question(question_id)
            result = app.state.ai.rewrite_parameterize(q)
            repo.add_ai_usage(result.usage)
            parameters = validate_numeric_parameters(result.parameters)
            normalized_template = normalize_markdown_math_delimiters(
                result.question_template_md
            )
            rendered_prompt = render_template_from_values(
                normalized_template, parameters
            )
            title = q.title.strip() or result.title.strip()
            return {
                "ok": True,
                "question_template_md": normalized_template,
                "rendered_prompt_md": normalize_markdown_math_delimiters(
                    rendered_prompt
                ),
                "solution_parameters_yaml": dump_parameters_yaml(parameters),
                "title": title,
            }
        except Exception as ex:
            logger.exception(
                "ai_rewrite_and_parameterize failed question_id=%s template_len=%s params_keys=%s figures=%s",
                question_id,
                (
                    len((q.solution.question_template_md or "").strip())
                    if "q" in locals()
                    else 0
                ),
                sorted((q.solution.parameters or {}).keys()) if "q" in locals() else [],
                [f.id for f in (q.figures or [])] if "q" in locals() else [],
            )
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/ai/generate-answer-formula")
    def ai_generate_answer_formula(question_id: str) -> dict:
        try:
            q = repo.get_question(question_id)
            error_feedback = ""
            for _ in range(3):
                result = app.state.ai.generate_answer_formula(
                    q, error_feedback=error_feedback
                )
                repo.add_ai_usage(result.usage)
                try:
                    run_answer_formula(
                        result.answer_formula_md, q.solution.parameters, strict=True
                    )
                    return {"ok": True, "answer_formula_md": result.answer_formula_md}
                except Exception as ex:
                    error_feedback = str(ex)
                    q.solution.answer_formula_md = result.answer_formula_md
            raise ValueError(
                "AI-generated answer formula failed validation after 3 attempts. "
                f"Last error: {error_feedback}"
            )
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/harness/run")
    def run_harness(question_id: str) -> dict:
        try:
            q = repo.get_question(question_id)
            answer_result = run_answer_formula(
                q.solution.answer_formula_md,
                q.solution.parameters,
                answer_text_md=q.solution.answer_guidance,
            )
            payload: dict[str, Any] = {
                "ok": True,
                "computed_answer_md": normalize_markdown_math_delimiters(
                    answer_result.answer_md
                ),
                "final_answer_text": answer_result.final_answer,
                "calculated_variables_md": normalize_markdown_math_delimiters(
                    answer_result.calculated_variables_md
                ),
            }
            if q.question_type == QuestionType.multiple_choice:
                if q.solution.mc_answer_specs:
                    harness = run_mc_formula_harness(
                        answer_formula_md=q.solution.answer_formula_md,
                        mc_answer_specs=q.solution.mc_answer_specs,
                        params=q.solution.parameters,
                        strict=True,
                    )
                    mc_preview = _compute_mc_preview(
                        answer_formula_md=q.solution.answer_formula_md,
                        mc_answer_specs=q.solution.mc_answer_specs,
                        params=q.solution.parameters,
                    )
                    payload["mc_correct_answer_md"] = harness.correct_answer_md
                    payload["mc_preview_rows"] = mc_preview["rows"]
                    payload["mc_preview_warning"] = mc_preview["warning"]
                    payload["mc_preview_choices"] = mc_preview["preview_choices"]
                    payload["choices_yaml"] = dump_choices_yaml(harness.choices)
                    payload["collisions"] = harness.collisions
                    if harness.collisions:
                        payload["ok"] = False
                        payload["error"] = "MC options are not unique."
                        return JSONResponse(payload, status_code=422)
                else:
                    funcs = [
                        (d.id, d.python_code) for d in q.solution.distractor_python_code
                    ]
                    harness = run_mc_harness(
                        answer_formula_md=q.solution.answer_formula_md,
                        distractor_python_codes=funcs,
                        params=q.solution.parameters,
                    )
                    payload["choices_yaml"] = dump_choices_yaml(harness.choices)
                    payload["collisions"] = harness.collisions
                    if harness.collisions:
                        payload["ok"] = False
                        payload["error"] = "MC options are not unique."
                        return JSONResponse(payload, status_code=422)
            q.solution.last_computed_answer_md = normalize_markdown_math_delimiters(
                answer_result.answer_md
            )
            repo.save_question(q)
            return payload
        except Exception as ex:
            logger.exception(
                "harness.run failed question_id=%s qtype=%s template_len=%s params_keys=%s figures=%s",
                question_id,
                q.question_type.value if "q" in locals() else "unknown",
                (
                    len((q.solution.question_template_md or "").strip())
                    if "q" in locals()
                    else 0
                ),
                sorted((q.solution.parameters or {}).keys()) if "q" in locals() else [],
                [f.id for f in (q.figures or [])] if "q" in locals() else [],
            )
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/ai/generate-mc-distractors")
    def ai_generate_mc_distractors(question_id: str) -> dict:
        try:
            q = repo.get_question(question_id)
            if q.question_type != QuestionType.multiple_choice:
                raise ValueError(
                    "Distractor generation is only available for multiple_choice questions."
                )
            last_collisions: list[str] = []
            last_funcs: list[DistractorFunction] = []
            best_unique_choices: list[MCChoice] = []
            best_attempt = 0
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                result = app.state.ai.generate_distractor_functions(q)
                repo.add_ai_usage(result.usage)
                last_funcs = result.distractors
                harness = run_mc_harness(
                    answer_formula_md=q.solution.answer_formula_md,
                    distractor_python_codes=[
                        (d.id, d.python_code) for d in result.distractors
                    ],
                    params=q.solution.parameters,
                )
                if not harness.collisions:
                    return {
                        "ok": True,
                        "distractor_functions_text": dump_distractor_functions_text(
                            result.distractors
                        ),
                        "choices_yaml": dump_choices_yaml(harness.choices),
                        "attempts": attempt,
                    }
                last_collisions = harness.collisions
                unique_choices = dedupe_choices(harness.choices)
                if len(unique_choices) > len(best_unique_choices):
                    best_unique_choices = unique_choices
                    best_attempt = attempt
                q.solution.distractor_python_code = result.distractors
            if best_unique_choices:
                return {
                    "ok": True,
                    "warning": (
                        "Could not generate a full unique MC set after 3 attempts. "
                        f"Returning {len(best_unique_choices)} unique choices from attempt {best_attempt}."
                    ),
                    "collisions": last_collisions,
                    "distractor_functions_text": dump_distractor_functions_text(
                        last_funcs
                    ),
                    "choices_yaml": dump_choices_yaml(best_unique_choices),
                    "attempts": max_attempts,
                }
            return JSONResponse(
                {
                    "ok": False,
                    "error": "Could not generate MC distractors.",
                    "collisions": last_collisions,
                    "distractor_functions_text": dump_distractor_functions_text(
                        last_funcs
                    ),
                },
                status_code=422,
            )
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/ai/generate-typed-solution")
    def ai_generate_typed_solution(question_id: str) -> dict:
        try:
            q = repo.get_question(question_id)
            result = app.state.ai.generate_typed_solution(q)
            repo.add_ai_usage(result.usage)
            return {
                "ok": True,
                "typed_solution_md": normalize_markdown_math_delimiters(result.text),
                "typed_solution_status": "fresh",
            }
        except Exception as ex:
            logger.exception(
                "ai_generate_typed_solution failed question_id=%s template_len=%s params_keys=%s",
                question_id,
                (
                    len((q.solution.question_template_md or "").strip())
                    if "q" in locals()
                    else 0
                ),
                sorted((q.solution.parameters or {}).keys()) if "q" in locals() else [],
            )
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/ai/preview/{action}")
    def ai_preview_prompt(question_id: str, action: str) -> dict:
        try:
            q = repo.get_question(question_id)
            valid_actions = {
                "rewrite-and-parameterize": "rewrite_parameterize",
                "generate-answer-formula": "generate_answer_formula",
                "generate-mc-distractors": "generate_distractor_functions",
                "generate-typed-solution": "generate_typed_solution",
            }
            if action not in valid_actions:
                raise ValueError("Unknown preview action.")
            preview = app.state.ai.preview_prompt(
                action=valid_actions[action], question=q
            )
            return {"ok": True, **preview}
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/export/docx")
    def export_docx(include_solutions: str | None = Form(None)) -> Response:
        include = include_solutions is not None
        content, warnings = render_project_docx_bytes(
            project_root=project_root, include_solutions=include
        )
        project = repo.load_project()
        filename = f"{_sanitize_docx_filename_stem(project.name)}.docx"
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
        }
        if warnings:
            headers["X-Exam-Helper-Export-Warnings"] = " | ".join(warnings)
        response = Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers=headers,
        )
        if warnings:
            response.set_cookie(
                "exam_helper_export_warning",
                " | ".join(warnings),
                max_age=300,
                samesite="lax",
            )
        return response

    @app.post("/questions/{question_id}/export/docx")
    def export_question_docx(
        question_id: str, include_solutions: str | None = Form(None)
    ) -> Response:
        include = include_solutions is not None
        content, warnings = render_question_docx_bytes(
            project_root=project_root,
            question_id=question_id,
            include_solutions=include,
        )
        question = repo.get_question(question_id)
        filename = f"{_sanitize_docx_filename_stem(question.id)}.docx"
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
        }
        if warnings:
            headers["X-Exam-Helper-Export-Warnings"] = " | ".join(warnings)
        response = Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers=headers,
        )
        if warnings:
            response.set_cookie(
                "exam_helper_export_warning",
                " | ".join(warnings),
                max_age=300,
                samesite="lax",
            )
        return response

    @app.post("/project/settings")
    def save_project_settings(
        openai_model: str = Form(DEFAULT_OPENAI_MODEL),
        prompt_overall: str = Form(""),
        prompt_solution_and_mc: str = Form(""),
        prompt_prompt_review: str = Form(""),
    ) -> RedirectResponse:
        project = repo.load_project()
        project.ai.model = openai_model.strip() or DEFAULT_OPENAI_MODEL
        project.ai.prompts.overall = prompt_overall
        project.ai.prompts.solution_and_mc = prompt_solution_and_mc
        project.ai.prompts.prompt_review = prompt_prompt_review
        repo.save_project(project)
        refresh_ai_service()
        return RedirectResponse("/", status_code=303)

    @app.post("/project/settings/autosave")
    def autosave_project_settings(
        openai_model: str = Form(DEFAULT_OPENAI_MODEL),
        prompt_overall: str = Form(""),
        prompt_solution_and_mc: str = Form(""),
        prompt_prompt_review: str = Form(""),
    ) -> JSONResponse:
        try:
            project = repo.load_project()
            project.ai.model = openai_model.strip() or DEFAULT_OPENAI_MODEL
            project.ai.prompts.overall = prompt_overall
            project.ai.prompts.solution_and_mc = prompt_solution_and_mc
            project.ai.prompts.prompt_review = prompt_prompt_review
            repo.save_project(project)
            refresh_ai_service()
            return JSONResponse({"ok": True})
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.get("/openai/models")
    def list_openai_models() -> dict:
        try:
            models = app.state.ai.list_models()
            return {"ok": True, "models": models}
        except Exception as ex:
            return JSONResponse(
                {"ok": False, "error": str(ex), "models": []}, status_code=422
            )

    @app.post("/project/usage/reset")
    def reset_project_usage() -> RedirectResponse:
        repo.reset_ai_usage()
        return RedirectResponse("/", status_code=303)

    return app
