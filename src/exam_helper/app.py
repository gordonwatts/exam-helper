from __future__ import annotations

import json
import logging
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml
from fastapi import Body, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from exam_helper.ai_service import AIService
from exam_helper.export_docx import render_project_docx_bytes
from exam_helper.models import AIUsageTotals, DistractorFunction, MCChoice, ProjectConfig, Question, QuestionType
from exam_helper.repository import ProjectRepository
from exam_helper.solution_runtime import SolutionRuntimeError, run_answer_function, run_mc_harness
from exam_helper.validation import validate_question


class AutosavePayload(BaseModel):
    title: str = ""
    question_type: str = "free_response"
    question_template_md: str = ""
    solution_parameters_yaml: str = "{}"
    answer_guidance: str = ""
    answer_python_code: str = ""
    distractor_functions_text: str = ""
    choices_yaml: str = "[]"
    typed_solution_md: str = ""
    typed_solution_status: str = "missing"
    figures_json: str = "[]"
    points: int = 5


def _sanitize_docx_filename_stem(project_name: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", project_name.lower()).strip("-")
    stem = re.sub(r"-{2,}", "-", stem)
    return stem or "exam"


def create_app(project_root: Path, openai_key: str | None) -> FastAPI:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    app = FastAPI(title="Exam Helper")
    repo = ProjectRepository(project_root)
    project = repo.load_project() if repo.project_file.exists() else ProjectConfig(name="(uninitialized)", course="")

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

    def parse_distractor_functions_text(raw_text: str) -> list[DistractorFunction]:
        text = (raw_text or "").strip()
        if not text:
            return []
        # Backward-compatible: accept YAML list payload as well.
        if text.startswith("- ") or text.startswith("["):
            raw = yaml.safe_load(text) or []
            if not isinstance(raw, list):
                raise ValueError("Distractor functions must be YAML list or plain text blocks.")
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

    def parse_choices_yaml(choices_yaml: str) -> list[MCChoice]:
        def _strip_disallowed_bold(text: str) -> str:
            out = text or ""
            out = re.sub(r"\*\*(.*?)\*\*", r"\1", out, flags=re.DOTALL)
            out = re.sub(r"__(.*?)__", r"\1", out, flags=re.DOTALL)
            out = re.sub(r"</?strong>", "", out, flags=re.IGNORECASE)
            out = re.sub(r"</?b>", "", out, flags=re.IGNORECASE)
            return out

        raw = yaml.safe_load(choices_yaml or "[]") or []
        if not isinstance(raw, list):
            raise ValueError("choices_yaml YAML must parse to a list.")
        choices = []
        for c in raw:
            item = MCChoice.model_validate(c)
            item.content_md = _strip_disallowed_bold(item.content_md)
            if item.rationale is not None:
                item.rationale = _strip_disallowed_bold(item.rationale)
            choices.append(item)
        return sorted(choices, key=lambda c: c.label)

    def dump_choices_yaml(choices: list[MCChoice]) -> str:
        payload = [c.model_dump(mode="json", exclude_none=True) for c in sorted(choices, key=lambda x: x.label)]
        return yaml.safe_dump(payload, sort_keys=False)

    def default_mc_choices_yaml() -> str:
        defaults = [
            {"label": "A", "content_md": "", "is_correct": True, "rationale": ""},
            {"label": "B", "content_md": "", "is_correct": False, "rationale": ""},
            {"label": "C", "content_md": "", "is_correct": False, "rationale": ""},
            {"label": "D", "content_md": "", "is_correct": False, "rationale": ""},
            {"label": "E", "content_md": "", "is_correct": False, "rationale": ""},
        ]
        return yaml.safe_dump(defaults, sort_keys=False)

    def _render_template_from_parameters(template: str, params: dict[str, Any]) -> str:
        rendered = template or ""
        for key, value in (params or {}).items():
            rendered = re.sub(r"\{\{\s*" + re.escape(str(key)) + r"\s*\}\}", str(value), rendered)
        return rendered

    def _suggest_next_question_id() -> str:
        try:
            existing = [q.id for q in repo.list_questions()]
        except Exception:
            existing = []
        used = set(existing)
        for i in range(1, 10000):
            candidate = f"q{i}"
            if candidate not in used:
                return candidate
        return "q_new"

    def _fallback_title_from_prompt(rendered_prompt: str) -> str:
        words = re.findall(r"[A-Za-z0-9+\-/]+", rendered_prompt or "")
        return " ".join(words[:8]).strip() or "Untitled question"

    def _mark_typed_solution_stale_if_needed(existing: Question | None, candidate: Question) -> None:
        if existing is None:
            return
        if (
            existing.solution.parameters != candidate.solution.parameters
            or existing.solution.answer_python_code != candidate.solution.answer_python_code
            or [d.python_code for d in existing.solution.distractor_python_code]
            != [d.python_code for d in candidate.solution.distractor_python_code]
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
        response = templates.TemplateResponse(
            request,
            "index.html",
            {
                "project_root": str(project_root),
                "project": project,
                "questions": questions,
                "export_warning": export_warning,
                "ai_model": (project.ai.model if project else "gpt-5.2"),
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
            "question_form.html",
            {
                "question": None,
                "choices_yaml": default_mc_choices_yaml(),
                "figures_json": "[]",
                "solution_parameters_yaml": dump_parameters_yaml({}),
                "distractor_functions_text": "",
                "question_id_default": _suggest_next_question_id(),
                "ai_enabled": bool(openai_key),
            },
        )

    @app.get("/questions/{question_id}/edit", response_class=HTMLResponse)
    def edit_question(request: Request, question_id: str) -> HTMLResponse:
        q = repo.get_question(question_id)
        choices_yaml = dump_choices_yaml(q.choices) if q.choices else default_mc_choices_yaml()
        figures_json = json.dumps([f.model_dump(mode="json") for f in q.figures])
        solution_parameters_yaml = dump_parameters_yaml(q.solution.parameters)
        distractor_functions_text = dump_distractor_functions_text(q.solution.distractor_python_code)
        return templates.TemplateResponse(
            request,
            "question_form.html",
            {
                "question": q,
                "choices_yaml": choices_yaml,
                "figures_json": figures_json,
                "solution_parameters_yaml": solution_parameters_yaml,
                "distractor_functions_text": distractor_functions_text,
                "ai_enabled": bool(openai_key),
                "question_id_default": q.id,
            },
        )

    @app.post("/questions/save")
    def save_question(
        question_id: str = Form(...),
        title: str = Form(""),
        points: int = Form(5),
        question_type: str = Form("free_response"),
        prompt_md: str = Form(""),
        question_template_md: str = Form(""),
        choices_yaml: str = Form("[]"),
        solution_parameters_yaml: str = Form("{}"),
        answer_guidance: str = Form(""),
        answer_python_code: str = Form(""),
        distractor_functions_text: str = Form(""),
        typed_solution_md: str = Form(""),
        typed_solution_status: str = Form("missing"),
        figures_json: str = Form("[]"),
    ) -> RedirectResponse:
        existing = None
        try:
            existing = repo.get_question(question_id)
        except Exception:
            existing = None
        choices = parse_choices_yaml(choices_yaml)
        figures = json.loads(figures_json or "[]")
        solution_parameters = parse_parameters_yaml(solution_parameters_yaml)
        distractor_funcs = parse_distractor_functions_text(distractor_functions_text)
        rendered_prompt = _render_template_from_parameters(question_template_md, solution_parameters)
        question = Question.model_validate(
            {
                "id": question_id,
                "title": title,
                "points": points,
                "question_type": QuestionType(question_type),
                "choices": choices,
                "solution": {
                    "question_template_md": question_template_md,
                    "parameters": solution_parameters,
                    "answer_guidance": answer_guidance,
                    "answer_python_code": answer_python_code,
                    "distractor_python_code": distractor_funcs,
                    "typed_solution_md": typed_solution_md,
                    "typed_solution_status": typed_solution_status,
                    "last_computed_answer_md": (
                        existing.solution.last_computed_answer_md if existing else ""
                    ),
                },
                "figures": figures,
            }
        )
        _mark_typed_solution_stale_if_needed(existing, question)
        repo.save_question(question)
        return RedirectResponse("/", status_code=303)

    @app.post("/questions/{question_id}/autosave")
    def autosave_question(question_id: str, payload: AutosavePayload = Body(...)) -> JSONResponse:
        try:
            existing = None
            try:
                existing = repo.get_question(question_id)
            except Exception:
                existing = None
            choices = parse_choices_yaml(payload.choices_yaml)
            figures = json.loads(payload.figures_json or "[]")
            solution_parameters = parse_parameters_yaml(payload.solution_parameters_yaml)
            distractor_funcs = parse_distractor_functions_text(payload.distractor_functions_text)
            question = Question.model_validate(
                {
                    "id": question_id,
                    "title": payload.title,
                    "question_type": QuestionType(payload.question_type),
                    "choices": choices,
                    "solution": {
                        "question_template_md": payload.question_template_md,
                        "parameters": solution_parameters,
                        "answer_guidance": payload.answer_guidance,
                        "answer_python_code": payload.answer_python_code,
                        "distractor_python_code": distractor_funcs,
                        "typed_solution_md": payload.typed_solution_md,
                        "typed_solution_status": payload.typed_solution_status,
                        "last_computed_answer_md": (
                            existing.solution.last_computed_answer_md if existing else ""
                        ),
                    },
                    "figures": figures,
                    "points": payload.points,
                }
            )
            _mark_typed_solution_stale_if_needed(existing, question)
            repo.save_question(question)
            return JSONResponse({"ok": True, "typed_solution_status": question.solution.typed_solution_status})
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/figures/validate")
    def validate_figure(data_base64: str = Form(...)) -> dict:
        import base64

        raw = base64.b64decode(data_base64.encode("ascii"))
        return {"sha256": sha256(raw).hexdigest(), "size": len(raw)}

    @app.post("/questions/{question_id}/validate")
    def validate_question_endpoint(question_id: str) -> dict:
        q = repo.get_question(question_id)
        errors = validate_question(q)
        return {"question_id": question_id, "errors": errors, "ok": not errors}

    @app.post("/questions/{question_id}/ai/rewrite-and-parameterize")
    def ai_rewrite_and_parameterize(question_id: str) -> dict:
        try:
            q = repo.get_question(question_id)
            result = app.state.ai.rewrite_parameterize(q)
            repo.add_ai_usage(result.usage)
            rendered_prompt = _render_template_from_parameters(result.question_template_md, result.parameters)
            title = q.title.strip()
            if not title:
                title = result.title.strip() or _fallback_title_from_prompt(rendered_prompt)
            return {
                "ok": True,
                "question_template_md": result.question_template_md,
                "rendered_prompt_md": rendered_prompt,
                "solution_parameters_yaml": dump_parameters_yaml(result.parameters),
                "title": title,
            }
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/ai/generate-answer-function")
    def ai_generate_answer_function(question_id: str) -> dict:
        try:
            q = repo.get_question(question_id)
            result = app.state.ai.generate_answer_function(q)
            repo.add_ai_usage(result.usage)
            return {
                "ok": True,
                "answer_python_code": result.answer_python_code,
            }
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/harness/run")
    def run_harness(question_id: str) -> dict:
        try:
            q = repo.get_question(question_id)
            answer_result = run_answer_function(q.solution.answer_python_code, q.solution.parameters)
            payload: dict[str, Any] = {
                "ok": True,
                "computed_answer_md": answer_result.answer_md,
                "final_answer_text": answer_result.final_answer,
            }
            if q.question_type == QuestionType.multiple_choice:
                funcs = [(d.id, d.python_code) for d in q.solution.distractor_python_code]
                harness = run_mc_harness(
                    answer_python_code=q.solution.answer_python_code,
                    distractor_python_codes=funcs,
                    params=q.solution.parameters,
                )
                payload["choices_yaml"] = dump_choices_yaml(harness.choices)
                payload["collisions"] = harness.collisions
                if harness.collisions:
                    payload["ok"] = False
                    payload["error"] = "MC options are not unique."
                    return JSONResponse(payload, status_code=422)
            q.solution.last_computed_answer_md = answer_result.answer_md
            repo.save_question(q)
            return payload
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/ai/generate-mc-distractors")
    def ai_generate_mc_distractors(question_id: str) -> dict:
        try:
            q = repo.get_question(question_id)
            if q.question_type != QuestionType.multiple_choice:
                raise ValueError("Distractor generation is only available for multiple_choice questions.")
            last_collisions: list[str] = []
            last_funcs: list[DistractorFunction] = []
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                result = app.state.ai.generate_distractor_functions(q)
                repo.add_ai_usage(result.usage)
                last_funcs = result.distractors
                harness = run_mc_harness(
                    answer_python_code=q.solution.answer_python_code,
                    distractor_python_codes=[(d.id, d.python_code) for d in result.distractors],
                    params=q.solution.parameters,
                )
                if not harness.collisions:
                    return {
                        "ok": True,
                        "distractor_functions_text": dump_distractor_functions_text(result.distractors),
                        "choices_yaml": dump_choices_yaml(harness.choices),
                        "attempts": attempt,
                    }
                last_collisions = harness.collisions
                q.solution.distractor_python_code = result.distractors
            return JSONResponse(
                {
                    "ok": False,
                    "error": "Could not generate unique MC distractors after 3 attempts.",
                    "collisions": last_collisions,
                    "distractor_functions_text": dump_distractor_functions_text(last_funcs),
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
                "typed_solution_md": result.text,
                "typed_solution_status": "fresh",
            }
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/ai/preview/{action}")
    def ai_preview_prompt(question_id: str, action: str) -> dict:
        try:
            q = repo.get_question(question_id)
            valid_actions = {
                "rewrite-and-parameterize": "rewrite_parameterize",
                "generate-answer-function": "generate_answer_function",
                "generate-mc-distractors": "generate_distractor_functions",
                "generate-typed-solution": "generate_typed_solution",
            }
            if action not in valid_actions:
                raise ValueError("Unknown preview action.")
            preview = app.state.ai.preview_prompt(action=valid_actions[action], question=q)
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

    @app.post("/project/settings")
    def save_project_settings(
        openai_model: str = Form("gpt-5.2"),
        prompt_overall: str = Form(""),
        prompt_solution_and_mc: str = Form(""),
        prompt_prompt_review: str = Form(""),
    ) -> RedirectResponse:
        project = repo.load_project()
        project.ai.model = openai_model.strip() or "gpt-5.2"
        project.ai.prompts.overall = prompt_overall
        project.ai.prompts.solution_and_mc = prompt_solution_and_mc
        project.ai.prompts.prompt_review = prompt_prompt_review
        repo.save_project(project)
        refresh_ai_service()
        return RedirectResponse("/", status_code=303)

    @app.post("/project/settings/autosave")
    def autosave_project_settings(
        openai_model: str = Form("gpt-5.2"),
        prompt_overall: str = Form(""),
        prompt_solution_and_mc: str = Form(""),
        prompt_prompt_review: str = Form(""),
    ) -> JSONResponse:
        try:
            project = repo.load_project()
            project.ai.model = openai_model.strip() or "gpt-5.2"
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
            return JSONResponse({"ok": False, "error": str(ex), "models": []}, status_code=422)

    @app.post("/project/usage/reset")
    def reset_project_usage() -> RedirectResponse:
        repo.reset_ai_usage()
        return RedirectResponse("/", status_code=303)

    return app
