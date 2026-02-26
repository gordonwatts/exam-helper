from __future__ import annotations

import json
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
from exam_helper.models import AIUsageTotals, MCChoice, ProjectConfig, Question, QuestionType
from exam_helper.repository import ProjectRepository
from exam_helper.solution_runtime import SolutionRuntimeError, run_solution_code
from exam_helper.validation import validate_question


class AutosavePayload(BaseModel):
    title: str = ""
    question_type: str = "free_response"
    prompt_md: str = ""
    mc_options_guidance: str = ""
    choices_yaml: str = "[]"
    solution_md: str = ""
    solution_python_code: str = ""
    solution_parameters_yaml: str = "{}"
    checker_code: str = ""
    figures_json: str = "[]"
    points: int = 5


def _sanitize_docx_filename_stem(project_name: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", project_name.lower()).strip("-")
    stem = re.sub(r"-{2,}", "-", stem)
    return stem or "exam"


def create_app(project_root: Path, openai_key: str | None) -> FastAPI:
    app = FastAPI(title="Exam Helper")
    repo = ProjectRepository(project_root)
    project = repo.load_project() if repo.project_file.exists() else ProjectConfig(name="(uninitialized)", course="")

    def make_ai_service(config: ProjectConfig) -> AIService:
        return AIService(
            api_key=openai_key,
            model=config.ai.model,
            prompts_override=config.ai.prompts,
        )

    def unpack_ai_text_and_usage(result: object) -> tuple[str, AIUsageTotals]:
        if isinstance(result, str):
            return result, AIUsageTotals()
        if hasattr(result, "text") and hasattr(result, "usage"):
            text = str(getattr(result, "text", ""))
            usage = getattr(result, "usage")
            if isinstance(usage, AIUsageTotals):
                return text, usage
            return text, AIUsageTotals.model_validate(usage)
        raise ValueError("Unexpected AI response shape.")

    def refresh_ai_service() -> None:
        latest = repo.load_project()
        app.state.ai = make_ai_service(latest)

    app.state.project_root = project_root
    app.state.openai_key = openai_key
    app.state.repo = repo
    app.state.ai = make_ai_service(project)
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    computed_output_start = "<!-- COMPUTED_OUTPUT_START -->"
    computed_output_end = "<!-- COMPUTED_OUTPUT_END -->"

    def parse_parameters_yaml(raw_yaml: str) -> dict[str, Any]:
        data = yaml.safe_load(raw_yaml or "{}")
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError("Solution parameters YAML must be a mapping.")
        return data

    def dump_parameters_yaml(params: dict[str, Any]) -> str:
        return yaml.safe_dump(params or {}, sort_keys=False).strip()

    def apply_computed_output(solution_md: str, final_answer_text: str) -> str:
        block = (
            f"{computed_output_start}\n"
            "### Computed final result\n\n"
            f"{final_answer_text.strip()}\n"
            f"{computed_output_end}"
        )
        text = (solution_md or "").strip()
        start = text.find(computed_output_start)
        end = text.find(computed_output_end)
        if start >= 0 and end > start:
            end_idx = end + len(computed_output_end)
            merged = f"{text[:start].rstrip()}\n\n{block}"
            if end_idx < len(text):
                merged += f"\n{text[end_idx:].lstrip()}"
            return merged.strip()
        if not text:
            return block
        return f"{text}\n\n{block}".strip()

    def parse_choices_yaml(choices_yaml: str) -> list[MCChoice]:
        raw_choices = yaml.safe_load(choices_yaml) or []
        choices = [MCChoice.model_validate(c) for c in raw_choices]
        if len(choices) != 5:
            raise ValueError("choices_yaml must define exactly five options (A-E).")
        if sum(1 for c in choices if c.is_correct) != 1:
            raise ValueError("choices_yaml must mark exactly one correct option.")
        labels = sorted((c.label or "").strip().upper() for c in choices)
        if labels != ["A", "B", "C", "D", "E"]:
            raise ValueError("choices_yaml options must be labeled A-E.")
        return sorted(choices, key=lambda c: c.label)

    def default_mc_choices_yaml() -> str:
        defaults = [
            {"label": "A", "content_md": "", "is_correct": True},
            {"label": "B", "content_md": "", "is_correct": False},
            {"label": "C", "content_md": "", "is_correct": False},
            {"label": "D", "content_md": "", "is_correct": False},
            {"label": "E", "content_md": "", "is_correct": False},
        ]
        return yaml.safe_dump(defaults, sort_keys=False)

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
            },
        )

    @app.get("/questions/{question_id}/edit", response_class=HTMLResponse)
    def edit_question(request: Request, question_id: str) -> HTMLResponse:
        q = repo.get_question(question_id)
        choices_yaml = yaml.safe_dump(
            [c.model_dump(mode="json") for c in q.choices], sort_keys=False
        )
        figures_json = json.dumps([f.model_dump(mode="json") for f in q.figures])
        solution_parameters_yaml = dump_parameters_yaml(q.solution.parameters)
        return templates.TemplateResponse(
            request,
            "question_form.html",
            {
                "question": q,
                "choices_yaml": choices_yaml if choices_yaml.strip() else default_mc_choices_yaml(),
                "figures_json": figures_json,
                "solution_parameters_yaml": solution_parameters_yaml,
            },
        )

    @app.post("/questions/save")
    def save_question(
        question_id: str = Form(...),
        title: str = Form(""),
        topic: str = Form(""),
        course_level: str = Form("intro"),
        tags: str = Form(""),
        points: int = Form(5),
        question_type: str = Form("free_response"),
        prompt_md: str = Form(""),
        mc_options_guidance: str = Form(""),
        choices_yaml: str = Form("[]"),
        solution_md: str = Form(""),
        solution_python_code: str = Form(""),
        solution_parameters_yaml: str = Form("{}"),
        checker_code: str = Form(""),
        figures_json: str = Form("[]"),
    ) -> RedirectResponse:
        raw_choices = yaml.safe_load(choices_yaml) or []
        choices = [MCChoice.model_validate(c) for c in raw_choices]
        figures = json.loads(figures_json or "[]")
        solution_parameters = parse_parameters_yaml(solution_parameters_yaml)

        question = Question.model_validate(
            {
                "id": question_id,
                "title": title,
                "topic": topic,
                "course_level": course_level,
                "tags": [t.strip() for t in tags.split(",") if t.strip()],
                "points": points,
                "question_type": QuestionType(question_type),
                "prompt_md": prompt_md,
                "mc_options_guidance": mc_options_guidance,
                "choices": choices,
                "solution": {
                    "worked_solution_md": solution_md,
                    "python_code": solution_python_code,
                    "parameters": solution_parameters,
                },
                "checker": {"python_code": checker_code},
                "figures": figures,
            }
        )
        repo.save_question(question)
        return RedirectResponse("/", status_code=303)

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

    @app.post("/questions/{question_id}/autosave")
    def autosave_question(question_id: str, payload: AutosavePayload = Body(...)) -> JSONResponse:
        try:
            raw_choices = yaml.safe_load(payload.choices_yaml) or []
            choices = [MCChoice.model_validate(c) for c in raw_choices]
            figures = json.loads(payload.figures_json or "[]")
            solution_parameters = parse_parameters_yaml(payload.solution_parameters_yaml)
            question = Question.model_validate(
                {
                    "id": question_id,
                    "title": payload.title,
                    "question_type": QuestionType(payload.question_type),
                    "prompt_md": payload.prompt_md,
                    "mc_options_guidance": payload.mc_options_guidance,
                    "choices": choices,
                    "solution": {
                        "worked_solution_md": payload.solution_md,
                        "python_code": payload.solution_python_code,
                        "parameters": solution_parameters,
                    },
                    "checker": {"python_code": payload.checker_code},
                    "figures": figures,
                    "points": payload.points,
                }
            )
            repo.save_question(question)
            return JSONResponse({"ok": True})
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

    @app.post("/questions/{question_id}/ai/improve-prompt")
    def ai_improve_prompt(question_id: str) -> dict:
        try:
            q = repo.get_question(question_id)
            result = app.state.ai.improve_prompt(q)
            text, usage = unpack_ai_text_and_usage(result)
            repo.add_ai_usage(usage)
            return {"ok": True, "prompt_md": text}
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/ai/draft-solution")
    def ai_draft_solution(question_id: str) -> dict:
        try:
            q = repo.get_question(question_id)
            if hasattr(app.state.ai, "draft_solution_with_code"):
                error_feedback = ""
                for _ in range(3):
                    result = app.state.ai.draft_solution_with_code(q, error_feedback=error_feedback)
                    repo.add_ai_usage(AIUsageTotals.model_validate(result.usage))
                    candidate = q.model_copy(deep=True)
                    candidate.solution.worked_solution_md = result.worked_solution_md
                    candidate.solution.python_code = result.python_code
                    candidate.solution.parameters = result.parameters
                    try:
                        run_result = run_solution_code(
                            candidate.solution.python_code,
                            candidate.solution.parameters,
                            {"question_type": candidate.question_type.value, "prompt_md": candidate.prompt_md},
                        )
                    except SolutionRuntimeError as ex:
                        error_feedback = str(ex)
                        q = candidate
                        continue
                    solution_md = apply_computed_output(
                        candidate.solution.worked_solution_md,
                        run_result.final_answer_text,
                    )
                    payload: dict[str, Any] = {
                        "ok": True,
                        "solution_md": solution_md,
                        "solution_python_code": candidate.solution.python_code,
                        "solution_parameters_yaml": dump_parameters_yaml(candidate.solution.parameters),
                        "final_answer_text": run_result.final_answer_text,
                    }
                    if run_result.choices_yaml:
                        payload["choices_yaml"] = run_result.choices_yaml
                    return payload
                raise ValueError(
                    f"AI-generated solution code failed validation after 3 attempts. Last error: {error_feedback}"
                )
            result = app.state.ai.draft_solution(q)
            text, usage = unpack_ai_text_and_usage(result)
            repo.add_ai_usage(usage)
            return {
                "ok": True,
                "solution_md": text,
                "solution_python_code": q.solution.python_code,
                "solution_parameters_yaml": dump_parameters_yaml(q.solution.parameters),
            }
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/ai/suggest-title")
    def ai_suggest_title(question_id: str) -> dict:
        try:
            q = repo.get_question(question_id)
            result = app.state.ai.suggest_title(q)
            text, usage = unpack_ai_text_and_usage(result)
            repo.add_ai_usage(usage)
            return {"ok": True, "title": text}
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/ai/distractors")
    def ai_distractors(question_id: str, count: int = 3) -> dict:
        _ = count
        try:
            q = repo.get_question(question_id)
            if not q.solution.python_code.strip():
                raise ValueError("Solution python code is required to generate MC choices.")
            run_result = run_solution_code(
                q.solution.python_code,
                q.solution.parameters,
                {"question_type": q.question_type.value, "prompt_md": q.prompt_md},
            )
            if not run_result.choices_yaml:
                raise ValueError("Solution code did not return choices_yaml for MC generation.")
            choices = parse_choices_yaml(run_result.choices_yaml)
            solution_md = apply_computed_output(q.solution.worked_solution_md, run_result.final_answer_text)
            payload = {
                "ok": True,
                "choices_yaml": yaml.safe_dump(
                    [c.model_dump(mode="json") for c in choices], sort_keys=False
                ),
                "solution_was_generated": False,
                "solution_md": solution_md,
                "final_answer_text": run_result.final_answer_text,
            }
            return payload
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/solution-code/run")
    def run_solution_code_endpoint(question_id: str) -> dict:
        try:
            q = repo.get_question(question_id)
            result = run_solution_code(
                q.solution.python_code,
                q.solution.parameters,
                {"question_type": q.question_type.value, "prompt_md": q.prompt_md},
            )
            payload: dict[str, Any] = {
                "ok": True,
                "final_answer_text": result.final_answer_text,
                "solution_md": apply_computed_output(q.solution.worked_solution_md, result.final_answer_text),
            }
            if result.choices_yaml:
                payload["choices_yaml"] = result.choices_yaml
            return payload
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/ai/sync-parameters")
    def ai_sync_parameters(question_id: str) -> dict:
        try:
            q = repo.get_question(question_id)
            if not hasattr(app.state.ai, "sync_parameters_draft"):
                raise ValueError("Current AI provider does not support parameter sync drafting.")
            draft, usage = app.state.ai.sync_parameters_draft(q)
            repo.add_ai_usage(AIUsageTotals.model_validate(usage))
            payload: dict[str, Any] = {
                "ok": True,
                "prompt_md": draft["prompt_md"],
                "solution_md": draft["worked_solution_md"],
            }
            try:
                run_result = run_solution_code(
                    q.solution.python_code,
                    q.solution.parameters,
                    {"question_type": q.question_type.value, "prompt_md": draft["prompt_md"]},
                )
                payload["final_answer_text"] = run_result.final_answer_text
                payload["solution_md"] = apply_computed_output(
                    draft["worked_solution_md"], run_result.final_answer_text
                )
                if run_result.choices_yaml:
                    payload["choices_yaml"] = run_result.choices_yaml
            except Exception as ex:
                payload["run_error"] = str(ex)
            return payload
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    @app.post("/questions/{question_id}/ai/preview/{action}")
    def ai_preview_prompt(question_id: str, action: str) -> dict:
        try:
            q = repo.get_question(question_id)
            valid_actions = {
                "suggest-title",
                "improve-prompt",
                "draft-solution",
                "draft-solution-with-code",
                "distractors",
                "sync-parameters",
            }
            if action not in valid_actions:
                raise ValueError("Unknown preview action.")
            normalized_action = action.replace("-", "_")
            solution_md = (
                q.solution.worked_solution_md
                if normalized_action
                in {"draft_solution", "draft_solution_with_code", "distractors", "sync_parameters"}
                else ""
            )
            preview = app.state.ai.preview_prompt(
                action=normalized_action,
                question=q,
                solution_md=solution_md,
            )
            return {"ok": True, **preview}
        except Exception as ex:
            return JSONResponse({"ok": False, "error": str(ex)}, status_code=422)

    return app
