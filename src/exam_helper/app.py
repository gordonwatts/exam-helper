from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import yaml
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from exam_helper.ai_service import AIService
from exam_helper.models import MCChoice, Question, QuestionType
from exam_helper.repository import ProjectRepository
from exam_helper.validation import validate_question


def create_app(project_root: Path, openai_key: str | None) -> FastAPI:
    app = FastAPI(title="Exam Helper")
    repo = ProjectRepository(project_root)
    app.state.project_root = project_root
    app.state.openai_key = openai_key
    app.state.repo = repo
    app.state.ai = AIService(api_key=openai_key)
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request) -> HTMLResponse:
        questions = repo.list_questions() if repo.project_file.exists() else []
        project = repo.load_project() if repo.project_file.exists() else None
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "project_root": str(project_root),
                "project": project,
                "questions": questions,
            },
        )

    @app.get("/questions/new", response_class=HTMLResponse)
    def new_question(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "question_form.html",
            {
                "question": None,
                "choices_yaml": "- label: A\n  content_md: ''\n  is_correct: true\n"
                "- label: B\n  content_md: ''\n  is_correct: false\n",
                "figures_json": "[]",
            },
        )

    @app.get("/questions/{question_id}/edit", response_class=HTMLResponse)
    def edit_question(request: Request, question_id: str) -> HTMLResponse:
        q = repo.get_question(question_id)
        choices_yaml = yaml.safe_dump(
            [c.model_dump(mode="json") for c in q.choices], sort_keys=False
        )
        figures_json = json.dumps([f.model_dump(mode="json") for f in q.figures])
        return templates.TemplateResponse(
            request,
            "question_form.html",
            {
                "question": q,
                "choices_yaml": choices_yaml,
                "figures_json": figures_json,
            },
        )

    @app.post("/questions/save")
    def save_question(
        question_id: str = Form(...),
        title: str = Form(...),
        topic: str = Form(""),
        course_level: str = Form("intro"),
        tags: str = Form(""),
        question_type: str = Form("free_response"),
        prompt_md: str = Form(""),
        choices_yaml: str = Form("[]"),
        solution_md: str = Form(""),
        checker_code: str = Form(""),
        figures_json: str = Form("[]"),
    ) -> RedirectResponse:
        raw_choices = yaml.safe_load(choices_yaml) or []
        choices = [MCChoice.model_validate(c) for c in raw_choices]
        figures = json.loads(figures_json or "[]")

        question = Question.model_validate(
            {
                "id": question_id,
                "title": title,
                "topic": topic,
                "course_level": course_level,
                "tags": [t.strip() for t in tags.split(",") if t.strip()],
                "question_type": QuestionType(question_type),
                "prompt_md": prompt_md,
                "choices": choices,
                "solution": {"worked_solution_md": solution_md},
                "checker": {"python_code": checker_code},
                "figures": figures,
            }
        )
        repo.save_question(question)
        return RedirectResponse("/", status_code=303)

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
        q = repo.get_question(question_id)
        text = app.state.ai.improve_prompt(q)
        return {"draft_prompt_md": text}

    @app.post("/questions/{question_id}/ai/draft-solution")
    def ai_draft_solution(question_id: str) -> dict:
        q = repo.get_question(question_id)
        text = app.state.ai.draft_solution(q)
        return {"draft_solution_md": text}

    @app.post("/questions/{question_id}/ai/distractors")
    def ai_distractors(question_id: str, count: int = Form(3)) -> dict:
        q = repo.get_question(question_id)
        choices = app.state.ai.distractors(q, count=count)
        return {"choices": [c.model_dump(mode="json") for c in choices]}

    return app
