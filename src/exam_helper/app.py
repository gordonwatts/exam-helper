from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from exam_helper.repository import ProjectRepository


def create_app(project_root: Path, openai_key: str | None) -> FastAPI:
    app = FastAPI(title="Exam Helper")
    repo = ProjectRepository(project_root)
    app.state.project_root = project_root
    app.state.openai_key = openai_key
    app.state.repo = repo

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        questions = []
        if repo.project_file.exists():
            try:
                questions = repo.list_questions()
            except Exception:
                questions = []
        return """
        <html>
          <head><title>Exam Helper</title></head>
          <body>
            <h1>Exam Helper</h1>
            <p>Local-first exam authoring is active.</p>
            <p>Project root: %s</p>
            <p>Questions: %d</p>
          </body>
        </html>
        """ % (
            project_root,
            len(questions),
        )

    return app
