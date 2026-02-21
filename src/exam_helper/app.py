from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse


def create_app() -> FastAPI:
    app = FastAPI(title="Exam Helper")

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return """
        <html>
          <head><title>Exam Helper</title></head>
          <body>
            <h1>Exam Helper</h1>
            <p>Local-first exam authoring is active.</p>
          </body>
        </html>
        """

    return app
