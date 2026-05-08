from __future__ import annotations

import logging
from pathlib import Path

import typer
import uvicorn

from exam_helper.app import create_app
from exam_helper.config import resolve_openai_api_key
from exam_helper.export_docx import export_project_to_docx
from exam_helper.models import DEFAULT_OPENAI_MODEL
from exam_helper.repository import ProjectRepository
from exam_helper.validation import validate_project

cli_app = typer.Typer(add_completion=False, help="Exam helper commands.")
export_app = typer.Typer(add_completion=False, help="Export artifacts.")
cli_app.add_typer(export_app, name="export")


def _configure_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s:%(message)s")


def cmd_init(
    path: str | Path,
    name: str = "Example Exam Project",
    course: str = "Calculus-based Intro Physics",
    openai_model: str = DEFAULT_OPENAI_MODEL,
) -> int:
    root = Path(path)
    repo = ProjectRepository(root)
    repo.init_project(name=name, course=course, openai_model=openai_model)
    return 0


def cmd_validate(path: str | Path) -> int:
    resolved = Path(path)
    if not resolved.exists():
        raise typer.BadParameter(f"Path does not exist: {resolved}")
    if resolved.is_file():
        resolved = resolved.parent
    repo = ProjectRepository(resolved)
    errors = validate_project(repo)
    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1
    print(f"Validation succeeded: {resolved}")
    return 0


def cmd_export_docx(
    path: str | Path, output: str | Path, include_solutions: bool = False
) -> int:
    warnings = export_project_to_docx(
        project_root=Path(path),
        output_path=Path(output),
        include_solutions=include_solutions,
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"Wrote {output}")
    return 0


def cmd_serve(
    path: str | Path = ".",
    host: str = "127.0.0.1",
    port: int = 8000,
    openai_key: str | None = None,
    verbose: int = 0,
) -> int:
    _configure_logging(verbose)

    resolved_path = Path(path)
    key = resolve_openai_api_key(openai_key, resolved_path)
    if key:
        print("OpenAI key loaded.")
    else:
        print("OpenAI key not configured. AI features will be unavailable.")
    app = create_app(project_root=resolved_path, openai_key=key)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=("debug" if verbose >= 2 else "info"),
    )
    return 0


@cli_app.command(help="Initialize a new exam project.")
def init(
    path: Path,
    name: str = typer.Option("Example Exam Project", help="Project name."),
    course: str = typer.Option("Calculus-based Intro Physics", help="Course name."),
    openai_model: str = typer.Option(DEFAULT_OPENAI_MODEL, help="Default AI model."),
) -> None:
    raise typer.Exit(
        code=cmd_init(path, name=name, course=course, openai_model=openai_model)
    )


@cli_app.command(help="Validate question files.")
def validate(path: Path) -> None:
    raise typer.Exit(code=cmd_validate(path))


@export_app.command("docx", help="Export DOCX.")
def export_docx(
    path: Path,
    output: Path = typer.Option(..., help="Output DOCX path."),
    include_solutions: bool = typer.Option(
        False, help="Include solutions in the export."
    ),
) -> None:
    raise typer.Exit(code=cmd_export_docx(path, output, include_solutions))


@cli_app.command(
    help=(
        "Serve the local web app.\n\n"
        "The path should point at the project root containing project.yaml.\n\n"
        "OpenAI key resolution:\n"
        "  1. --openai-key\n"
        "  2. EXAM_HELPER_OPENAI_KEY in the project .env path\n"
        "  3. EXAM_HELPER_OPENAI_KEY in ~/.env"
    )
)
def serve(
    path: Path = typer.Argument(
        Path("."),
        help="Path to the exam project directory.",
    ),
    host: str = typer.Option("127.0.0.1", help="Host to bind."),
    port: int = typer.Option(8000, help="Port to bind."),
    openai_key: str | None = typer.Option(
        None,
        "--openai-key",
        help="OpenAI API key override.",
    ),
    verbose: int = typer.Option(
        0,
        "-v",
        "--verbose",
        count=True,
        help="Increase logging verbosity; repeat for more detail.",
    ),
) -> None:
    raise typer.Exit(
        code=cmd_serve(
            path, host=host, port=port, openai_key=openai_key, verbose=verbose
        )
    )


def main() -> int:
    try:
        cli_app(prog_name="exam-helper", standalone_mode=False)
    except typer.Exit as exc:
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
