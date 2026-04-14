from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from exam_helper.repository import ProjectRepository


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(base_url: str, proc: subprocess.Popen[str]) -> None:
    deadline = time.time() + 30
    last_error: str | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout = ""
            if proc.stdout is not None:
                stdout = proc.stdout.read()
            raise RuntimeError(
                f"server exited early with code {proc.returncode}:\n{stdout}"
            )
        try:
            resp = httpx.get(base_url, timeout=1.0)
            if resp.status_code == 200:
                return
            last_error = f"unexpected status {resp.status_code}"
        except Exception as exc:  # pragma: no cover - transient startup
            last_error = str(exc)
        time.sleep(0.2)
    raise TimeoutError(f"server did not become ready: {last_error}")


def test_browser_question_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import Error as PlaywrightError, sync_playwright

    repo = ProjectRepository(tmp_path)
    repo.init_project("Smoke Exam", "Physics 1")
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "exam_helper.cli",
            "serve",
            str(tmp_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_server(base_url + "/", proc)
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except PlaywrightError as exc:
                pytest.skip(f"Chromium is not available: {exc}")
            try:
                page = browser.new_page()

                page.goto(base_url + "/")
                page.get_by_role("link", name="New Question").click()
                page.wait_for_url(base_url + "/questions/new2")

                page.locator("#question_id").fill("q_browser")
                page.locator("#title").fill("Browser Test")
                page.locator("#question_template_md").fill(
                    "A block moves with speed {{v}} m/s."
                )
                page.locator("#solution_parameters_yaml").fill("v: 7.5")
                page.locator("#typed_solution_md").fill("It moves at 7.5 m/s.")
                page.locator("#points").fill("8")

                page.get_by_role("button", name="Save and Return").click()
                page.wait_for_url(base_url + "/")

                page.get_by_role("link", name="Edit").click()
                page.wait_for_url(base_url + "/questions/q_browser/edit2")

                assert page.locator("#question_id").input_value() == "q_browser"
                assert page.locator("#title").input_value() == "Browser Test"
                assert (
                    page.locator("#question_template_md").input_value()
                    == "A block moves with speed {{v}} m/s."
                )
                assert (
                    page.locator("#solution_parameters_yaml").input_value().strip()
                    == "v: 7.5"
                )
                assert (
                    page.locator("#typed_solution_md").input_value()
                    == "It moves at 7.5 m/s."
                )
                assert page.locator("#points").input_value() == "8"
            finally:
                browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
