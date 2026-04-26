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
        seed = httpx.post(
            base_url + "/questions/save",
            data={
                "question_id": "q_browser",
                "title": "Browser Test",
                "question_type": "free_response",
                "question_template_md": "A block moves with speed {{v}} m/s.",
                "solution_parameters_yaml": "v: 7.5",
                "answer_formula_md": "v = float(v)\nanswer = v",
                "answer_guidance": "Use kinematics: {{v}} m/s.",
                "mc_options_guidance": "Avoid sign-error distractors.",
                "distractor_functions_text": "",
                "choices_yaml": "[]",
                "typed_solution_md": "It moves at 7.5 m/s.",
                "typed_solution_status": "fresh",
                "figures_json": "[]",
                "points": 8,
            },
            follow_redirects=False,
            timeout=5.0,
        )
        assert seed.status_code == 303
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except PlaywrightError as exc:
                pytest.skip(f"Chromium is not available: {exc}")
            try:
                page = browser.new_page()

                page.goto(base_url + "/")
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
                assert page.locator("#answer_formula_md").input_value().strip() == (
                    "v = float(v)\nanswer = v"
                )
                assert page.locator("#rendered_answer_md").inner_text() == (
                    "Use kinematics: 7.5 m/s."
                )
                assert page.locator("#points").input_value() == "8"
                assert (
                    page.locator("#answer_guidance").input_value()
                    == "Use kinematics: {{v}} m/s."
                )
                assert page.locator(
                    "#calculated_variables_md"
                ).inner_text().strip() == ("v = 7.5\nanswer = 7.5")
                assert (
                    page.locator("#mc_options_guidance").input_value()
                    == "Avoid sign-error distractors."
                )
                assert page.locator("#distractor_functions_text").input_value() == ""
                assert page.locator("#typed_solution_status").input_value() == "fresh"

                page.locator("#title").fill("Browser Test Updated")

                page.get_by_role("button", name="Save and Return").click()
                page.wait_for_url(base_url + "/")

                page.get_by_role("link", name="Edit").click()
                page.wait_for_url(base_url + "/questions/q_browser/edit2")

                assert page.locator("#question_id").input_value() == "q_browser"
                assert page.locator("#title").input_value() == "Browser Test Updated"
                assert (
                    page.locator("#question_template_md").input_value()
                    == "A block moves with speed {{v}} m/s."
                )
                assert (
                    page.locator("#solution_parameters_yaml").input_value().strip()
                    == "v: 7.5"
                )
                assert page.locator("#answer_formula_md").input_value().strip() == (
                    "v = float(v)\nanswer = v"
                )
                assert page.locator("#rendered_answer_md").inner_text() == (
                    "Use kinematics: 7.5 m/s."
                )
                assert page.locator("#points").input_value() == "8"
                assert (
                    page.locator("#answer_guidance").input_value()
                    == "Use kinematics: {{v}} m/s."
                )
                assert page.locator(
                    "#calculated_variables_md"
                ).inner_text().strip() == ("v = 7.5\nanswer = 7.5")
                assert (
                    page.locator("#mc_options_guidance").input_value()
                    == "Avoid sign-error distractors."
                )
                assert page.locator("#distractor_functions_text").input_value() == ""
                assert page.locator("#typed_solution_status").input_value() == "fresh"
            finally:
                browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def test_browser_chat_response_updates_visible_mc_rows(tmp_path: Path) -> None:
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import Error as PlaywrightError, sync_playwright

    repo = ProjectRepository(tmp_path)
    repo.init_project("MC Exam", "Physics 1")
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
        seed = httpx.post(
            base_url + "/questions/save",
            data={
                "question_id": "q_browser_mc",
                "title": "Browser MC Test",
                "question_type": "multiple_choice",
                "question_template_md": "What is the value?",
                "solution_parameters_yaml": "x: 2",
                "answer_formula_md": "answer = x",
                "answer_guidance": "The answer is {{answer}}.",
                "mc_options_guidance": "",
                "distractor_functions_text": "",
                "choices_yaml": "[]",
                "typed_solution_md": "",
                "typed_solution_status": "fresh",
                "figures_json": "[]",
                "points": 5,
            },
            follow_redirects=False,
            timeout=5.0,
        )
        assert seed.status_code == 303
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(headless=True)
            except PlaywrightError as exc:
                pytest.skip(f"Chromium is not available: {exc}")
            try:
                page = browser.new_page()
                page.goto(base_url + "/questions/q_browser_mc/edit2")
                page.wait_for_url(base_url + "/questions/q_browser_mc/edit2")

                page.evaluate("""() => {
                        setEditorValuesFromResponse({
                          question_type: "multiple_choice",
                          mc_answer_specs_json: JSON.stringify([
                            { formula_md: "answer = x + 1", rationale_md: "off by one" },
                            { formula_md: "answer = x + 2", rationale_md: "off by two" },
                            { formula_md: "answer = x + 3", rationale_md: "off by three" },
                            { formula_md: "answer = x + 4", rationale_md: "off by four" }
                          ]),
                          mc_preview_rows: [
                            { preview_md: "22.4", warning: "" },
                            { preview_md: "23.4", warning: "" },
                            { preview_md: "24.4", warning: "" },
                            { preview_md: "25.4", warning: "" }
                          ],
                          mc_preview_choices: [
                            { label: "A", content_md: "22.4", rationale: "" },
                            { label: "B", content_md: "23.4", rationale: "" },
                            { label: "C", content_md: "24.4", rationale: "" },
                            { label: "D", content_md: "25.4", rationale: "" }
                          ]
                        });
                    }""")

                assert (
                    page.locator("#mc_answer_1_formula_md").input_value().strip()
                    == "answer = x + 1"
                )
                assert (
                    page.locator("#mc_answer_1_rationale_md").input_value().strip()
                    == "off by one"
                )
                assert (
                    page.locator("#mc_answer_4_formula_md").input_value().strip()
                    == "answer = x + 4"
                )
                assert (
                    page.locator("#mc_answer_4_rationale_md").input_value().strip()
                    == "off by four"
                )
                preview_height = page.locator("#mc_answer_1_preview").evaluate(
                    "(el) => el.getBoundingClientRect().height"
                )
                assert preview_height < 40
            finally:
                browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
