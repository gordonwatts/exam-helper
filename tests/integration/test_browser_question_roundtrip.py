from __future__ import annotations

import base64
import hashlib
import json
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
                page.wait_for_url(base_url + "/questions/q_browser/edit")

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
                page.wait_for_url(base_url + "/questions/q_browser/edit")

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
                page.goto(base_url + "/questions/q_browser_mc/edit")
                page.wait_for_url(base_url + "/questions/q_browser_mc/edit")

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


def test_browser_figure_preview_modal_opens_and_closes(tmp_path: Path) -> None:
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import Error as PlaywrightError, sync_playwright

    repo = ProjectRepository(tmp_path)
    repo.init_project("Figure Exam", "Physics 1")
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"

    # 1x1 transparent PNG
    b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/w8AAgMBgU7Y5e0AAAAASUVORK5CYII="
    )
    sha256 = hashlib.sha256(base64.b64decode(b64.encode("ascii"))).hexdigest()
    figures_json = json.dumps(
        [
            {
                "id": "fig_1",
                "mime_type": "image/png",
                "data_base64": b64,
                "sha256": sha256,
                "caption": "tiny figure",
            }
        ]
    )

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
                "question_id": "q_figure",
                "title": "Figure Test",
                "question_type": "free_response",
                "question_template_md": "Look at <figure fig_1>.",
                "solution_parameters_yaml": "{}",
                "answer_formula_md": "answer = 1",
                "answer_guidance": "",
                "mc_options_guidance": "",
                "distractor_functions_text": "",
                "choices_yaml": "[]",
                "typed_solution_md": "",
                "typed_solution_status": "fresh",
                "figures_json": figures_json,
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
                page.goto(base_url + "/questions/q_figure/edit")
                page.wait_for_url(base_url + "/questions/q_figure/edit")

                preview_button = page.locator("[data-open-figure='0']")
                assert preview_button.is_visible()
                assert (
                    preview_button.locator("img")
                    .get_attribute("src")
                    .startswith("data:image/png;base64,")
                )
                preview_button.click()

                modal = page.locator("#figure_modal")
                assert modal.get_attribute("aria-hidden") == "false"
                assert page.locator("#figure_modal_title").inner_text() == "fig_1"
                assert "tiny figure" in page.locator("#figure_modal_meta").inner_text()
                assert modal.locator("#figure_modal_image").get_attribute("src")

                page.locator("#figure_modal .figure-modal__backdrop").click()
                assert modal.get_attribute("aria-hidden") == "true"

                page.get_by_role("button", name="Save and Return").click()
                page.wait_for_url(base_url + "/")

                page.get_by_role("link", name="Edit").click()
                page.wait_for_url(base_url + "/questions/q_figure/edit")

                preview_button = page.locator("[data-open-figure='0']")
                assert preview_button.is_visible()
                assert (
                    preview_button.locator("img")
                    .get_attribute("src")
                    .startswith("data:image/png;base64,")
                )
                preview_button.click()
                page.locator("#figure_modal_close").click()
                assert modal.get_attribute("aria-hidden") == "true"
            finally:
                browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def test_browser_figure_drop_accepts_jpg_svg_and_warns_on_bad_files(
    tmp_path: Path,
) -> None:
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import Error as PlaywrightError, sync_playwright

    repo = ProjectRepository(tmp_path)
    repo.init_project("Figure Upload Exam", "Physics 1")
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
                "question_id": "q_upload",
                "title": "Figure Upload Test",
                "question_type": "free_response",
                "question_template_md": "Upload a figure.",
                "solution_parameters_yaml": "{}",
                "answer_formula_md": "answer = 1",
                "answer_guidance": "",
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
                page.goto(base_url + "/questions/q_upload/edit")
                page.wait_for_url(base_url + "/questions/q_upload/edit")

                page.evaluate("""() => {
                        const section = document.querySelector("#figures_section");
                        const transfer = new DataTransfer();
                        transfer.items.add(new File(
                          [new Uint8Array([1, 2, 3, 4])],
                          "photo.jpg",
                          { type: "" }
                        ));
                        transfer.items.add(new File(
                          [new Uint8Array([5, 6, 7, 8])],
                          "diagram.svg",
                          { type: "" }
                        ));
                        transfer.items.add(new File(
                          [new TextEncoder().encode("not an image")],
                          "notes.txt",
                          { type: "text/plain" }
                        ));
                        section.dispatchEvent(new DragEvent("dragover", {
                          bubbles: true,
                          cancelable: true,
                          dataTransfer: transfer
                        }));
                        section.dispatchEvent(new DragEvent("drop", {
                          bubbles: true,
                          cancelable: true,
                          dataTransfer: transfer
                        }));
                    }""")

                page.wait_for_function(
                    "() => document.querySelectorAll('#figures_preview .figure-item').length === 2"
                )

                notice = page.locator("#figure_notice")
                assert notice.is_visible()
                assert "notes.txt" in notice.inner_text()
                assert ".jpg" in notice.inner_text()

                first_src = page.locator(
                    "#figures_preview .figure-item:nth-of-type(1) img"
                ).get_attribute("src")
                second_src = page.locator(
                    "#figures_preview .figure-item:nth-of-type(2) img"
                ).get_attribute("src")
                assert first_src is not None
                assert second_src is not None
                assert first_src.startswith("data:image/jpeg;base64,")
                assert second_src.startswith("data:image/svg+xml;base64,")
            finally:
                browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
