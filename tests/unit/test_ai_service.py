from __future__ import annotations

import base64
import hashlib
import json

from exam_helper.ai_service import AIService
from exam_helper.models import FigureData, Question


class _FakeResponses:
    def __init__(self, output_text: str | None = None, outputs: list | None = None):
        self._output_text = output_text or ""
        self._outputs = outputs or []

    def create(self, **kwargs):
        class R:
            pass

        r = R()
        r.output_text = self._output_text
        r.output = self._outputs
        r.id = "resp_1"
        return r


class _FakeClient:
    def __init__(self, output_text: str | None = None, outputs: list | None = None):
        self.responses = _FakeResponses(output_text, outputs)


class _SequencedResponses:
    def __init__(self, steps: list[dict]):
        self._steps = list(steps)

    def create(self, **kwargs):
        class R:
            pass

        if not self._steps:
            raise AssertionError("Unexpected extra responses.create() call")
        step = self._steps.pop(0)
        r = R()
        r.output_text = step.get("output_text", "")
        r.output = step.get("output", [])
        r.id = step.get("id", "resp_seq")
        return r


class _FakeToolCall:
    type = "function_call"

    def __init__(self, name: str, arguments: str, call_id: str = "call_1"):
        self.name = name
        self.arguments = arguments
        self.call_id = call_id


class _SequencedClient:
    def __init__(self, steps: list[dict]):
        self.responses = _SequencedResponses(steps)


def test_ai_service_rewrite_parameterize(monkeypatch) -> None:
    from exam_helper import ai_service as mod

    payload = """{"question_template_md":"A car moves at {{v}} m/s","parameters":{"v":12},"title":"Car Motion"}"""
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: _FakeClient(payload))
    svc = AIService(api_key="k")
    q = Question(id="q1", title="", prompt_md="old")
    out = svc.rewrite_parameterize(q)
    assert out.question_template_md == "A car moves at {{v}} m/s"
    assert out.parameters["v"] == 12
    assert out.title == "Car Motion"


def test_ai_service_rewrite_parameterize_accepts_yaml_payload(monkeypatch) -> None:
    from exam_helper import ai_service as mod

    payload = "question_template_md: A car moves at {{v}} m/s\nparameters:\n  v: 9\ntitle: YAML Motion\n"
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: _FakeClient(payload))
    svc = AIService(api_key="k")
    q = Question(id="q1", title="", prompt_md="old")
    out = svc.rewrite_parameterize(q)
    assert out.question_template_md == "A car moves at {{v}} m/s"
    assert out.parameters["v"] == 9
    assert out.title == "YAML Motion"


def test_ai_service_rewrite_parameterize_accepts_question_title_alias(
    monkeypatch,
) -> None:
    from exam_helper import ai_service as mod

    payload = """{"question_template_md":"A car moves at {{v}} m/s","parameters":{"v":9},"question_title":"Photon Work Function"}"""
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: _FakeClient(payload))
    svc = AIService(api_key="k")
    q = Question(id="q1", title="", prompt_md="old")
    out = svc.rewrite_parameterize(q)
    assert out.title == "Photon Work Function"


def test_ai_service_rewrite_parameterize_coerces_string_parameters(monkeypatch) -> None:
    from exam_helper import ai_service as mod

    payload = "\n".join(
        [
            "question_template_md: A car moves at {{v}} m/s",
            "parameters: |",
            "  v: 12",
            "  figure_ref: fig_1",
            'title: ""',
            "",
        ]
    )
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: _FakeClient(payload))
    svc = AIService(api_key="k")
    q = Question(id="q1", title="", prompt_md="old")
    out = svc.rewrite_parameterize(q)
    assert out.parameters["v"] == 12
    assert "figure_ref" not in out.parameters


def test_ai_service_rewrite_parameterize_coerces_list_parameters(monkeypatch) -> None:
    from exam_helper import ai_service as mod

    payload = """{"question_template_md":"A car moves at {{v}} m/s","parameters":[{"name":"v","value":12},{"name":"figure_ref","value":"fig_1"}],"title":""}"""
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: _FakeClient(payload))
    svc = AIService(api_key="k")
    q = Question(id="q1", title="", prompt_md="old")
    out = svc.rewrite_parameterize(q)
    assert out.parameters["v"] == 12
    assert "figure_ref" not in out.parameters


def test_ai_service_rewrite_parameterize_rejects_non_numeric_parameters(
    monkeypatch,
) -> None:
    from exam_helper import ai_service as mod

    payload = """{"question_template_md":"A car moves at {{v}} m/s","parameters":{"v":"10 m/s"},"title":""}"""
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: _FakeClient(payload))
    svc = AIService(api_key="k")
    q = Question(id="q1", title="", prompt_md="old")

    try:
        svc.rewrite_parameterize(q)
    except ValueError as ex:
        assert "numeric scalars" in str(ex)
    else:
        raise AssertionError("rewrite_parameterize should reject non-numeric values")


def test_ai_service_rewrite_parameterize_normalizes_fraction_and_drops_figure_ref(
    monkeypatch,
) -> None:
    from exam_helper import ai_service as mod

    payload = """{"question_template_md":"![]({{figure_ref}}) factor={{work_function_factor}}","parameters":{"figure_ref":"fig_1","work_function_factor":"\\\\tfrac{1}{2}"},"title":""}"""
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: _FakeClient(payload))
    svc = AIService(api_key="k")
    q = Question(id="q1", title="", prompt_md="old")
    out = svc.rewrite_parameterize(q)
    assert "figure_ref" not in out.parameters
    assert out.parameters["work_function_factor"] == 0.5
    assert "{{figure_ref}}" not in out.question_template_md
    assert "fig_1" in out.question_template_md


def test_ai_service_generate_answer_formula(monkeypatch) -> None:
    from exam_helper import ai_service as mod

    payload = """{"answer_formula_md":"x = 1\\nanswer = x"}"""
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: _FakeClient(payload))
    svc = AIService(api_key="k")
    q = Question(id="q1", title="t", prompt_md="old")
    out = svc.generate_answer_formula(q)
    assert "answer = x" in out.answer_formula_md


def test_ai_service_generate_distractor_functions(monkeypatch) -> None:
    from exam_helper import ai_service as mod

    payload = """{"distractors":[{"id":"d1","python_code":"def distractor(params):\\n    return {'distractor_md':'1','rationale':'r'}"},{"id":"d2","python_code":"def distractor(params):\\n    return {'distractor_md':'2','rationale':'r'}"},{"id":"d3","python_code":"def distractor(params):\\n    return {'distractor_md':'3','rationale':'r'}"},{"id":"d4","python_code":"def distractor(params):\\n    return {'distractor_md':'4','rationale':'r'}"}]}"""
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: _FakeClient(payload))
    svc = AIService(api_key="k")
    q = Question(id="q1", title="t", prompt_md="old")
    out = svc.generate_distractor_functions(q)
    assert len(out.distractors) == 4


def test_ai_service_summarize_figure(monkeypatch) -> None:
    from exam_helper import ai_service as mod

    class _RecordingResponses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)

            class R:
                pass

            r = R()
            r.output_text = "A block slides down an incline."
            r.output = []
            r.id = "resp_1"
            return r

    class _RecordingClient:
        def __init__(self):
            self.responses = _RecordingResponses()

    recording_client = _RecordingClient()
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: recording_client)
    svc = AIService(api_key="k")

    out = svc.summarize_figure("image/png", "aGVsbG8=")

    assert out.text == "A block slides down an incline."
    call = recording_client.responses.calls[0]
    assert call["input"][0]["content"][0]["text"].startswith(
        "You write concise figure summaries"
    )
    assert call["input"][1]["content"][1]["image_url"].startswith(
        "data:image/png;base64,"
    )


def test_ai_service_summarize_figure_rejects_invalid_inputs() -> None:
    svc = AIService(api_key="k")

    try:
        svc.summarize_figure("text/plain", "aGVsbG8=")
    except ValueError as ex:
        assert "Unsupported figure mime type" in str(ex)
    else:
        raise AssertionError("summarize_figure should reject unsupported mime types")

    try:
        svc.summarize_figure("image/png", "not-base64")
    except ValueError as ex:
        assert "valid base64" in str(ex)
    else:
        raise AssertionError("summarize_figure should reject invalid base64")


def test_usage_parses_total_cost_from_formatted_string() -> None:
    svc = AIService(api_key="k")

    class _Usage:
        def model_dump(self):
            return {
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
                "total_cost_usd": "$0.0123",
            }

    class _Response:
        usage = _Usage()

    usage = svc._usage_from_response(_Response())
    assert usage.total_tokens == 18
    assert abs(usage.total_cost_usd - 0.0123) < 1e-9


def test_generate_typed_solution_falls_back_to_plain_text(monkeypatch) -> None:
    from exam_helper import ai_service as mod

    monkeypatch.setattr(
        mod, "OpenAI", lambda api_key: _FakeClient("Worked solution in markdown.")
    )
    svc = AIService(api_key="k")
    q = Question(id="q1", title="t")
    out = svc.generate_typed_solution(q)
    assert out.text == "Worked solution in markdown."


def test_generate_typed_solution_extracts_typed_solution_md_from_yaml_like_payload(
    monkeypatch,
) -> None:
    from exam_helper import ai_service as mod

    payload = "typed_solution_md: |\n  Step 1: Use conservation.\n  Final: 2.0 m/s\n"
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: _FakeClient(payload))
    svc = AIService(api_key="k")
    q = Question(id="q1", title="t")
    out = svc.generate_typed_solution(q)
    assert "Step 1" in out.text
    assert "Final: 2.0 m/s" in out.text


def test_ai_service_chat_edit_question_uses_tool_calls(monkeypatch) -> None:
    from exam_helper import ai_service as mod

    monkeypatch.setattr(
        mod,
        "OpenAI",
        lambda api_key: _SequencedClient(
            [
                {
                    "id": "resp_1",
                    "output": [
                        _FakeToolCall(
                            "set_question_text",
                            '{"question_template_md":"A cleaner prompt."}',
                            call_id="call_q",
                        ),
                        _FakeToolCall(
                            "set_answer_formula",
                            '{"answer_formula_md":"x = 6\\nanswer = x"}',
                            call_id="call_a",
                        ),
                        _FakeToolCall("compute_answer", "{}", call_id="call_c"),
                    ],
                },
                {
                    "output": [
                        _FakeToolCall(
                            "finish",
                            '{"assistant_message":"Updated the prompt and answer formula.","warnings":["Validated the deterministic answer formula."]}',
                            call_id="call_finish",
                        )
                    ]
                },
            ]
        ),
    )
    svc = AIService(api_key="k")
    q = Question(id="q1", title="Original")
    q.solution.parameters = {"x": 6}
    q.solution.answer_guidance = "{{answer}}"
    out = svc.chat_edit_question(q, "rewrite this")
    assert out.assistant_message == "Updated the prompt and answer formula."
    assert out.question.solution.question_template_md == "A cleaner prompt."
    assert "answer = x" in out.question.solution.answer_formula_md
    assert out.question.solution.last_computed_answer_md == "6"
    assert out.changed_fields == [
        "answer_formula_md",
        "last_computed_answer_md",
        "question_template_md",
    ]
    assert out.warnings == ["Validated the deterministic answer formula."]


def test_ai_service_chat_edit_question_includes_figure_metadata_in_prompt(
    monkeypatch,
) -> None:
    from exam_helper import ai_service as mod

    class _RecordingResponses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)

            class R:
                pass

            r = R()
            r.output_text = '{"assistant_message":"Updated.","warnings":[]}'
            r.output = []
            r.id = "resp_1"
            return r

    class _RecordingClient:
        def __init__(self):
            self.responses = _RecordingResponses()

    recording_client = _RecordingClient()
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: recording_client)
    svc = AIService(api_key="k")

    b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/w8AAgMBgU7Y5e0AAAAASUVORK5CYII="
    )
    fig = FigureData(
        id="fig_1",
        mime_type="image/png",
        data_base64=b64,
        sha256=hashlib.sha256(base64.b64decode(b64.encode("ascii"))).hexdigest(),
        caption="small diagram",
    )
    q = Question(id="q1", title="Original")
    q.figures = [fig]

    out = svc.chat_edit_question(q, "rewrite this")

    assert out.assistant_message == "Updated."
    system_prompt = recording_client.responses.calls[0]["input"][0]["content"][0][
        "text"
    ]
    assert "screenshot or pasted image" in system_prompt
    assert "fill every relevant field" in system_prompt
    prompt_text = "\n".join(
        item["text"]
        for item in recording_client.responses.calls[0]["input"][1]["content"]
        if item.get("type") == "input_text"
    )
    state_json = prompt_text.split("Current editor state:\n", 1)[1].split(
        "\n\nPersisted recent chat history:\n", 1
    )[0]
    state = json.loads(state_json)
    assert "Figures associated with this question:" in prompt_text
    assert "fig_1: small diagram (image/png)" in prompt_text
    assert '"figure_ids"' in prompt_text
    assert "fig_1" in prompt_text
    assert '"figure_summaries"' in prompt_text
    assert "small diagram" in prompt_text
    assert state["figure_summaries"] == ["fig_1: small diagram (image/png)"]


def test_ai_service_chat_edit_question_truncates_history_by_requested_count(
    monkeypatch,
) -> None:
    from exam_helper import ai_service as mod
    from exam_helper.models import ChatTurn

    class _RecordingResponses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)

            class R:
                pass

            r = R()
            r.output_text = '{"assistant_message":"Updated.","warnings":[]}'
            r.output = []
            r.id = "resp_1"
            return r

    class _RecordingClient:
        def __init__(self):
            self.responses = _RecordingResponses()

    recording_client = _RecordingClient()
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: recording_client)
    svc = AIService(api_key="k")
    q = Question(id="q1", title="Original")
    q.solution.chat_history = [
        ChatTurn(user_message=f"user {idx}", assistant_message=f"assistant {idx}")
        for idx in range(1, 7)
    ]

    out = svc.chat_edit_question(q, "rewrite this", history_keep_count=3)

    assert out.assistant_message == "Updated."
    assert len(recording_client.responses.calls) == 1
    prompt_text = recording_client.responses.calls[0]["input"][1]["content"][0]["text"]
    history_block = prompt_text.split("Persisted recent chat history:\n", 1)[1].split(
        "\n\nCurrent author request:\n", 1
    )[0]
    assert "user 4" in history_block
    assert "assistant 6" in history_block
    assert "user 1" not in history_block
    assert "assistant 3" not in history_block


def test_ai_service_chat_edit_question_uses_full_history_when_keep_is_zero(
    monkeypatch,
) -> None:
    from exam_helper import ai_service as mod
    from exam_helper.models import ChatTurn

    class _RecordingResponses:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)

            class R:
                pass

            r = R()
            r.output_text = '{"assistant_message":"Updated.","warnings":[]}'
            r.output = []
            r.id = "resp_1"
            return r

    class _RecordingClient:
        def __init__(self):
            self.responses = _RecordingResponses()

    recording_client = _RecordingClient()
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: recording_client)
    svc = AIService(api_key="k")
    q = Question(id="q1", title="Original")
    q.solution.chat_history = [
        ChatTurn(user_message=f"user {idx}", assistant_message=f"assistant {idx}")
        for idx in range(1, 4)
    ]

    out = svc.chat_edit_question(q, "rewrite this", history_keep_count=0)

    assert out.assistant_message == "Updated."
    prompt_text = recording_client.responses.calls[0]["input"][1]["content"][0]["text"]
    history_block = prompt_text.split("Persisted recent chat history:\n", 1)[1].split(
        "\n\nCurrent author request:\n", 1
    )[0]
    assert "user 1" in history_block
    assert "assistant 3" in history_block
