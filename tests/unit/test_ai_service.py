from __future__ import annotations

from exam_helper.ai_service import AIService
from exam_helper.models import Question


class _FakeResponses:
    def __init__(self, output_text: str):
        self._output_text = output_text

    def create(self, **kwargs):
        class R:
            pass

        r = R()
        r.output_text = self._output_text
        return r


class _FakeClient:
    def __init__(self, output_text: str):
        self.responses = _FakeResponses(output_text)


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


def test_ai_service_generate_answer_function(monkeypatch) -> None:
    from exam_helper import ai_service as mod

    payload = """{"answer_python_code":"def solve(params):\\n    return {'answer_md':'x','final_answer':'x'}"}"""
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: _FakeClient(payload))
    svc = AIService(api_key="k")
    q = Question(id="q1", title="t", prompt_md="old")
    out = svc.generate_answer_function(q)
    assert "def solve(params)" in out.answer_python_code


def test_ai_service_generate_distractor_functions(monkeypatch) -> None:
    from exam_helper import ai_service as mod

    payload = """{"distractors":[{"id":"d1","python_code":"def distractor(params):\\n    return {'distractor_md':'1','rationale':'r'}"},{"id":"d2","python_code":"def distractor(params):\\n    return {'distractor_md':'2','rationale':'r'}"},{"id":"d3","python_code":"def distractor(params):\\n    return {'distractor_md':'3','rationale':'r'}"},{"id":"d4","python_code":"def distractor(params):\\n    return {'distractor_md':'4','rationale':'r'}"}]}"""
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: _FakeClient(payload))
    svc = AIService(api_key="k")
    q = Question(id="q1", title="t", prompt_md="old")
    out = svc.generate_distractor_functions(q)
    assert len(out.distractors) == 4


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
