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


def test_ai_service_improve_prompt(monkeypatch) -> None:
    from exam_helper import ai_service as mod

    monkeypatch.setattr(mod, "OpenAI", lambda api_key: _FakeClient("better prompt"))
    svc = AIService(api_key="k")
    q = Question(id="q1", title="t", prompt_md="old")
    result = svc.improve_prompt(q)
    assert result.text == "better prompt"
    assert result.usage.total_tokens == 0


def test_ai_service_generate_mc_options(monkeypatch) -> None:
    from exam_helper import ai_service as mod

    payload = """```json
[
  {"label":"A","content_md":"1","is_correct":false,"rationale":"r1"},
  {"label":"B","content_md":"2","is_correct":true,"rationale":"r2"},
  {"label":"C","content_md":"3","is_correct":false,"rationale":"r3"},
  {"label":"D","content_md":"4","is_correct":false,"rationale":"r4"},
  {"label":"E","content_md":"5","is_correct":false,"rationale":"r5"}
]
```"""
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: _FakeClient(payload))
    svc = AIService(api_key="k")
    q = Question(id="q1", title="t", prompt_md="old")
    out, usage = svc.generate_mc_options_from_solution(q, "Problem (verbatim): old")
    assert len(out) == 5
    assert sum(1 for c in out if c.is_correct) == 1
    assert usage.total_tokens == 0


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


def test_usage_parses_split_input_and_output_costs() -> None:
    svc = AIService(api_key="k")

    class _Usage:
        def model_dump(self):
            return {
                "input_tokens": 20,
                "output_tokens": 5,
                "input_cost": "0.004",
                "output_cost": "0.0015",
            }

    class _Response:
        usage = _Usage()

    usage = svc._usage_from_response(_Response())
    assert usage.total_tokens == 25
    assert abs(usage.total_cost_usd - 0.0055) < 1e-9
