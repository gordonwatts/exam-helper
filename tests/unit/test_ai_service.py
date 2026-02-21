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
    assert svc.improve_prompt(q) == "better prompt"


def test_ai_service_generate_mc_options(monkeypatch) -> None:
    from exam_helper import ai_service as mod

    payload = """```json
{
  "choices": [
    {"label":"A","content_md":"1","is_correct":false,"rationale":"r1"},
    {"label":"B","content_md":"2","is_correct":true,"rationale":"r2"},
    {"label":"C","content_md":"3","is_correct":false,"rationale":"r3"},
    {"label":"D","content_md":"4","is_correct":false,"rationale":"r4"},
    {"label":"E","content_md":"5","is_correct":false,"rationale":"r5"}
  ],
  "solution_md": "Problem (verbatim): old\\n\\nSolution: \\\\(2+2=4\\\\)"
}
```"""
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: _FakeClient(payload))
    svc = AIService(api_key="k")
    q = Question(id="q1", title="t", prompt_md="old")
    out = svc.generate_mc_options_with_solution(q)
    assert len(out["choices"]) == 5
    assert sum(1 for c in out["choices"] if c.is_correct) == 1
    assert "Problem (verbatim): old" in out["solution_md"]
