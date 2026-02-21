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


def test_ai_service_distractors(monkeypatch) -> None:
    from exam_helper import ai_service as mod

    payload = '[{"label":"B","content_md":"wrong","rationale":"sign error"}]'
    monkeypatch.setattr(mod, "OpenAI", lambda api_key: _FakeClient(payload))
    svc = AIService(api_key="k")
    q = Question(id="q1", title="t", prompt_md="old")
    out = svc.distractors(q, count=1)
    assert len(out) == 1
    assert out[0].is_correct is False
