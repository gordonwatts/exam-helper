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
