# Validation

Validation is deterministic and code-based (not LLM arithmetic).

## What is checked

- schema correctness
- deterministic checker execution
- symbolic equivalence and units

## Checker contract

In each question YAML, checker code must define:

```python
def grade(student_answer, context):
    return {"verdict": "correct" | "partial" | "incorrect", "score": 0.0, "feedback": "..."}
```

## Run validation

```bash
uv run exam-helper validate my-exam
```
