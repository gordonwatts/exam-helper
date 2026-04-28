from __future__ import annotations

from typing import Any
import re


def coerce_numeric_scalar(value: Any, *, strict: bool = False) -> Any:
    if isinstance(value, bool):
        if strict:
            raise ValueError("parameter values must be numeric scalars.")
        return value
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        if strict:
            raise ValueError("parameter values must be numeric scalars.")
        return value
    text = value.strip()
    if not text:
        if strict:
            raise ValueError("parameter values must be numeric scalars.")
        return value

    if re.fullmatch(r"[+-]?\d+", text):
        try:
            return int(text)
        except Exception:
            if strict:
                raise ValueError("parameter values must be numeric scalars.")
            return value

    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", text):
        try:
            return float(text)
        except Exception:
            if strict:
                raise ValueError("parameter values must be numeric scalars.")
            return value

    frac_match = re.fullmatch(r"([+-]?\d+)\s*/\s*([+-]?\d+)", text)
    if frac_match:
        denom = int(frac_match.group(2))
        if denom != 0:
            return int(frac_match.group(1)) / denom
        if strict:
            raise ValueError("parameter values must be numeric scalars.")
        return value

    latex_frac_match = re.fullmatch(r"\\(?:t?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", text)
    if latex_frac_match:
        denom = int(latex_frac_match.group(2))
        if denom != 0:
            return int(latex_frac_match.group(1)) / denom
    if strict:
        raise ValueError("parameter values must be numeric scalars.")
    return value
