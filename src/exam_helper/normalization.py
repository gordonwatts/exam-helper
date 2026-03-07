from __future__ import annotations

import io
import re
import tokenize

_STRING_PREFIX_RE = re.compile(r"(?is)^([rubf]*)('''|\"\"\"|'|\")")
_VALID_SINGLE_CHAR_ESCAPES = {"\\", "'", '"', "a", "b", "f", "n", "r", "t", "v"}
_VALID_MULTI_CHAR_ESCAPES = {"x", "u", "U", "N"}


def normalize_markdown_math_delimiters(text: str) -> str:
    """Convert LaTeX \\(...\\), \\[...\\] delimiters to dollar-style markdown."""
    out = text or ""
    out = re.sub(r"\\+\((.*?)\\+\)", r"$\1$", out, flags=re.DOTALL)
    out = re.sub(r"\\+\[(.*?)\\+\]", r"$$\1$$", out, flags=re.DOTALL)
    return out


def _split_string_token(token_text: str) -> tuple[str, str, str] | None:
    m = _STRING_PREFIX_RE.match(token_text)
    if not m:
        return None
    prefix = m.group(1)
    quote = m.group(2)
    end_len = len(quote)
    if len(token_text) < (len(prefix) + end_len * 2):
        return None
    body = token_text[len(prefix) + end_len : -end_len]
    return prefix, quote, body


def _escape_latex_backslashes(body: str) -> str:
    out: list[str] = []
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= n:
            out.append("\\")
            i += 1
            continue
        nxt = body[i + 1]
        if nxt == "\\":
            out.append("\\\\")
            i += 2
            continue
        if nxt.isalpha():
            j = i + 2
            while j < n and body[j].isalpha():
                j += 1
            command = body[i + 1 : j]
            if len(command) >= 2:
                out.append("\\\\" + command)
                i = j
                continue
            if nxt in _VALID_SINGLE_CHAR_ESCAPES or nxt in _VALID_MULTI_CHAR_ESCAPES:
                out.append("\\" + nxt)
            else:
                out.append("\\\\" + nxt)
            i += 2
            continue
        if nxt in _VALID_SINGLE_CHAR_ESCAPES or nxt in _VALID_MULTI_CHAR_ESCAPES:
            out.append("\\" + nxt)
            i += 2
            continue
        if nxt.isdigit():
            out.append("\\" + nxt)
            i += 2
            continue
        out.append("\\\\" + nxt)
        i += 2
    return "".join(out)


def normalize_python_code_string_literals(code: str) -> str:
    """Escape backslashes in non-raw string tokens when they look LaTeX-like."""
    source = code or ""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return source

    updated: list[tokenize.TokenInfo] = []
    for token in tokens:
        if token.type != tokenize.STRING:
            updated.append(token)
            continue
        split = _split_string_token(token.string)
        if split is None:
            updated.append(token)
            continue
        prefix, quote, body = split
        if "r" in prefix.casefold():
            updated.append(token)
            continue
        rebuilt = f"{prefix}{quote}{_escape_latex_backslashes(body)}{quote}"
        updated.append(token._replace(string=rebuilt))
    return tokenize.untokenize(updated)
