from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass
class FormattedModelOutput:
    code: str
    explanation: str
    is_empty: bool = False
    is_fallback: bool = False


FENCE_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_+#.-]*)\s*\n?(?P<code>.*?)```", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
EXPLANATION_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:explanation|notes?|changes?|why|reasoning)\s*:?\s*(?:\*\*)?\s*$",
    re.IGNORECASE,
)
INLINE_EXPLANATION_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:explanation|notes?|changes?|why|reasoning)\s*:?\s*(?:\*\*)?\s*",
    re.IGNORECASE,
)


def strip_code_fences(text: str) -> str:
    formatted = extract_code_and_explanation(text)
    if formatted.code:
        return formatted.code
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_code_and_explanation(text: str, language: str = "python") -> FormattedModelOutput:
    raw = THINK_RE.sub("", text or "").strip()
    if not raw:
        return FormattedModelOutput(
            code="",
            explanation="The local model returned an empty response.",
            is_empty=True,
        )

    if _looks_like_fallback(raw):
        return FormattedModelOutput(
            code=_remove_markdown_fences(raw),
            explanation="Rejected fallback-style output.",
            is_fallback=True,
        )

    fenced = list(FENCE_RE.finditer(raw))
    if fenced:
        best = _choose_fenced_block(fenced, language)
        code = best.group("code").strip()
        explanation = (raw[: best.start()] + raw[best.end() :]).strip()
        return FormattedModelOutput(code=code, explanation=_clean_explanation(explanation))

    lines = _drop_leading_prose(raw.splitlines(), language)
    code_lines, explanation_lines = _split_code_and_explanation(lines)
    code = "\n".join(code_lines).strip()
    explanation = "\n".join(explanation_lines).strip()

    if language == "python" and code:
        valid_prefix = _longest_valid_python_prefix(code)
        if valid_prefix and valid_prefix != code:
            explanation_tail = code[len(valid_prefix) :].strip()
            code = valid_prefix
            explanation = "\n\n".join(part for part in [explanation_tail, explanation] if part)

    return FormattedModelOutput(code=code.strip(), explanation=_clean_explanation(explanation))


def extract_explanation_only(text: str) -> FormattedModelOutput:
    raw = THINK_RE.sub("", text or "").strip()
    if not raw:
        return FormattedModelOutput(
            code="",
            explanation="The local model returned an empty response.",
            is_empty=True,
        )
    return FormattedModelOutput(code="", explanation=_clean_explanation(raw))


def _choose_fenced_block(matches: list[re.Match[str]], language: str) -> re.Match[str]:
    aliases = {
        "python": {"python", "py"},
        "javascript": {"javascript", "js", "jsx"},
        "typescript": {"typescript", "ts", "tsx"},
        "cpp": {"cpp", "c++", "cc", "cxx"},
        "csharp": {"csharp", "cs", "c#"},
        "java": {"java"},
        "rust": {"rust", "rs"},
    }
    wanted = aliases.get(language, {language})
    for match in matches:
        if match.group("lang").lower() in wanted:
            return match
    return matches[0]


def _drop_leading_prose(lines: list[str], language: str) -> list[str]:
    for index, line in enumerate(lines):
        if _looks_like_code_start(line, language):
            return lines[index:]
    return lines


def _looks_like_code_start(line: str, language: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if language == "python":
        return bool(
            re.match(
                r"^(def |class |import |from |@|[A-Za-z_]\w*\s*=|for |while |if |with |try:|result\s*=)",
                stripped,
            )
        )
    language_tokens = {
        "javascript": (";", "{", "}", "function ", "class ", "const ", "let ", "var ", "export ", "import "),
        "typescript": (";", "{", "}", "function ", "class ", "const ", "let ", "var ", "export ", "import ", "interface ", "type "),
        "java": (";", "{", "}", "class ", "interface ", "enum ", "public ", "private ", "protected ", "import ", "package "),
        "cpp": (";", "{", "}", "#include", "class ", "struct ", "namespace ", "template", "std::"),
        "csharp": (";", "{", "}", "class ", "interface ", "namespace ", "using ", "public ", "private ", "protected "),
        "rust": (";", "{", "}", "fn ", "pub ", "impl ", "struct ", "enum ", "trait ", "use "),
    }
    return any(token in stripped for token in language_tokens.get(language, (";", "{", "}")))


def _split_code_and_explanation(lines: list[str]) -> tuple[list[str], list[str]]:
    for index, line in enumerate(lines):
        if EXPLANATION_RE.match(line):
            return lines[:index], lines[index + 1 :]
    return lines, []


def _longest_valid_python_prefix(code: str) -> str:
    lines = code.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).strip()
        if not candidate:
            continue
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return ""


def _clean_explanation(explanation: str) -> str:
    explanation = _remove_markdown_fences(explanation)
    cleaned_lines = []
    for line in explanation.splitlines():
        if EXPLANATION_RE.match(line):
            continue
        cleaned_lines.append(INLINE_EXPLANATION_RE.sub("", line).strip("* "))
    return "\n".join(line for line in cleaned_lines if line).strip()


def _remove_markdown_fences(text: str) -> str:
    fenced = list(FENCE_RE.finditer(text))
    if fenced:
        return "\n\n".join(match.group("code").strip() for match in fenced if match.group("code").strip()).strip()
    return re.sub(r"```[A-Za-z0-9_+#.-]*", "", text).replace("```", "").strip()


def _looks_like_fallback(text: str) -> bool:
    lowered = text.lower()
    return "fallback test generated" in lowered or "assert true" in lowered
