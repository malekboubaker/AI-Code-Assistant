from __future__ import annotations

import ast
import subprocess
import tempfile
from pathlib import Path

LIMITED_SYNTAX_WARNING = "Syntax validation is currently basic or unavailable for this language."


def check_syntax(code: str, language: str) -> tuple[bool, list[str]]:
    if not code.strip():
        return False, ["Empty output."]
    if language == "python":
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as exc:
            return False, [f"Python syntax error: {exc}"]
    if language == "javascript":
        return _node_check(code, language)
    if language == "typescript":
        return True, [LIMITED_SYNTAX_WARNING]
    if language in {"java", "cpp", "csharp", "rust"}:
        return True, [LIMITED_SYNTAX_WARNING]
    return True, [LIMITED_SYNTAX_WARNING]


def _node_check(code: str, language: str) -> tuple[bool, list[str]]:
    suffix = ".ts" if language == "typescript" else ".js"
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as handle:
        handle.write(code)
        temp_path = handle.name
    try:
        result = subprocess.run(
            ["node", "--check", temp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, []
        return False, [result.stderr.strip() or result.stdout.strip()]
    except (FileNotFoundError, subprocess.SubprocessError):
        return True, [LIMITED_SYNTAX_WARNING]
    finally:
        Path(temp_path).unlink(missing_ok=True)
