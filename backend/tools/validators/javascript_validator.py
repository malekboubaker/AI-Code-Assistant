from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from backend.tools.validators.base import BaseValidator, LIMITED_SYNTAX_WARNING, ValidatorResult, npm_test, output_tail, run_command


class JavaScriptValidator(BaseValidator):
    name = "javascript"

    def _validate(self, code: str, project_path: str | None = None, run_tests: bool = False) -> ValidatorResult:
        warnings: list[str] = []
        errors: list[str] = []
        syntax_valid = True

        if not shutil.which("node"):
            warnings.append(f"{LIMITED_SYNTAX_WARNING} Node.js is not available.")
        else:
            with tempfile.TemporaryDirectory(prefix="ai_validate_js_") as tmp:
                path = Path(tmp) / "temp.js"
                path.write_text(_wrap_javascript_if_needed(code), encoding="utf-8")
                try:
                    result = run_command(["node", "--check", str(path)], timeout=self.timeout_seconds)
                    syntax_valid = result.returncode == 0
                    if not syntax_valid:
                        errors.append(output_tail(result))
                except (subprocess.SubprocessError, TimeoutError) as exc:
                    syntax_valid = False
                    errors.append(str(exc))

        tests_passed = None
        if run_tests:
            tests_passed, test_warnings = npm_test(project_path)
            warnings.extend(test_warnings)

        return ValidatorResult(
            valid=syntax_valid and (tests_passed is not False),
            syntax_valid=syntax_valid,
            tests_passed=tests_passed,
            warnings=warnings,
            errors=errors,
            validator=self.name,
        )


def _wrap_javascript_if_needed(code: str) -> str:
    stripped = code.strip()
    if _looks_like_class_method(stripped):
        return f"class TempValidation {{\n{stripped}\n}}\n"
    return code


def _looks_like_class_method(code: str) -> bool:
    if code.startswith(("function ", "class ", "const ", "let ", "var ", "export ", "import ")):
        return False
    return "(" in code and ")" in code and "{" in code and "}" in code
