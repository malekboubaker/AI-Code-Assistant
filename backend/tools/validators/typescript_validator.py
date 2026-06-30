from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from backend.tools.validators.base import BaseValidator, LIMITED_SYNTAX_WARNING, ValidatorResult, npm_test, output_tail, run_command


class TypeScriptValidator(BaseValidator):
    name = "typescript"

    def _validate(self, code: str, project_path: str | None = None, run_tests: bool = False) -> ValidatorResult:
        warnings: list[str] = []
        errors: list[str] = []
        syntax_valid = True

        if not shutil.which("tsc"):
            warnings.append(f"{LIMITED_SYNTAX_WARNING} TypeScript compiler (tsc) is not available.")
        else:
            with tempfile.TemporaryDirectory(prefix="ai_validate_ts_") as tmp:
                path = Path(tmp) / "temp.ts"
                path.write_text(_wrap_typescript_if_needed(code), encoding="utf-8")
                try:
                    result = run_command(
                        [
                            "tsc",
                            "--noEmit",
                            "--target",
                            "ES2020",
                            "--module",
                            "commonjs",
                            "--skipLibCheck",
                            str(path),
                        ],
                        timeout=self.timeout_seconds,
                    )
                    output = output_tail(result)
                    syntax_valid = result.returncode == 0 or not _has_syntax_error(output)
                    if result.returncode != 0 and syntax_valid:
                        warnings.append(output or "TypeScript compiler reported non-syntax diagnostics.")
                    elif not syntax_valid:
                        errors.append(output)
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


def _wrap_typescript_if_needed(code: str) -> str:
    stripped = code.strip()
    if _looks_like_class_method(stripped):
        return f"class TempValidation {{\n{stripped}\n}}\n"
    return code


def _looks_like_class_method(code: str) -> bool:
    if code.startswith(("function ", "class ", "const ", "let ", "var ", "export ", "import ", "interface ", "type ")):
        return False
    return "(" in code and ")" in code and "{" in code and "}" in code


def _has_syntax_error(output: str) -> bool:
    syntax_codes = {
        "TS1005",
        "TS1109",
        "TS1128",
        "TS1136",
        "TS1138",
        "TS1160",
        "TS1180",
        "TS1434",
        "TS1443",
    }
    return any(code in output for code in syntax_codes)
