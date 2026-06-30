from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from backend.tools.validators.base import BaseValidator, LIMITED_SYNTAX_WARNING, ValidatorResult, output_tail, run_command


class CppValidator(BaseValidator):
    name = "cpp"

    def _validate(self, code: str, project_path: str | None = None, run_tests: bool = False) -> ValidatorResult:
        warnings: list[str] = []
        errors: list[str] = []
        syntax_valid = True
        compiler = "g++" if shutil.which("g++") else "clang++" if shutil.which("clang++") else None

        if compiler is None:
            warnings.append(f"{LIMITED_SYNTAX_WARNING} g++ or clang++ is not available.")
        else:
            with tempfile.TemporaryDirectory(prefix="ai_validate_cpp_") as tmp:
                path = Path(tmp) / "temp.cpp"
                path.write_text(_prepare_cpp_code(code), encoding="utf-8")
                try:
                    result = run_command([compiler, "-std=c++17", "-fsyntax-only", str(path)], timeout=self.timeout_seconds)
                    output = output_tail(result)
                    syntax_valid = result.returncode == 0 or _looks_like_missing_context(output)
                    if result.returncode != 0 and syntax_valid:
                        warnings.append(output or "C++ compiler reported non-syntax diagnostics.")
                    elif not syntax_valid:
                        errors.append(output)
                except (subprocess.SubprocessError, TimeoutError) as exc:
                    syntax_valid = False
                    errors.append(str(exc))

        if run_tests:
            warnings.append("C++ project tests are not run by fast validation; use the project build tooling separately.")

        return ValidatorResult(
            valid=syntax_valid,
            syntax_valid=syntax_valid,
            tests_passed=None,
            warnings=warnings,
            errors=errors,
            validator=self.name,
        )


def _prepare_cpp_code(code: str) -> str:
    stripped = code.strip()
    if stripped.startswith("#include") or "int main(" in stripped or "class " in stripped or "struct " in stripped:
        return code
    if _looks_like_function_or_method(stripped):
        return "#include <bits/stdc++.h>\n" + code
    return "#include <bits/stdc++.h>\nint main() {\n" + code + "\nreturn 0;\n}\n"


def _looks_like_function_or_method(code: str) -> bool:
    if code.startswith(("if ", "for ", "while ", "switch ")):
        return False
    first_line = code.splitlines()[0].strip()
    return "(" in first_line and ")" in first_line and "{" in first_line and not first_line.startswith(("if", "for", "while", "switch"))


def _looks_like_missing_context(output: str) -> bool:
    missing_context_markers = [
        "was not declared in this scope",
        "not declared in this scope",
        "unknown type name",
        "use of undeclared identifier",
        "cannot find",
        "does not name a type",
    ]
    return any(marker in output for marker in missing_context_markers)
