from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from backend.tools.validators.base import BaseValidator, LIMITED_SYNTAX_WARNING, ValidatorResult, looks_like_method_only, output_tail, run_command, split_imports_and_body


class JavaValidator(BaseValidator):
    name = "java"

    def _validate(self, code: str, project_path: str | None = None, run_tests: bool = False) -> ValidatorResult:
        warnings: list[str] = []
        errors: list[str] = []
        syntax_valid = True

        if not shutil.which("javac"):
            warnings.append(f"{LIMITED_SYNTAX_WARNING} javac is not available.")
        else:
            prepared_code, class_name = _prepare_java_code(code)
            with tempfile.TemporaryDirectory(prefix="ai_validate_java_") as tmp:
                path = Path(tmp) / f"{class_name}.java"
                path.write_text(prepared_code, encoding="utf-8")
                try:
                    result = run_command(["javac", str(path)], timeout=self.timeout_seconds)
                    syntax_valid = result.returncode == 0
                    if not syntax_valid:
                        errors.append(output_tail(result))
                except (subprocess.SubprocessError, TimeoutError) as exc:
                    syntax_valid = False
                    errors.append(str(exc))

        if run_tests:
            warnings.append("Java project tests are not run by fast validation; use the project build tooling separately.")

        return ValidatorResult(
            valid=syntax_valid,
            syntax_valid=syntax_valid,
            tests_passed=None,
            warnings=warnings,
            errors=errors,
            validator=self.name,
        )


def _prepare_java_code(code: str) -> tuple[str, str]:
    class_match = re.search(r"\bpublic\s+class\s+([A-Za-z_]\w*)", code)
    if class_match:
        return code, class_match.group(1)
    class_match = re.search(r"\b(?:class|interface|enum|record)\s+([A-Za-z_]\w*)", code)
    if class_match:
        return code, "TempValidation"
    if looks_like_method_only(code):
        imports, body = split_imports_and_body(code)
        prepared = "\n".join(imports + ["public class TempValidation {", body, "}"])
        return prepared, "TempValidation"
    return f"public class TempValidation {{\n{code}\n}}\n", "TempValidation"
