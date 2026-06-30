from __future__ import annotations

import ast

from backend.tools.test_runner import run_tests as run_python_tests
from backend.tools.validators.base import BaseValidator, ValidatorResult


class PythonValidator(BaseValidator):
    name = "python"

    def _validate(self, code: str, project_path: str | None = None, run_tests: bool = False) -> ValidatorResult:
        errors: list[str] = []
        warnings: list[str] = []
        try:
            ast.parse(code)
            syntax_valid = True
        except SyntaxError as exc:
            syntax_valid = False
            errors.append(f"Python syntax error: {exc}")

        tests_passed = None
        if run_tests:
            tests_passed, test_warnings = run_python_tests(project_path)
            warnings.extend(test_warnings)

        return ValidatorResult(
            valid=syntax_valid and (tests_passed is not False),
            syntax_valid=syntax_valid,
            tests_passed=tests_passed,
            warnings=warnings,
            errors=errors,
            validator=self.name,
        )
