from __future__ import annotations

import logging

from backend.api.schemas import ValidationResult
from backend.tools.formatter import strip_code_fences
from backend.tools.syntax_checker import check_syntax
from backend.tools.test_runner import run_tests

logger = logging.getLogger(__name__)


class ValidationAgent:
    def validate_explanation(self, explanation: str) -> ValidationResult:
        if explanation.strip():
            result = ValidationResult(valid=True, syntax_valid=None, tests_passed=None, warnings=[])
        else:
            result = ValidationResult(
                valid=False,
                syntax_valid=None,
                tests_passed=None,
                warnings=["Explanation is empty."],
            )
        logger.debug("Explanation validation result: %s", result.model_dump())
        return result

    def validate(self, output: str, language: str, project_path: str | None = None, run_project_tests: bool = False) -> ValidationResult:
        code = strip_code_fences(output)
        logger.debug("Validation received raw output: %r", output)
        logger.debug("Validation cleaned code: %r", code)
        warnings: list[str] = []
        if not code.strip():
            result = ValidationResult(valid=False, syntax_valid=False, tests_passed=None, warnings=["Generated code is empty."])
            logger.debug("Validation result: %s", result.model_dump())
            return result
        lowered = code.lower()
        if "fallback test generated" in lowered or "assert true" in lowered:
            result = ValidationResult(
                valid=False,
                syntax_valid=False,
                tests_passed=None,
                warnings=["Rejected fallback or placeholder output."],
            )
            logger.debug("Validation result: %s", result.model_dump())
            return result
        if "```" in output or "```" in code:
            result = ValidationResult(
                valid=False,
                syntax_valid=False,
                tests_passed=None,
                warnings=["Generated code still contains Markdown fences."],
            )
            logger.debug("Validation result: %s", result.model_dump())
            return result
        syntax_valid, warnings = check_syntax(code, language)
        tests_passed = None
        if run_project_tests:
            tests_passed, test_warnings = run_tests(project_path)
            warnings.extend(test_warnings)
        valid = syntax_valid and (tests_passed is not False)
        result = ValidationResult(valid=valid, syntax_valid=syntax_valid, tests_passed=tests_passed, warnings=warnings)
        logger.debug("Validation result: %s", result.model_dump())
        return result
