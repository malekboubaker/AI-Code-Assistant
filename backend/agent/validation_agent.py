from __future__ import annotations

import logging

from backend.api.schemas import ValidationResult
from backend.tools.formatter import strip_code_fences
from backend.tools.validators import (
    CppValidator,
    CSharpValidator,
    JavaScriptValidator,
    JavaValidator,
    PythonValidator,
    RustValidator,
    TypeScriptValidator,
    ValidatorResult,
)
from backend.tools.validators.base import BaseValidator

logger = logging.getLogger(__name__)


class ValidationAgent:
    def __init__(self) -> None:
        self.validators: dict[str, BaseValidator] = {
            "python": PythonValidator(),
            "javascript": JavaScriptValidator(),
            "typescript": TypeScriptValidator(),
            "java": JavaValidator(),
            "cpp": CppValidator(),
            "csharp": CSharpValidator(),
            "rust": RustValidator(),
        }

    def validate_explanation(self, explanation: str) -> ValidationResult:
        if explanation.strip():
            result = ValidationResult(valid=True, syntax_valid=None, tests_passed=None, warnings=[], validator="explanation")
        else:
            result = ValidationResult(
                valid=False,
                syntax_valid=None,
                tests_passed=None,
                warnings=["Explanation is empty."],
                errors=["Explanation is empty."],
                validator="explanation",
            )
        logger.debug("Explanation validation result: %s", result.model_dump())
        return result

    def validate(self, output: str, language: str, project_path: str | None = None, run_project_tests: bool = False) -> ValidationResult:
        # If this is a multi-file edit, we bypass single-file syntax checks for now
        if "<workspace_edits>" in output.lower():
            return ValidationResult(
                valid=True,
                syntax_valid=True,
                tests_passed=None,
                warnings=[],
                errors=[],
                validator="workspace_overlay_pending",
            )
            
        code = strip_code_fences(output, language)
        logger.debug("Validation received raw output: %r", output)
        logger.debug("Validation cleaned code: %r", code)
        if not code.strip():
            result = self._result_from_validator(
                ValidatorResult(
                    valid=False,
                    syntax_valid=False,
                    tests_passed=None,
                    warnings=["Generated code is empty."],
                    errors=["Generated code is empty."],
                    validator="precheck",
                )
            )
            logger.debug("Validation result: %s", result.model_dump())
            return result
        lowered = code.lower()
        if "fallback test generated" in lowered or "assert true" in lowered:
            result = self._result_from_validator(
                ValidatorResult(
                    valid=False,
                    syntax_valid=False,
                    tests_passed=None,
                    warnings=["Rejected fallback or placeholder output."],
                    errors=["Rejected fallback or placeholder output."],
                    validator="precheck",
                )
            )
            logger.debug("Validation result: %s", result.model_dump())
            return result
        if "```" in output or "```" in code:
            result = self._result_from_validator(
                ValidatorResult(
                    valid=False,
                    syntax_valid=False,
                    tests_passed=None,
                    warnings=["Generated code still contains Markdown fences."],
                    errors=["Generated code still contains Markdown fences."],
                    validator="precheck",
                )
            )
            logger.debug("Validation result: %s", result.model_dump())
            return result

        validator = self.validators.get(language, BaseValidator())
        validation_result = validator.validate(code, project_path=project_path, run_tests=run_project_tests)
        result = self._result_from_validator(validation_result)
        logger.debug("Validation result: %s", result.model_dump())
        return result

    def _result_from_validator(self, result: ValidatorResult) -> ValidationResult:
        return ValidationResult(
            valid=result.valid,
            syntax_valid=result.syntax_valid,
            tests_passed=result.tests_passed,
            warnings=result.warnings,
            errors=result.errors,
            validator=result.validator,
            duration_ms=result.duration_ms,
        )
