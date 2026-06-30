from __future__ import annotations

import subprocess
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path


LIMITED_SYNTAX_WARNING = "Syntax validation is currently basic or unavailable for this language."
FAST_VALIDATION_TIMEOUT_SECONDS = 5
TEST_TIMEOUT_SECONDS = 60


@dataclass
class ValidatorResult:
    valid: bool
    syntax_valid: bool | None
    tests_passed: bool | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    validator: str = "unknown"
    duration_ms: int = 0
    tests_executed: bool = False
    test_framework: str | None = None
    tests_run: int = 0
    tests_failed: int = 0
    test_duration_ms: int = 0
    test_exit_code: int | None = None

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "syntax_valid": self.syntax_valid,
            "tests_passed": self.tests_passed,
            "warnings": self.warnings,
            "errors": self.errors,
            "validator": self.validator,
            "duration_ms": self.duration_ms,
            "tests_executed": self.tests_executed,
            "test_framework": self.test_framework,
            "tests_run": self.tests_run,
            "tests_failed": self.tests_failed,
            "test_duration_ms": self.test_duration_ms,
            "test_exit_code": self.test_exit_code,
        }


class BaseValidator:
    name = "base"
    timeout_seconds = FAST_VALIDATION_TIMEOUT_SECONDS

    def validate(self, code: str, project_path: str | None = None, run_tests: bool = False) -> ValidatorResult:
        started = time.perf_counter()
        result = self._validate(code, project_path=project_path, run_tests=run_tests)
        result.validator = self.name
        result.duration_ms = round((time.perf_counter() - started) * 1000)
        result.valid = (result.syntax_valid is not False) and (result.tests_passed is not False) and not result.errors
        return result

    def _validate(self, code: str, project_path: str | None = None, run_tests: bool = False) -> ValidatorResult:
        return ValidatorResult(valid=True, syntax_valid=True, warnings=[LIMITED_SYNTAX_WARNING], validator=self.name)


def run_command(command: list[str], cwd: Path | None = None, timeout: int = FAST_VALIDATION_TIMEOUT_SECONDS) -> subprocess.CompletedProcess[str]:
    executable = shutil.which(command[0]) if command else None
    if executable:
        command = [executable, *command[1:]]
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def output_tail(result: subprocess.CompletedProcess[str], limit: int = 2000) -> str:
    output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part).strip()
    return output[-limit:] if output else ""


def npm_test(project_path: str | None) -> tuple[bool | None, list[str]]:
    if not project_path:
        return None, ["No project path supplied; skipped npm test."]
    root = Path(project_path)
    if not (root / "package.json").exists():
        return None, ["package.json not found; skipped npm test."]
    try:
        result = run_command(["npm", "test", "--", "--runInBand"], cwd=root, timeout=TEST_TIMEOUT_SECONDS)
        return result.returncode == 0, [output_tail(result)]
    except FileNotFoundError:
        return None, ["npm is not available; skipped npm test."]
    except (subprocess.SubprocessError, TimeoutError) as exc:
        return False, [str(exc)]


def split_imports_and_body(code: str) -> tuple[list[str], str]:
    imports: list[str] = []
    body_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "package ", "using ", "#include", "use ")):
            imports.append(line)
        else:
            body_lines.append(line)
    return imports, "\n".join(body_lines).strip()


def looks_like_method_only(code: str) -> bool:
    stripped = code.strip()
    if not stripped:
        return False
    if any(token in stripped for token in (" class ", "interface ", " enum ", " struct ", "namespace ")):
        return False
    if stripped.startswith(("class ", "public class ", "private class ", "interface ", "struct ", "enum ", "namespace ")):
        return False
    return "(" in stripped and ")" in stripped and "{" in stripped and "}" in stripped
