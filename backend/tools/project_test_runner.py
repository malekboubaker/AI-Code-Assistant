from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

logger = logging.getLogger(__name__)

MAX_TEST_DURATION_SECONDS = int(os.getenv("MAX_TEST_DURATION_SECONDS", "60"))
MAX_OUTPUT_CHARS = 4000
MAX_MEMORY_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB, best-effort on POSIX only


@dataclass
class TestExecutionResult:
    __test__ = False  # not a pytest test class
    tests_executed: bool = False
    tests_passed: bool | None = None
    test_framework: str | None = None
    tests_run: int = 0
    tests_failed: int = 0
    test_duration_ms: int = 0
    test_exit_code: int | None = None
    output_summary: str = ""
    skip_reason: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class TestPlan:
    __test__ = False  # not a pytest test class
    framework: str
    command: list[str]
    fallback: list[str] | None = None


class ProjectTestRunner:
    """Detects and safely runs a project's test suite for the validation pipeline.

    Test execution is opt-in (driven by ``run_tests=true``); this class is only invoked
    when the caller explicitly requests it. It is intentionally structured so future
    project checks (coverage, linting, static analysis, security scanning, benchmarks)
    can be added as sibling runners that return the same structured result shape.
    """

    def __init__(self, max_duration_seconds: int = MAX_TEST_DURATION_SECONDS) -> None:
        self.max_duration_seconds = max_duration_seconds

    def run(self, language: str | None, project_path: str | None) -> TestExecutionResult:
        result = TestExecutionResult()
        root = _safe_root(project_path)
        if root is None:
            result.skip_reason = "unsafe_or_missing_project_path"
            result.warnings.append("Project path is missing or unsafe; project tests were skipped.")
            return result

        plan = _detect_plan(language, root)
        if plan is None:
            result.skip_reason = "no_test_configuration"
            result.warnings.append(f"No {language or 'project'} test configuration detected; skipped project tests.")
            return result

        result.test_framework = plan.framework
        commands = [plan.command] + ([plan.fallback] if plan.fallback else [])
        for command in commands:
            executable = shutil.which(command[0])
            if not executable:
                continue
            started = perf_counter()
            try:
                completed = subprocess.run(
                    [executable, *command[1:]],
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=self.max_duration_seconds,
                    preexec_fn=_resource_limiter(),
                )
            except subprocess.TimeoutExpired:
                result.tests_executed = True
                result.tests_passed = False
                result.test_duration_ms = _elapsed_ms(started)
                result.warnings.append("Test execution timeout")
                result.errors.append(
                    f"{plan.framework} tests exceeded the {self.max_duration_seconds}s limit and were stopped."
                )
                return result
            except (OSError, subprocess.SubprocessError) as exc:
                result.warnings.append(f"Failed to run {plan.framework}: {exc}")
                continue

            output = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
            tests_run, tests_failed = _parse_counts(plan.framework, output)
            result.tests_executed = True
            result.test_exit_code = completed.returncode
            result.tests_passed = completed.returncode == 0
            result.tests_run = tests_run
            result.tests_failed = tests_failed
            result.test_duration_ms = _elapsed_ms(started)
            result.output_summary = output[-MAX_OUTPUT_CHARS:]
            if not result.tests_passed:
                result.errors.append(
                    f"{plan.framework} reported failing tests (exit code {completed.returncode})."
                )
            logger.info(
                "Project tests executed: framework=%s passed=%s run=%s failed=%s exit=%s duration_ms=%s",
                plan.framework,
                result.tests_passed,
                result.tests_run,
                result.tests_failed,
                result.test_exit_code,
                result.test_duration_ms,
            )
            return result

        result.skip_reason = "runner_unavailable"
        result.warnings.append(f"The {plan.framework} runner was not found on PATH; skipped project tests.")
        return result


def _detect_plan(language: str | None, root: Path) -> TestPlan | None:
    lang = (language or "").lower()
    if lang == "python":
        markers = ("pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini", "conftest.py")
        has_config = any((root / marker).exists() for marker in markers)
        has_tests_dir = (root / "tests").is_dir() or (root / "test").is_dir()
        if has_config or has_tests_dir:
            return TestPlan("pytest", ["pytest", "-q"], fallback=["python", "-m", "pytest", "-q"])
        return None
    if lang in {"javascript", "typescript"}:
        if (root / "package.json").exists():
            return TestPlan("npm", ["npm", "test", "--silent"])
        return None
    if lang == "java":
        if (root / "pom.xml").exists():
            return TestPlan("maven", ["mvn", "-q", "test"])
        if (root / "build.gradle").exists() or (root / "build.gradle.kts").exists():
            return TestPlan("gradle", ["gradle", "test", "-q"])
        return None
    if lang == "csharp":
        if any(root.glob("*.sln")) or any(root.glob("*.csproj")):
            return TestPlan("dotnet", ["dotnet", "test"])
        return None
    if lang == "rust":
        if (root / "Cargo.toml").exists():
            return TestPlan("cargo", ["cargo", "test"])
        return None
    return None


def _safe_root(project_path: str | None) -> Path | None:
    """Resolve and validate the workspace root; reject unsafe or missing paths."""
    if not project_path:
        return None
    try:
        root = Path(project_path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if not root.exists() or not root.is_dir():
        return None
    if root.parent == root:  # filesystem / drive root
        return None
    return root


def _resource_limiter():
    """Best-effort memory cap for the test subprocess (POSIX only)."""
    if os.name != "posix":
        return None
    try:
        import resource
    except ImportError:
        return None

    def _limit() -> None:
        try:
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            ceiling = MAX_MEMORY_BYTES if hard == resource.RLIM_INFINITY else min(MAX_MEMORY_BYTES, hard)
            resource.setrlimit(resource.RLIMIT_AS, (ceiling, hard))
        except (ValueError, OSError):
            pass

    return _limit


def _parse_counts(framework: str, text: str) -> tuple[int, int]:
    if framework == "pytest":
        passed = _first_int(re.search(r"(\d+) passed", text))
        failed = _first_int(re.search(r"(\d+) failed", text))
        errors = _first_int(re.search(r"(\d+) error", text))
        skipped = _first_int(re.search(r"(\d+) skipped", text))
        return passed + failed + errors + skipped, failed + errors
    if framework == "npm":
        jest = re.search(r"Tests:\s*(?:(\d+) failed[ ,]*)?(?:(\d+) passed[ ,]*)?(?:(\d+) total)", text)
        if jest and jest.group(3):
            return int(jest.group(3)), int(jest.group(1) or 0)
        passing = _first_int(re.search(r"(\d+) passing", text))
        failing = _first_int(re.search(r"(\d+) failing", text))
        if passing or failing:
            return passing + failing, failing
        return 0, 0
    if framework == "maven":
        runs = re.findall(r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+)", text)
        if runs:
            run, failures, errors = runs[-1]
            return int(run), int(failures) + int(errors)
        return 0, 0
    if framework == "dotnet":
        match = re.search(r"Failed:\s*(\d+),\s*Passed:\s*(\d+),\s*Skipped:\s*(\d+),\s*Total:\s*(\d+)", text)
        if match:
            return int(match.group(4)), int(match.group(1))
        return 0, 0
    if framework == "cargo":
        results = re.findall(r"test result:[^\n]*?(\d+) passed;\s*(\d+) failed", text)
        if results:
            passed = sum(int(p) for p, _ in results)
            failed = sum(int(f) for _, f in results)
            return passed + failed, failed
        return 0, 0
    return 0, 0


def _first_int(match: re.Match[str] | None) -> int:
    return int(match.group(1)) if match else 0


def _elapsed_ms(start: float) -> int:
    return round((perf_counter() - start) * 1000)
