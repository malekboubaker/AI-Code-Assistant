from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from backend.tools.validators.base import BaseValidator, LIMITED_SYNTAX_WARNING, TEST_TIMEOUT_SECONDS, ValidatorResult, output_tail, run_command


class RustValidator(BaseValidator):
    name = "rust"

    def _validate(self, code: str, project_path: str | None = None, run_tests: bool = False) -> ValidatorResult:
        warnings: list[str] = []
        errors: list[str] = []
        syntax_valid = True

        if not shutil.which("rustc"):
            warnings.append(f"{LIMITED_SYNTAX_WARNING} rustc is not available.")
        else:
            with tempfile.TemporaryDirectory(prefix="ai_validate_rust_") as tmp:
                path = Path(tmp) / "temp.rs"
                path.write_text(_prepare_rust_code(code), encoding="utf-8")
                try:
                    result = run_command(["rustc", "--emit=metadata", str(path)], timeout=self.timeout_seconds)
                    syntax_valid = result.returncode == 0
                    if not syntax_valid:
                        errors.append(output_tail(result))
                except (subprocess.SubprocessError, TimeoutError) as exc:
                    syntax_valid = False
                    errors.append(str(exc))

        tests_passed = None
        if run_tests:
            tests_passed, test_warnings = _cargo_check_or_test(project_path)
            warnings.extend(test_warnings)

        return ValidatorResult(
            valid=syntax_valid and (tests_passed is not False),
            syntax_valid=syntax_valid,
            tests_passed=tests_passed,
            warnings=warnings,
            errors=errors,
            validator=self.name,
        )


def _prepare_rust_code(code: str) -> str:
    stripped = code.strip()
    if "fn main" in stripped or stripped.startswith(("use ", "pub ", "fn ", "struct ", "enum ", "impl ", "trait ")):
        return code
    return "fn main() {\n" + code + "\n}\n"


def _cargo_check_or_test(project_path: str | None) -> tuple[bool | None, list[str]]:
    if not project_path:
        return None, ["No project path supplied; skipped cargo check/test."]
    root = Path(project_path)
    if not (root / "Cargo.toml").exists():
        return None, ["Cargo.toml not found; skipped cargo check/test."]
    if not shutil.which("cargo"):
        return None, ["cargo is not available; skipped cargo check/test."]
    command = ["cargo", "test"] if _has_rust_tests(root) else ["cargo", "check"]
    try:
        result = run_command(command, cwd=root, timeout=TEST_TIMEOUT_SECONDS)
        return result.returncode == 0, [output_tail(result)]
    except (subprocess.SubprocessError, TimeoutError) as exc:
        return False, [str(exc)]


def _has_rust_tests(root: Path) -> bool:
    tests_dir = root / "tests"
    if tests_dir.exists():
        return True
    return any("#[test]" in path.read_text(encoding="utf-8", errors="ignore") for path in root.rglob("*.rs"))
