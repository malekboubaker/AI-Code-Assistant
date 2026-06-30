from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from backend.tools.validators.base import BaseValidator, LIMITED_SYNTAX_WARNING, ValidatorResult, looks_like_method_only, output_tail, run_command, split_imports_and_body


class CSharpValidator(BaseValidator):
    name = "csharp"

    def _validate(self, code: str, project_path: str | None = None, run_tests: bool = False) -> ValidatorResult:
        warnings: list[str] = []
        errors: list[str] = []
        syntax_valid = True
        checker = "csc" if shutil.which("csc") else "dotnet" if shutil.which("dotnet") else None

        if checker is None:
            warnings.append(f"{LIMITED_SYNTAX_WARNING} dotnet or csc is not available.")
        else:
            with tempfile.TemporaryDirectory(prefix="ai_validate_csharp_") as tmp:
                path = Path(tmp) / "TempValidation.cs"
                path.write_text(_prepare_csharp_code(code), encoding="utf-8")
                try:
                    command = ["csc", "-nologo", "-target:library", str(path)] if checker == "csc" else ["dotnet", "build", _create_dotnet_project(Path(tmp), path)]
                    result = run_command(command, cwd=Path(tmp), timeout=self.timeout_seconds)
                    syntax_valid = result.returncode == 0
                    if not syntax_valid:
                        output = output_tail(result)
                        if _looks_like_tool_environment_failure(output):
                            syntax_valid = None
                            warnings.append(f"{LIMITED_SYNTAX_WARNING} C# checker could not run in this environment: {output}")
                        else:
                            errors.append(output)
                except (subprocess.SubprocessError, TimeoutError) as exc:
                    syntax_valid = False
                    errors.append(str(exc))

        if run_tests:
            warnings.append("C# project tests are not run by fast validation; use the project build tooling separately.")

        return ValidatorResult(
            valid=syntax_valid,
            syntax_valid=syntax_valid,
            tests_passed=None,
            warnings=warnings,
            errors=errors,
            validator=self.name,
        )


def _prepare_csharp_code(code: str) -> str:
    stripped = code.strip()
    if " class " in f" {stripped}" or stripped.startswith(("namespace ", "using ")):
        return code
    if looks_like_method_only(code):
        imports, body = split_imports_and_body(code)
        return "\n".join(imports + ["public class TempValidation {", body, "}"])
    return "public class TempValidation {\npublic void Validate() {\n" + code + "\n}\n}\n"


def _create_dotnet_project(tmp: Path, source_path: Path) -> str:
    project_path = tmp / "TempValidation.csproj"
    project_path.write_text(
        '<Project Sdk="Microsoft.NET.Sdk">\n'
        '  <PropertyGroup>\n'
        '    <TargetFramework>net8.0</TargetFramework>\n'
        '    <ImplicitUsings>enable</ImplicitUsings>\n'
        '    <Nullable>enable</Nullable>\n'
        '  </PropertyGroup>\n'
        '</Project>\n',
        encoding="utf-8",
    )
    return str(project_path)


def _looks_like_tool_environment_failure(output: str) -> bool:
    lowered = output.lower()
    markers = [
        "unauthorizedaccessexception",
        "permission denied",
        "access is denied",
        "failed to create",
        "firsttimeuseconfigurer",
    ]
    return any(marker in lowered for marker in markers)
