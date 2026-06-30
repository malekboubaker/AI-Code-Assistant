import subprocess
from pathlib import Path

from backend.tools.validators.base import LIMITED_SYNTAX_WARNING
from backend.tools.validators.cpp_validator import CppValidator
from backend.tools.validators.csharp_validator import CSharpValidator
from backend.tools.validators.java_validator import JavaValidator
from backend.tools.validators.javascript_validator import JavaScriptValidator
from backend.tools.validators.python_validator import PythonValidator
from backend.tools.validators.rust_validator import RustValidator
from backend.tools.validators.typescript_validator import TypeScriptValidator


def completed(returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


def test_python_ast_validation_still_works():
    valid = PythonValidator().validate("def add(a, b):\n    return a + b\n")
    invalid = PythonValidator().validate("def add(:\n    pass\n")

    assert valid.syntax_valid is True
    assert valid.errors == []
    assert invalid.syntax_valid is False
    assert invalid.errors


def test_valid_javascript(monkeypatch):
    monkeypatch.setattr("backend.tools.validators.javascript_validator.shutil.which", lambda name: "node")
    monkeypatch.setattr("backend.tools.validators.javascript_validator.run_command", lambda command, timeout=5, cwd=None: completed())

    result = JavaScriptValidator().validate("function add(a, b) { return a + b; }")

    assert result.valid is True
    assert result.syntax_valid is True
    assert result.validator == "javascript"


def test_invalid_javascript(monkeypatch):
    monkeypatch.setattr("backend.tools.validators.javascript_validator.shutil.which", lambda name: "node")
    monkeypatch.setattr(
        "backend.tools.validators.javascript_validator.run_command",
        lambda command, timeout=5, cwd=None: completed(1, "SyntaxError: Unexpected token"),
    )

    result = JavaScriptValidator().validate("function {")

    assert result.valid is False
    assert result.syntax_valid is False
    assert "SyntaxError" in result.errors[0]


def test_typescript_validation_when_tsc_missing(monkeypatch):
    monkeypatch.setattr("backend.tools.validators.typescript_validator.shutil.which", lambda name: None)

    result = TypeScriptValidator().validate("const value: number = 1;")

    assert result.valid is True
    assert result.syntax_valid is True
    assert any("tsc" in warning for warning in result.warnings)


def test_java_method_wrapping(monkeypatch):
    captured = {}
    monkeypatch.setattr("backend.tools.validators.java_validator.shutil.which", lambda name: "javac")

    def fake_run(command, timeout=5, cwd=None):
        source_path = Path(command[1])
        captured["source"] = source_path.read_text(encoding="utf-8")
        return completed()

    monkeypatch.setattr("backend.tools.validators.java_validator.run_command", fake_run)

    result = JavaValidator().validate("public int add(int a, int b) { return a + b; }")

    assert result.valid is True
    assert "public class TempValidation" in captured["source"]
    assert "public int add" in captured["source"]


def test_cpp_validation_warning_if_compiler_missing(monkeypatch):
    monkeypatch.setattr("backend.tools.validators.cpp_validator.shutil.which", lambda name: None)

    result = CppValidator().validate("int main() { return 0; }")

    assert result.valid is True
    assert result.syntax_valid is True
    assert LIMITED_SYNTAX_WARNING in result.warnings[0]


def test_csharp_validation_warning_if_dotnet_missing(monkeypatch):
    monkeypatch.setattr("backend.tools.validators.csharp_validator.shutil.which", lambda name: None)

    result = CSharpValidator().validate("public int Add(int a, int b) { return a + b; }")

    assert result.valid is True
    assert result.syntax_valid is True
    assert LIMITED_SYNTAX_WARNING in result.warnings[0]


def test_rust_validation_warning_if_rustc_missing(monkeypatch):
    monkeypatch.setattr("backend.tools.validators.rust_validator.shutil.which", lambda name: None)

    result = RustValidator().validate("fn add(a: i32, b: i32) -> i32 { a + b }")

    assert result.valid is True
    assert result.syntax_valid is True
    assert LIMITED_SYNTAX_WARNING in result.warnings[0]
