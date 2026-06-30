import subprocess
from types import SimpleNamespace

import pytest

from backend.tools import project_test_runner as ptr
from backend.tools.project_test_runner import ProjectTestRunner, TestExecutionResult


def test_detects_pytest_from_tests_folder(tmp_path):
    (tmp_path / "tests").mkdir()
    plan = ptr._detect_plan("python", tmp_path)
    assert plan is not None
    assert plan.framework == "pytest"
    assert plan.command[0] == "pytest"


def test_detects_pyproject_and_other_frameworks(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    assert ptr._detect_plan("python", tmp_path).framework == "pytest"

    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    assert ptr._detect_plan("javascript", tmp_path).command == ["npm", "test", "--silent"]
    assert ptr._detect_plan("typescript", tmp_path).framework == "npm"

    (tmp_path / "Cargo.toml").write_text("[package]\n", encoding="utf-8")
    assert ptr._detect_plan("rust", tmp_path).command == ["cargo", "test"]

    (tmp_path / "pom.xml").write_text("<project/>", encoding="utf-8")
    assert ptr._detect_plan("java", tmp_path).framework == "maven"


def test_detects_gradle_and_dotnet(tmp_path):
    (tmp_path / "build.gradle").write_text("", encoding="utf-8")
    assert ptr._detect_plan("java", tmp_path).framework == "gradle"

    (tmp_path / "App.csproj").write_text("<Project/>", encoding="utf-8")
    assert ptr._detect_plan("csharp", tmp_path).framework == "dotnet"


def test_no_configuration_returns_none(tmp_path):
    assert ptr._detect_plan("python", tmp_path) is None
    assert ptr._detect_plan("javascript", tmp_path) is None


def test_parse_counts_per_framework():
    assert ptr._parse_counts("pytest", "===== 41 passed, 1 failed in 0.5s =====") == (42, 1)
    assert ptr._parse_counts("npm", "Tests:       1 failed, 41 passed, 42 total") == (42, 1)
    assert ptr._parse_counts("maven", "Tests run: 42, Failures: 1, Errors: 0, Skipped: 0") == (42, 1)
    assert ptr._parse_counts("dotnet", "Failed:     1, Passed:     5, Skipped:     0, Total:     6") == (6, 1)
    assert ptr._parse_counts("cargo", "test result: ok. 5 passed; 0 failed; 0 ignored") == (5, 0)


def test_safe_root_rejects_missing_and_empty():
    assert ptr._safe_root(None) is None
    assert ptr._safe_root("") is None
    assert ptr._safe_root("/path/that/should/not/exist/xyz123") is None


def test_run_skips_without_configuration(tmp_path):
    result = ProjectTestRunner().run("python", str(tmp_path))
    assert result.tests_executed is False
    assert result.skip_reason == "no_test_configuration"


def test_run_executes_and_parses_results(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()

    monkeypatch.setattr(ptr.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **kwargs):
        return SimpleNamespace(returncode=0, stdout="===== 3 passed in 0.1s =====", stderr="")

    monkeypatch.setattr(ptr.subprocess, "run", fake_run)

    result = ProjectTestRunner().run("python", str(tmp_path))

    assert result.tests_executed is True
    assert result.tests_passed is True
    assert result.test_framework == "pytest"
    assert result.tests_run == 3
    assert result.tests_failed == 0
    assert result.test_exit_code == 0


def test_run_reports_failures(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()
    monkeypatch.setattr(ptr.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        ptr.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(returncode=1, stdout="1 passed, 2 failed", stderr=""),
    )

    result = ProjectTestRunner().run("python", str(tmp_path))

    assert result.tests_executed is True
    assert result.tests_passed is False
    assert result.tests_run == 3
    assert result.tests_failed == 2
    assert result.errors


def test_run_handles_timeout(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()
    monkeypatch.setattr(ptr.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(cmd=command, timeout=60)

    monkeypatch.setattr(ptr.subprocess, "run", fake_run)

    result = ProjectTestRunner().run("python", str(tmp_path))

    assert result.tests_executed is True
    assert result.tests_passed is False
    assert any("timeout" in warning.lower() for warning in result.warnings)


def test_run_reports_runner_unavailable(tmp_path, monkeypatch):
    (tmp_path / "tests").mkdir()
    monkeypatch.setattr(ptr.shutil, "which", lambda name: None)

    result = ProjectTestRunner().run("python", str(tmp_path))

    assert result.tests_executed is False
    assert result.skip_reason == "runner_unavailable"
