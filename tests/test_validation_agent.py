from backend.agent.validation_agent import ValidationAgent
from backend.tools.syntax_checker import LIMITED_SYNTAX_WARNING


def test_validation_warns_for_basic_or_unavailable_languages(monkeypatch):
    monkeypatch.setattr("backend.tools.validators.java_validator.shutil.which", lambda name: None)
    result = ValidationAgent().validate("public class Demo {}", "java")

    assert result.valid is True
    assert result.syntax_valid is True
    assert any(LIMITED_SYNTAX_WARNING in warning for warning in result.warnings)
    assert result.validator == "java"


def test_validation_warns_for_typescript_basic_validation(monkeypatch):
    monkeypatch.setattr("backend.tools.validators.typescript_validator.shutil.which", lambda name: None)
    result = ValidationAgent().validate("const value: number = 1;", "typescript")

    assert result.valid is True
    assert result.syntax_valid is True
    assert any(LIMITED_SYNTAX_WARNING in warning for warning in result.warnings)
    assert result.validator == "typescript"
