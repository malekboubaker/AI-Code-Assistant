from backend.agent.validation_agent import ValidationAgent
from backend.tools.syntax_checker import LIMITED_SYNTAX_WARNING


def test_validation_warns_for_basic_or_unavailable_languages():
    result = ValidationAgent().validate("public class Demo {}", "java")

    assert result.valid is True
    assert result.syntax_valid is True
    assert LIMITED_SYNTAX_WARNING in result.warnings


def test_validation_warns_for_typescript_basic_validation():
    result = ValidationAgent().validate("const value: number = 1;", "typescript")

    assert result.valid is True
    assert result.syntax_valid is True
    assert LIMITED_SYNTAX_WARNING in result.warnings
