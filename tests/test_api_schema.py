from backend.api.schemas import GenerateRequest


def test_generate_request_accepts_supported_languages():
    for language in ["python", "javascript", "typescript", "java", "cpp", "csharp", "rust"]:
        request = GenerateRequest(instruction="explain", language=language)
        assert request.language == language


def test_generate_request_accepts_selection_metadata():
    request = GenerateRequest(
        instruction="Explain this code",
        code="def run():\n    return True\n",
        language="python",
        has_selection=True,
        surrounding_context="def before(): pass\n\ndef run():\n    return True\n",
    )

    assert request.has_selection is True
    assert "before" in request.surrounding_context
