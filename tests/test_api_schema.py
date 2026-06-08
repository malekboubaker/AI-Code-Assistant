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


def test_generate_request_accepts_chat_history():
    request = GenerateRequest(
        instruction="Now refactor the second function",
        language="python",
        chat_history=[
            {"role": "user", "content": "Explain this file."},
            {"role": "assistant", "content": "The second function is normalize_name()."},
        ],
    )

    assert len(request.chat_history) == 2
    assert request.chat_history[1].role == "assistant"
    assert "normalize_name" in request.chat_history[1].content
