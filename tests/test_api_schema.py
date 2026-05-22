from backend.api.schemas import GenerateRequest


def test_generate_request_accepts_supported_languages():
    for language in ["python", "javascript", "typescript", "java", "cpp", "csharp", "rust"]:
        request = GenerateRequest(instruction="explain", language=language)
        assert request.language == language
