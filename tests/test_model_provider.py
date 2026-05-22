import json

import backend.model.pretrained_qwen as pretrained_qwen
from backend.model.base import GenerationOptions
from backend.model.generation_config import default_generation_options
from backend.model.pretrained_qwen import PretrainedQwenProvider


def test_test_gen_empty_ollama_response_does_not_create_fake_fallback():
    provider = PretrainedQwenProvider()
    calls = []

    def empty_response(prompt, options):
        calls.append(prompt)
        return ""

    provider._generate_once = empty_response

    assert provider.generate("[TASK: test_gen]\nReturn pytest code.") == ""
    assert len(calls) == 2
    assert calls[0].startswith("/no_think\nReturn only the final answer.")
    assert "previous Ollama response field was empty" in calls[1]
    assert calls[1].startswith("/no_think\nReturn only the final answer.")


def test_code_task_retries_once_on_empty_ollama_response():
    provider = PretrainedQwenProvider()
    calls = []
    option_values = []

    def empty_then_code(prompt, options):
        calls.append(prompt)
        option_values.append(options.max_tokens)
        if len(calls) == 1:
            return ""
        return "result = [item * 2 for item in items]"

    provider._generate_once = empty_then_code

    assert (
        provider.generate(
            "[TASK: perf_opt]\nUser instruction:\nOptimize this code.\n\nCurrent code:\nresult = []"
        )
        == "result = [item * 2 for item in items]"
    )
    assert len(calls) == 2
    assert calls[0].startswith("/no_think\nReturn only the final answer.")
    assert "Return ONLY valid code" in calls[1]
    assert option_values[1] >= 192


def test_ollama_request_keeps_model_warm_and_uses_options(monkeypatch):
    provider = PretrainedQwenProvider(model="qwen-test")
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"response": "ok"}'

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(pretrained_qwen.urllib.request, "urlopen", fake_urlopen)

    assert provider._generate_once("prompt", GenerationOptions(max_tokens=64, temperature=0.1)) == "ok"
    assert captured["payload"]["keep_alive"] == "30m"
    assert captured["payload"]["think"] is False
    assert "stop" not in captured["payload"]["options"]
    assert captured["payload"]["options"]["num_predict"] == 64
    assert captured["payload"]["options"]["temperature"] == 0.1


def test_generation_options_are_task_specific():
    assert default_generation_options("auto_complete").max_tokens == 64
    assert default_generation_options("auto_complete").temperature == 0.1
    assert default_generation_options("perf_opt").max_tokens == 192
    assert default_generation_options("test_gen").max_tokens == 384
    assert default_generation_options("project_explain").max_tokens == 300
    assert default_generation_options("code_gen").temperature == 0.2
