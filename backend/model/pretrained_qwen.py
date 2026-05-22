from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from backend.config.settings import settings
from backend.model.base import GenerationOptions, ModelProvider

logger = logging.getLogger(__name__)

CODE_TASKS = {
    "auto_complete",
    "code_gen",
    "bug_detection",
    "bug_fix",
    "perf_opt",
    "test_gen",
    "refactoring",
}


class PretrainedQwenProvider(ModelProvider):
    name = "pretrained_qwen"

    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model

    def generate(self, prompt: str, options: GenerationOptions | None = None) -> str:
        options = options or GenerationOptions()
        task = self._task_from_prompt(prompt)
        prepared_prompt = self._prepare_prompt(prompt, task)
        response_text = self._generate_once(prepared_prompt, options)
        if not response_text.strip() and task in CODE_TASKS:
            logger.warning("Ollama returned empty response for task=%s; retrying once with stricter prompt.", task)
            retry_prompt = self._empty_retry_prompt(prompt, task)
            retry_options = self._retry_options(options, task)
            response_text = self._generate_once(retry_prompt, retry_options)
        if not response_text.strip() and task in CODE_TASKS:
            logger.error("Ollama returned empty response for task=%s after retry; no fallback code will be generated.", task)
        return response_text.strip()

    def _generate_once(self, prompt: str, options: GenerationOptions) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "num_predict": options.max_tokens,
                "temperature": options.temperature,
                "top_p": options.top_p,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            logger.debug("Final prompt sent to Ollama model=%s:\n%s", self.model, prompt)
            logger.debug("Ollama stop sequences disabled for this request.")
            with urllib.request.urlopen(request, timeout=300) as response:
                raw_body = response.read().decode("utf-8")
                logger.debug("Raw Ollama HTTP body: %s", raw_body)
                body = json.loads(raw_body)
                raw_response = body.get("response", "")
                if raw_response is None:
                    raw_response = ""
                response_text = str(raw_response)
                logger.debug("Raw response returned by Ollama: %r", response_text)
                if not response_text.strip():
                    self._log_empty_response_body(body)
                return response_text
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                "Local Ollama model is not reachable. Start Ollama and pull the configured model "
                f"({self.model})."
            ) from exc

    def _task_from_prompt(self, prompt: str) -> str | None:
        for task in CODE_TASKS | {"project_explain"}:
            if f"[TASK: {task}]" in prompt:
                return task
        return None

    def _prepare_prompt(self, prompt: str, task: str | None) -> str:
        if task in CODE_TASKS and not prompt.lstrip().startswith("/no_think"):
            return "/no_think\nReturn only the final answer.\n\n" + prompt
        return prompt

    def _empty_retry_prompt(self, prompt: str, task: str | None) -> str:
        user_instruction = self._extract_section(prompt, "User instruction")
        current_code = self._extract_section(prompt, "Current code")
        if task == "test_gen":
            instruction = (
                "Return ONLY valid pytest code. Include imports when needed. "
                "Create concrete tests. Do not use assert True placeholders."
            )
        elif task == "auto_complete":
            instruction = "Return ONLY the missing code to insert. Do not repeat existing code."
        else:
            instruction = "Return ONLY valid code. No Markdown, no explanation, no headings."
        return (
            "/no_think\n"
            "Return only the final answer.\n"
            f"[TASK: {task or 'code_gen'}]\n"
            "The previous Ollama response field was empty. Retry once with a shorter prompt.\n"
            f"{instruction}\n"
            "The response must not be empty.\n\n"
            "User instruction:\n"
            f"{user_instruction or '(none)'}\n\n"
            "Current code:\n"
            f"{current_code or '(none)'}"
        )

    def _retry_options(self, options: GenerationOptions, task: str | None) -> GenerationOptions:
        min_tokens = 192 if task == "perf_opt" else options.max_tokens
        return GenerationOptions(
            max_tokens=max(options.max_tokens, min_tokens),
            temperature=options.temperature,
            top_p=options.top_p,
        )

    def _extract_section(self, prompt: str, heading: str) -> str:
        marker = f"{heading}:"
        start = prompt.find(marker)
        if start == -1:
            return ""
        start += len(marker)
        next_marker = prompt.find("\n\n", start)
        if next_marker == -1:
            return prompt[start:].strip()
        return prompt[start:next_marker].strip()

    def _log_empty_response_body(self, body: dict) -> None:
        logger.warning(
            (
                "Ollama returned an empty response field. response=%r thinking=%r done=%s "
                "done_reason=%s total_duration=%s eval_count=%s prompt_eval_count=%s raw_body=%s"
            ),
            body.get("response"),
            body.get("thinking"),
            body.get("done"),
            body.get("done_reason"),
            body.get("total_duration"),
            body.get("eval_count"),
            body.get("prompt_eval_count"),
            json.dumps(body, ensure_ascii=False),
        )
