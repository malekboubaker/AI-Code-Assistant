from backend.api.schemas import TaskName
from backend.config.settings import settings
from backend.model.base import GenerationOptions


TASK_MAX_TOKENS: dict[TaskName, int] = {
    "perf_opt": 192,
    "refactoring": 256,
    "bug_fix": 256,
    "code_gen": 256,
    "test_gen": 384,
    "project_explain": 300,
    "auto_complete": 64,
}


def default_generation_options(task: TaskName | None = None) -> GenerationOptions:
    max_tokens = TASK_MAX_TOKENS.get(task, settings.max_generation_tokens)
    temperature = 0.1 if task == "auto_complete" else 0.2
    return GenerationOptions(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
    )
