from __future__ import annotations

from dataclasses import dataclass

from backend.api.schemas import GenerateRequest, TaskName
from backend.tools.language_detector import detect_language


@dataclass
class RequestContext:
    task: TaskName
    language: str
    instruction: str
    code: str
    file_path: str | None
    project_path: str | None
    imports: list[str]


class ContextAgent:
    def build(self, request: GenerateRequest, task: TaskName) -> RequestContext:
        language = request.language or detect_language(request.file_path, request.code)
        return RequestContext(
            task=task,
            language=language,
            instruction=request.instruction,
            code=request.code,
            file_path=request.file_path,
            project_path=request.project_path,
            imports=[],
        )
