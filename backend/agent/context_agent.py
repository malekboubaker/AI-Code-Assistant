from __future__ import annotations

import re
from dataclasses import dataclass

from backend.api.schemas import ChatHistoryMessage, GenerateRequest, TaskName
from backend.tools.language_detector import detect_language
from backend.agent.task_router import TaskRouterAgent
from backend.agent.conversation_memory import memory_store, ConversationTurn
from backend.rag.file_matching import extract_file_references, extract_requested_entities

ExplanationScope = str

PROJECT_SCOPE_PATTERNS = (
    r"\b(?:this|the|current)?\s*(?:project|repo|repository|workspace|codebase)\b",
    r"\b(?:this|the|current)?\s*(?:application|app|system)\b",
    r"\barchitecture\b",
    r"\boverall\b",
    r"\bwhat\s+does\s+(?:this|the)\s+(?:project|repo|repository|application|app|system|codebase)\s+do\b",
    r"\bhow\s+(?:the\s+)?(?:components|modules|services|agents)\s+(?:work|fit)\s+together\b",
)

FILE_SCOPE_PATTERNS = (
    r"\b(?:this|the|current|active)\s+file\b",
    r"\bfile-level\b",
)

SELECTION_SCOPE_PATTERNS = (
    r"\b(?:this|the|selected)\s+(?:code|function|method|class|snippet|selection)\b",
    r"\bfunction-level\b",
    r"\bmethod-level\b",
    r"\bselected\s+code\b",
)


@dataclass
class RequestContext:
    task: TaskName
    language: str
    instruction: str
    code: str
    file_path: str | None
    project_path: str | None
    imports: list[str]
    explanation_scope: ExplanationScope = "selection"
    has_selection: bool = False
    selected_code_primary: bool = False
    active_file_path: str | None = None
    surrounding_context: str = ""
    chat_history: list[ChatHistoryMessage] | None = None


class ContextAgent:
    def build(self, request: GenerateRequest, task: TaskName) -> RequestContext:
        project_path = request.project_path or ""
        
        # 1. Resolve semantic conversational references ("the previous file", "both", etc.)
        resolved_refs = memory_store.resolve_references(project_path, request.instruction)
        
        # 2. Extract explicitly named files in this new instruction
        explicit_entities = extract_requested_entities(request.instruction)
        explicit_files = [e.name for e in explicit_entities if e.kind == "file"]
        
        # 3. Combine them into a new effective instruction string to pass down the pipeline
        effective_instruction = request.instruction
        if resolved_refs and not explicit_files:
            # If the user used conversational references but didn't explicitly name a file,
            # we inject the resolved files into the instruction so the task router & RAG see them
            appended = " ".join(resolved_refs)
            effective_instruction = f"{request.instruction} {appended}"
            
        language = request.language or detect_language(request.file_path, request.code)
        
        # Update request instruction so TaskRouter handles it correctly if it runs again
        request.instruction = effective_instruction
        
        # Re-detect the task using the updated instruction (which now contains explicit filenames)
        router = TaskRouterAgent()
        updated_task = router.detect(effective_instruction, task if task != "project_explain" else None)
        
        has_selection = infer_has_selection(request, updated_task)
        explanation_scope = infer_explanation_scope(request, updated_task, has_selection=has_selection)
        selected_code_primary = bool(has_selection and request.code.strip() and explanation_scope != "project")
        
        # 4. Record this turn in the conversation memory
        turn = ConversationTurn(
            user_intent=request.instruction,
            task=updated_task,
            active_file=request.file_path,
            files_referenced=explicit_files or resolved_refs
        )
        memory_store.add_turn(project_path, turn)
        
        return RequestContext(
            task=updated_task,
            language=language,
            instruction=effective_instruction,
            code=request.code,
            file_path=request.file_path,
            project_path=project_path,
            imports=[],
            explanation_scope=explanation_scope,
            has_selection=has_selection,
            selected_code_primary=selected_code_primary,
            active_file_path=request.file_path,
            surrounding_context=request.surrounding_context,
            chat_history=list(request.chat_history),
        )


def infer_explanation_scope(
    request: GenerateRequest,
    task: TaskName,
    *,
    has_selection: bool | None = None,
) -> ExplanationScope:
    explicit_selection = has_selection if has_selection is not None else infer_has_selection(request, task)
    if task != "project_explain":
        return "selection" if explicit_selection else "file"

    instruction = request.instruction.lower()
    if _matches_any(PROJECT_SCOPE_PATTERNS, instruction):
        return "project"
    if _matches_any(FILE_SCOPE_PATTERNS, instruction):
        return "file"
    if _matches_any(SELECTION_SCOPE_PATTERNS, instruction):
        return "selection"
    if explicit_selection:
        return "selection"
    if request.file_path:
        return "file"
    if not request.code and request.project_path:
        return "project"
    return "selection"


def _matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def infer_has_selection(request: GenerateRequest, task: TaskName) -> bool:
    if request.has_selection is not None:
        return bool(request.has_selection and request.code.strip())
    if not request.code.strip():
        return False
    instruction = request.instruction.lower()
    if _matches_any(SELECTION_SCOPE_PATTERNS, instruction):
        return True
    if task in {"bug_fix", "refactoring", "perf_opt", "test_gen"} and re.search(r"\bselected\b|\bthis\s+function\b", instruction):
        return True
    return False
