from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


TaskName = Literal[
    "auto_complete",
    "code_gen",
    "bug_detection",
    "bug_fix",
    "perf_opt",
    "test_gen",
    "refactoring",
    "explain",
    "project_explain",
    "file_explain",
    "compare",
]

LanguageName = Literal["python", "javascript", "typescript", "java", "cpp", "csharp", "rust"]


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class GenerateRequest(BaseModel):
    response_id: str | None = None
    instruction: str = Field(..., description="User instruction or question.")
    code: str = Field("", description="Selected/current code.")
    task: TaskName | None = None
    language: LanguageName | str | None = Field(
        None,
        description="Programming language. Supported code languages include python, javascript, typescript, java, cpp, csharp, and rust for RAG/explanation.",
    )
    file_path: str | None = None
    project_path: str | None = None
    has_selection: bool | None = Field(
        None,
        description="True when code is an explicit editor selection rather than cursor/file context.",
    )
    surrounding_context: str = Field(
        "",
        description="Optional nearby file context around the selected code.",
    )
    chat_history: list[ChatHistoryMessage] = Field(
        default_factory=list,
        description="Recent sidebar chat messages from the current session.",
    )
    use_rag: bool = True
    accepted: bool = False
    run_tests: bool = False


class RagSource(BaseModel):
    content: str
    score: float
    language: str | None = None
    file_path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    chunk_type: str | None = None
    symbol_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    valid: bool
    syntax_valid: bool | None = None
    tests_passed: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    validator: str = "unknown"
    duration_ms: int = 0


class WorkspaceEdit(BaseModel):
    file_path: str
    reason: str = ""
    original_content: str | None = None
    new_content: str


class GenerateResponse(BaseModel):
    task: TaskName
    language: str
    generated_code: str
    explanation: str = ""
    diff: str | None = None
    used_rag: bool = False
    rag_sources: list[RagSource] = Field(default_factory=list)
    validation: ValidationResult
    edits: list[WorkspaceEdit] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectReportRequest(BaseModel):
    project_path: str

class ProjectReportResponse(BaseModel):
    status: str
    message: str | None = None
    project_id: str
    project_name: str
    project_path: str
    generated_at: str
    duration_ms: int
    report_markdown: str | None = None

class IndexRequest(BaseModel):
    project_path: str
    collection_name: str | None = None
    max_files: int | None = None


class IndexResponse(BaseModel):
    files_indexed: int
    chunks_indexed: int
    collection_name: str


class RagStatusResponse(BaseModel):
    project_id: str
    project_path: str
    indexed: bool
    project_map_exists: bool
    point_count: int
    last_indexed: str | None = None
    detected_languages: dict[str, int] = Field(default_factory=dict)
    frameworks: list[str] = Field(default_factory=list)
    entry_points: list[str] = Field(default_factory=list)
    qdrant_collection: str
    qdrant_ready: bool = False


class RagIndexRequest(BaseModel):
    project_path: str
    mode: Literal["incremental", "full"] = "incremental"


class RagIndexResponse(BaseModel):
    status: str
    project_id: str
    files_scanned: int
    files_indexed: int
    files_skipped: int
    chunks_created: int
    chunks_stored: int
    project_map_exists: bool
    duration_ms: int


class RagResetRequest(BaseModel):
    project_path: str


class RagResetResponse(BaseModel):
    status: str
    project_id: str
    deleted_points: int


class HealthResponse(BaseModel):
    status: str
    model_provider: str
    qdrant_ready: bool


class CancelResponse(BaseModel):
    status: str
    message: str
