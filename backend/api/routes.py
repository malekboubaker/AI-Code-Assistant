from __future__ import annotations

from fastapi import APIRouter, Query

from backend.agent.orchestrator import AgentOrchestrator
from backend.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    RagIndexRequest,
    RagIndexResponse,
    RagResetRequest,
    RagResetResponse,
    RagStatusResponse,
)
from backend.config.settings import settings
from backend.rag.indexer import index_project, index_project_report, reset_project_index
from backend.rag.qdrant_store import QdrantStore
from backend.rag.status import get_project_rag_status, project_map_exists


router = APIRouter(prefix="/api/v1")
orchestrator = AgentOrchestrator()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_provider=settings.model_provider,
        qdrant_ready=QdrantStore().ready,
    )


@router.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    return orchestrator.run(request)


@router.post("/complete", response_model=GenerateResponse)
def complete(request: GenerateRequest) -> GenerateResponse:
    request.task = "auto_complete"
    return orchestrator.run(request)


@router.post("/index", response_model=IndexResponse)
def index(request: IndexRequest) -> IndexResponse:
    files, chunks, collection = index_project(request.project_path, max_files=request.max_files)
    return IndexResponse(files_indexed=files, chunks_indexed=chunks, collection_name=request.collection_name or collection)


@router.get("/rag/status", response_model=RagStatusResponse)
def rag_status(project_path: str = Query(...)) -> RagStatusResponse:
    return RagStatusResponse(**get_project_rag_status(project_path))


@router.post("/rag/index", response_model=RagIndexResponse)
def rag_index(request: RagIndexRequest) -> RagIndexResponse:
    report = index_project_report(request.project_path, full=request.mode == "full")
    return RagIndexResponse(
        status="success",
        project_id=report.project_id,
        files_scanned=report.files_scanned,
        files_indexed=report.files_reindexed,
        files_skipped=report.files_skipped,
        chunks_created=report.chunks_indexed,
        chunks_stored=report.chunks_indexed,
        project_map_exists=project_map_exists(request.project_path),
        duration_ms=report.total_time_ms,
    )


@router.post("/rag/reset", response_model=RagResetResponse)
def rag_reset(request: RagResetRequest) -> RagResetResponse:
    project_id, deleted_points = reset_project_index(request.project_path)
    return RagResetResponse(status="success", project_id=project_id, deleted_points=deleted_points)
