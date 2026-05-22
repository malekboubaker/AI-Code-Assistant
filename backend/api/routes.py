from __future__ import annotations

from fastapi import APIRouter

from backend.agent.orchestrator import AgentOrchestrator
from backend.api.schemas import GenerateRequest, GenerateResponse, HealthResponse, IndexRequest, IndexResponse
from backend.config.settings import settings
from backend.rag.indexer import index_project
from backend.rag.qdrant_store import QdrantStore


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
