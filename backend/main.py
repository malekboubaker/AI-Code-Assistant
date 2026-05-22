from __future__ import annotations

from fastapi import FastAPI

from backend.api.routes import router
from backend.utils.logging_utils import configure_logging

configure_logging()

app = FastAPI(
    title="Local AI Code Assistant",
    description="100% local model-independent agent backend with local RAG.",
    version="0.1.0",
)
app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "docs": "/docs", "health": "/api/v1/health"}
