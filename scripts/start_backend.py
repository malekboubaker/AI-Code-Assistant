from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import settings


if __name__ == "__main__":
    print("=" * 60)
    print("Starting AI Code Assistant Backend")
    print(f"Effective Qdrant URL: {settings.qdrant_url}")
    print(f"Effective Ollama URL: {settings.ollama_base_url}")
    print("=" * 60)
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=True)
