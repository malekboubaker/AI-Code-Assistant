from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("AI_ASSIST_HOST", "127.0.0.1")
    port: int = int(os.getenv("AI_ASSIST_PORT", "8000"))
    model_provider: str = os.getenv("MODEL_PROVIDER", "pretrained_qwen")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "code_chunks")
    qdrant_storage_path: Path = ROOT_DIR / "qdrant_storage" / "fallback_vectors.jsonl"
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    rag_threshold: float = float(os.getenv("RAG_THRESHOLD", "0.60"))
    max_generation_tokens: int = int(os.getenv("MAX_GENERATION_TOKENS", "768"))
    temperature: float = float(os.getenv("GENERATION_TEMPERATURE", "0.2"))
    continuation_limit: int = int(os.getenv("CONTINUATION_LIMIT", "3"))
    model_context_window: int = int(os.getenv("MODEL_CONTEXT_WINDOW", "32768"))


settings = Settings()
