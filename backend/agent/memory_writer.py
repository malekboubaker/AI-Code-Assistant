from __future__ import annotations

import logging

from backend.api.schemas import ValidationResult
from backend.rag.embedder import LocalEmbedder
from backend.rag.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class MemoryWriterAgent:
    def __init__(self, embedder: LocalEmbedder | None = None, store: QdrantStore | None = None) -> None:
        self.embedder = embedder or LocalEmbedder()
        self.store = store or QdrantStore()

    def maybe_store(
        self,
        content: str,
        language: str,
        task: str,
        validation: ValidationResult,
        accepted: bool = False,
        file_path: str | None = None,
        is_fallback: bool = False,
    ) -> bool:
        skip_reason = self._skip_reason(content, validation, is_fallback)
        if skip_reason:
            logger.debug("Memory storage skipped for task=%s: %s", task, skip_reason)
            return False
        should_store = accepted or validation.tests_passed is True
        if not should_store:
            logger.debug("Memory storage skipped for task=%s: output was not accepted and tests did not pass.", task)
            return False

        payload = {
            "content": content,
            "language": language,
            "file_path": file_path or "generated://local-agent",
            "start_line": 1,
            "end_line": len(content.splitlines()),
            "chunk_type": "generated_result",
            "symbol_name": None,
            "parent_scope": None,
            "imports": [],
            "task_tags": [task],
            "source": "generated_memory",
            "validated": validation.valid,
            "created_by": "memory_writer",
        }
        self.store.upsert(self.embedder.embed(content), payload)
        logger.debug("Memory storage allowed for task=%s: accepted=%s tests_passed=%s", task, accepted, validation.tests_passed)
        return True

    def _skip_reason(self, content: str, validation: ValidationResult, is_fallback: bool) -> str | None:
        lowered = content.lower()
        if not content.strip():
            return "generated_code is empty"
        if is_fallback:
            return "output was marked fallback/empty"
        if "fallback test generated" in lowered or "assert true" in lowered:
            return "placeholder fallback output detected"
        if "```" in content or "**explanation:**" in lowered:
            return "Markdown or explanation marker detected in generated_code"
        if not validation.valid:
            return "validation failed"
        if validation.syntax_valid is not True:
            return "syntax_valid is false"
        return None
