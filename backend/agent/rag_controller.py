from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backend.api.schemas import RagSource
from backend.config.settings import settings
from backend.rag.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass
class RagDecision:
    use_rag: bool
    context: str = ""
    sources: list[RagSource] = field(default_factory=list)
    best_score: float | None = None
    threshold: float = settings.rag_threshold
    skip_reason: str | None = None


class RagControllerAgent:
    def __init__(self, retriever: Retriever | None = None) -> None:
        self.retriever = retriever or Retriever()

    def decide(self, query: str, enabled: bool = True, language: str | None = None) -> RagDecision:
        if not enabled:
            logger.info("RAG skipped: request use_rag=false.")
            return RagDecision(use_rag=False, threshold=settings.rag_threshold, skip_reason="request_disabled")
        try:
            results = self.retriever.search(query, top_k=settings.rag_top_k, language=language)
        except TypeError:
            results = self.retriever.search(query, top_k=settings.rag_top_k)
        best_score = results[0].score if results else None
        logger.info(
            "RAG search completed: enabled=%s top_k=%s best_score=%s threshold=%s result_count=%s",
            enabled,
            settings.rag_top_k,
            best_score,
            settings.rag_threshold,
            len(results),
        )
        if results and results[0].score >= settings.rag_threshold:
            logger.info("RAG enabled: best_score=%s threshold=%s", results[0].score, settings.rag_threshold)
            return RagDecision(
                use_rag=True,
                context=self._format_context(results),
                sources=results,
                best_score=best_score,
                threshold=settings.rag_threshold,
            )
        skip_reason = "no_results" if not results else "below_threshold"
        logger.info(
            "RAG skipped: reason=%s best_score=%s threshold=%s",
            skip_reason,
            best_score,
            settings.rag_threshold,
        )
        return RagDecision(
            use_rag=False,
            sources=results,
            best_score=best_score,
            threshold=settings.rag_threshold,
            skip_reason=skip_reason,
        )

    def _format_context(self, sources: list[RagSource]) -> str:
        blocks = []
        for source in sources:
            if source.metadata.get("source") == "project_map" or source.chunk_type == "project_map":
                blocks.append(
                    "### Project map summary\n"
                    f"score: {source.score:.3f}\n"
                    f"{_trim_context(source.content, 2400)}"
                )
                continue
            location = f"{source.file_path}:{source.start_line}-{source.end_line}"
            blocks.append(
                f"### Retrieved chunk ({location})\n"
                f"symbol: {source.symbol_name or 'unknown'} | score: {source.score:.3f}\n"
                f"metadata: type={source.chunk_type or 'unknown'} folder={source.metadata.get('folder', 'unknown')}\n"
                f"```{source.language or ''}\n{_trim_context(source.content, 6000)}\n```"
            )
        return "\n\n".join(blocks)


def _trim_context(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip() + "\n...[trimmed]"
