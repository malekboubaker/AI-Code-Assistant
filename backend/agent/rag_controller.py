from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backend.api.schemas import RagSource
from backend.config.settings import settings
from backend.rag.project_identity import normalize_project_path, project_id_for_path
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
    project_id: str | None = None
    project_path: str | None = None
    qdrant_collection: str | None = None
    raw_results_count: int = 0
    filtered_results_count: int = 0
    sources_project_ids: list[str] = field(default_factory=list)
    project_point_count: int | None = None
    project_map_exists: bool = False
    reliable_source_count: int = 0


class RagControllerAgent:
    def __init__(self, retriever: Retriever | None = None) -> None:
        self.retriever = retriever or Retriever()

    def decide(
        self,
        query: str,
        enabled: bool = True,
        language: str | None = None,
        active_file: str | None = None,
        project_path: str | None = None,
        task: str | None = None,
    ) -> RagDecision:
        project_id = project_id_for_path(project_path) if project_path else None
        normalized_project_path = normalize_project_path(project_path) if project_path else None
        qdrant_collection = getattr(getattr(self.retriever, "store", None), "collection_name", None)
        if not enabled:
            logger.info("RAG skipped: request use_rag=false.")
            return RagDecision(
                use_rag=False,
                threshold=settings.rag_threshold,
                skip_reason="request_disabled",
                project_id=project_id,
                project_path=normalized_project_path,
                qdrant_collection=qdrant_collection,
            )
        if not project_id:
            logger.info("RAG skipped: missing project_path.")
            return RagDecision(
                use_rag=False,
                threshold=settings.rag_threshold,
                skip_reason="missing_project_path",
                project_path=normalized_project_path,
                qdrant_collection=qdrant_collection,
            )
        project_point_count = None
        if hasattr(self.retriever, "count_project_points"):
            project_point_count = self.retriever.count_project_points(project_id)
            if project_point_count <= 0:
                logger.info("RAG skipped: project_id=%s is not indexed.", project_id)
                return RagDecision(
                    use_rag=False,
                    threshold=settings.rag_threshold,
                    skip_reason="project_not_indexed",
                    project_id=project_id,
                    project_path=normalized_project_path,
                    qdrant_collection=qdrant_collection,
                    project_point_count=project_point_count,
                    project_map_exists=False,
                    reliable_source_count=0,
                )
        try:
            results = self.retriever.search(
                query,
                top_k=settings.rag_top_k,
                language=language,
                active_file=active_file,
                project_path=project_path,
                task=task,
            )
        except TypeError:
            results = self.retriever.search(query, top_k=settings.rag_top_k)
        diagnostics = getattr(self.retriever, "last_diagnostics", {}) or {}
        best_score = results[0].score if results else None
        project_map_exists = bool(diagnostics.get("project_map_exists")) or _has_project_map_source(results)
        reliable_source_count = int(diagnostics.get("reliable_source_count", _reliable_project_source_count(results)))
        logger.info(
            "RAG search completed: enabled=%s top_k=%s best_score=%s threshold=%s result_count=%s",
            enabled,
            settings.rag_top_k,
            best_score,
            settings.rag_threshold,
            len(results),
        )
        if results and results[0].score >= settings.rag_threshold:
            if task == "project_explain" and not project_map_exists and reliable_source_count <= 0:
                return RagDecision(
                    use_rag=False,
                    sources=results,
                    best_score=best_score,
                    threshold=settings.rag_threshold,
                    skip_reason="missing_project_map_full_index_required" if project_point_count else "no_reliable_project_context",
                    project_id=project_id,
                    project_path=normalized_project_path,
                    qdrant_collection=qdrant_collection,
                    raw_results_count=int(diagnostics.get("rag_raw_results_count", len(results))),
                    filtered_results_count=int(diagnostics.get("rag_filtered_results_count", len(results))),
                    sources_project_ids=list(diagnostics.get("rag_sources_project_ids", [])),
                    project_point_count=project_point_count,
                    project_map_exists=project_map_exists,
                    reliable_source_count=reliable_source_count,
                )
            logger.info("RAG enabled: best_score=%s threshold=%s", results[0].score, settings.rag_threshold)
            return RagDecision(
                use_rag=True,
                context=self._format_context(results),
                sources=results,
                best_score=best_score,
                threshold=settings.rag_threshold,
                project_id=project_id,
                project_path=normalized_project_path,
                qdrant_collection=qdrant_collection,
                raw_results_count=int(diagnostics.get("rag_raw_results_count", len(results))),
                filtered_results_count=int(diagnostics.get("rag_filtered_results_count", len(results))),
                sources_project_ids=list(diagnostics.get("rag_sources_project_ids", [])),
                project_point_count=project_point_count,
                project_map_exists=project_map_exists,
                reliable_source_count=reliable_source_count,
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
            project_id=project_id,
            project_path=normalized_project_path,
            qdrant_collection=qdrant_collection,
            raw_results_count=int(diagnostics.get("rag_raw_results_count", len(results))),
            filtered_results_count=int(diagnostics.get("rag_filtered_results_count", len(results))),
            sources_project_ids=list(diagnostics.get("rag_sources_project_ids", [])),
            project_point_count=project_point_count,
            project_map_exists=project_map_exists,
            reliable_source_count=reliable_source_count,
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


def _has_project_map_source(results: list[RagSource]) -> bool:
    return any(source.metadata.get("source") == "project_map" or source.chunk_type == "project_map" for source in results)


def _reliable_project_source_count(results: list[RagSource]) -> int:
    return sum(1 for source in results if _is_reliable_project_source(source))


def _is_reliable_project_source(source: RagSource) -> bool:
    if source.metadata.get("source") == "project_map" or source.chunk_type == "project_map":
        return True
    if source.score < settings.rag_threshold:
        return False
    if source.metadata.get("is_doc_file") or source.metadata.get("is_config_file") or source.metadata.get("is_entry_point"):
        return True
    try:
        importance = float(source.metadata.get("importance_score", 0.0))
    except (TypeError, ValueError):
        importance = 0.0
    if importance >= 0.3:
        return True
    relative = str(source.metadata.get("relative_file_path") or source.metadata.get("relative_path") or "").lower()
    return relative.startswith("readme") or relative.endswith("/readme.md")
