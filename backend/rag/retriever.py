from __future__ import annotations

import logging
import re

from backend.api.schemas import RagSource
from backend.config.settings import settings
from backend.rag.embedder import LocalEmbedder
from backend.rag.project_identity import normalize_project_path, project_id_for_path
from backend.rag.qdrant_store import QdrantStore
from backend.rag.file_matching import extract_requested_entities, matches_entity

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedder: LocalEmbedder | None = None, store: QdrantStore | None = None) -> None:
        self.embedder = embedder or LocalEmbedder()
        self.store = store or QdrantStore()
        self.last_diagnostics: dict[str, object] = {}

    def search(
        self,
        query: str,
        top_k: int | None = None,
        language: str | None = None,
        active_file: str | None = None,
        project_path: str | None = None,
        task: str | None = None,
        explanation_scope: str | None = None,
    ) -> list[RagSource]:
        project_id = project_id_for_path(project_path) if project_path else None
        normalized_project_path = normalize_project_path(project_path) if project_path else None
        if not project_id:
            self.last_diagnostics = {
                "project_id": None,
                "project_path": normalized_project_path,
                "qdrant_collection": getattr(self.store, "collection_name", "unknown"),
                "rag_raw_results_count": 0,
                "rag_filtered_results_count": 0,
                "rag_sources_project_ids": [],
                "project_map_exists": False,
                "reliable_source_count": 0,
                "rag_skip_reason": "missing_project_path",
            }
            return []
        requested_top_k = top_k or settings.rag_top_k
        vector = self.embedder.embed(query)
        symbols = extract_query_symbols(query)
        terms = extract_query_terms(query)
        file_fragments = symbol_file_fragments(symbols + terms)
        semantic_hits = self.store.search(vector, max(requested_top_k * 20, 50), project_id=project_id)
        keyword_hits = self.store.keyword_search(
            symbols + terms,
            file_fragments,
            limit=max(requested_top_k * 10, 30),
            project_id=project_id,
        )
        project_map_hits = self._project_map_hits(query, project_id=project_id)
        wants_project_overview = task == "project_explain" or explanation_scope == "project"
        overview_hits = self._hierarchical_project_report_hits(project_id, project_map_hits) if wants_project_overview else []
        
        # Exact File Resolution for compare and file_explain tasks
        exact_hits = []
        requested_entities = extract_requested_entities(query)
        if task in ("compare", "file_explain") and requested_entities:
            logger.info("Exact file resolution triggered for entities: %s", requested_entities)
            # Fetch candidates from the store. We fetch heavily to ensure we find all files.
            candidates = self.store.keyword_search([], [], limit=1000, project_id=project_id)
            for entity in requested_entities:
                if entity.kind != "file":
                    continue
                # Find all chunks belonging to this exact file
                file_chunks = [hit for hit in candidates if matches_entity(hit.get("payload", {}), entity)]
                if file_chunks:
                    exact_hits.extend(file_chunks)
            if exact_hits:
                logger.info("Exact file resolution found %s chunks", len(exact_hits))
                
        logger.info(
            (
                "Retriever candidates: semantic_hits=%s keyword_hits=%s project_map_hits=%s "
                "overview_hits=%s project_id=%s symbols=%s terms=%s file_fragments=%s"
            ),
            len(semantic_hits),
            len(keyword_hits),
            len(project_map_hits),
            len(overview_hits),
            project_id,
            symbols,
            terms,
            sorted(file_fragments),
        )
        
        if exact_hits:
            hits = exact_hits
        elif wants_project_overview:
            hits = project_map_hits + overview_hits
        else:
            hits = project_map_hits + semantic_hits + keyword_hits
        filtered_hits = [hit for hit in hits if hit.get("payload", {}).get("project_id") == project_id]
        ranked_hits = self._rank_hits(query, hits, language=language, active_file=active_file)
        sources: list[RagSource] = []
        seen_locations: set[tuple[str | None, int | None, int | None]] = set()
        for hit in ranked_hits:
            payload = hit.get("payload", {})
            if payload.get("project_id") != project_id:
                logger.warning(
                    "Skipping cross-project RAG hit: expected_project_id=%s hit_project_id=%s file_path=%s",
                    project_id,
                    payload.get("project_id"),
                    payload.get("file_path"),
                )
                continue
            location = (payload.get("file_path"), payload.get("start_line"), payload.get("end_line"))
            if location in seen_locations:
                logger.debug("Skipping duplicate RAG hit for location=%s", location)
                continue
            seen_locations.add(location)
            metadata = dict(payload)
            metadata["semantic_score"] = float(hit.get("semantic_score", hit.get("score", 0.0)))
            metadata["keyword_boost"] = float(hit.get("keyword_boost", 0.0))
            sources.append(
                RagSource(
                    content=payload.get("content", ""),
                    score=float(hit.get("score", 0.0)),
                    language=payload.get("language"),
                    file_path=payload.get("file_path"),
                    start_line=payload.get("start_line"),
                    end_line=payload.get("end_line"),
                    chunk_type=payload.get("chunk_type"),
                    symbol_name=payload.get("symbol_name"),
                    metadata=metadata,
                )
            )
            if not wants_project_overview and len(sources) >= requested_top_k:
                break
        self.last_diagnostics = {
            "project_id": project_id,
            "project_path": normalized_project_path,
            "qdrant_collection": getattr(self.store, "collection_name", "unknown"),
            "rag_raw_results_count": len(hits),
            "rag_filtered_results_count": len(filtered_hits),
            "rag_sources_project_ids": sorted(
                {str(source.metadata.get("project_id")) for source in sources if source.metadata.get("project_id")}
            ),
            "project_map_exists": _has_project_map_source(sources),
            "reliable_source_count": _reliable_project_source_count(sources),
            "rag_skip_reason": None if sources else "no_filtered_results",
        }
        if sources:
            logger.info("Retriever best_score=%s source=%s", sources[0].score, sources[0].file_path)
            for source in sources:
                logger.info(
                    "RAG source: score=%s file_path=%s symbol_name=%s chunk_type=%s semantic_score=%s keyword_boost=%s",
                    source.score,
                    source.file_path,
                    source.symbol_name,
                    source.chunk_type,
                    source.metadata.get("semantic_score"),
                    source.metadata.get("keyword_boost"),
                )
        else:
            logger.info("Retriever returned no sources for query.")
        return sources

    def _rank_hits(
        self,
        query: str,
        hits: list[dict],
        language: str | None = None,
        active_file: str | None = None,
    ) -> list[dict]:
        symbols = extract_query_symbols(query)
        terms = extract_query_terms(query)
        file_fragments = symbol_file_fragments(symbols + terms)
        ranked = []
        for hit in hits:
            payload = hit.get("payload", {})
            semantic_score = float(hit.get("score", 0.0))
            boost = keyword_boost(payload, symbols, terms, file_fragments)
            language_match = bool(language and payload.get("language") == language)
            if language_match:
                boost += 0.08
            boost += importance_boost(payload)
            boost += active_file_boost(payload, active_file)
            if payload.get("source") == "project_map":
                boost += 0.10
            adjusted = min(1.0, semantic_score + boost)
            ranked_hit = dict(hit)
            ranked_hit["semantic_score"] = semantic_score
            ranked_hit["keyword_boost"] = boost
            ranked_hit["language_match"] = language_match
            ranked_hit["importance_score"] = safe_float(payload.get("importance_score", 0.0))
            ranked_hit["score"] = adjusted
            ranked.append(ranked_hit)
        return sorted(
            ranked,
            key=lambda item: (
                item.get("score", 0.0),
                bool(item.get("language_match")),
                item.get("importance_score", 0.0),
            ),
            reverse=True,
        )

    def count_project_points(self, project_id: str) -> int:
        return int(self.store.point_count(project_id=project_id) or 0)

    def _project_map_hits(self, query: str, project_id: str) -> list[dict]:
        try:
            rows = self.store.scroll_payload_rows(limit=20, project_id=project_id, filters={"source": "project_map"})
        except TypeError:
            rows = self.store.scroll_payload_rows(limit=1200, project_id=project_id)
        hits = []
        for row in rows:
            payload = row.get("payload", {})
            if payload.get("source") != "project_map":
                continue
            hits.append({"id": row.get("id"), "score": project_map_score(payload, query), "payload": payload})
        return hits[:1]

    def _hierarchical_project_report_hits(self, project_id: str, project_map_hits: list[dict]) -> list[dict]:
        project_map = {}
        if project_map_hits:
            project_map = project_map_hits[0].get("payload", {}).get("project_map") or {}
        
        important = set(project_map.get("important_files", []) if isinstance(project_map, dict) else [])
        entry_points = set(project_map.get("entry_points", []) if isinstance(project_map, dict) else [])
        config_files = set(project_map.get("config_files", []) if isinstance(project_map, dict) else [])
        docs = set(project_map.get("documentation_files", []) if isinstance(project_map, dict) else [])
        
        rows = self.store.scroll_payload_rows(limit=1500, project_id=project_id)
        
        readmes = []
        folder_summaries = []
        file_summaries = []
        raw_code = []
        
        for row in rows:
            payload = row.get("payload", {})
            if payload.get("project_id") != project_id:
                continue
                
            chunk_type = payload.get("chunk_type")
            rel = str(payload.get("relative_file_path") or payload.get("relative_path") or "")
            score = 1.0
            
            if chunk_type == "folder_summary":
                folder_summaries.append({"id": row.get("id"), "score": score, "payload": payload})
            elif chunk_type == "file_summary":
                if rel in entry_points: score = 0.95
                elif rel in important: score = 0.90
                elif payload.get("is_config_file"): score = 0.85
                elif payload.get("importance_score", 0) > 0.5: score = 0.80
                else: score = 0.60
                file_summaries.append({"id": row.get("id"), "score": score, "payload": payload})
            elif chunk_type == "doc" and (rel.lower().startswith("readme") or rel.lower().endswith("readme.md")):
                readmes.append({"id": row.get("id"), "score": 1.0, "payload": payload})
            elif payload.get("source") == "project_code":
                if rel in entry_points: score = 0.70
                elif rel in important: score = 0.65
                else: score = 0.50
                raw_code.append({"id": row.get("id"), "score": score, "payload": payload})
                
        # Sort collections by score
        file_summaries.sort(key=lambda x: x["score"], reverse=True)
        raw_code.sort(key=lambda x: x["score"], reverse=True)
        
        # Adaptive Context Budget: Stop around ~25,000 characters
        MAX_CHARS = 25000
        current_chars = 0
        if project_map_hits:
            current_chars += len(str(project_map_hits[0].get("payload", {}).get("content", "")))
            
        final_hits = []
        
        def _add_if_fits(collection, take_all=False, limit=None):
            nonlocal current_chars
            added = 0
            for hit in collection:
                if limit and added >= limit:
                    break
                content_len = len(str(hit.get("payload", {}).get("content", "")))
                if not take_all and current_chars + content_len > MAX_CHARS:
                    continue
                final_hits.append(hit)
                current_chars += content_len
                added += 1

        # Strict Hierarchy
        _add_if_fits(readmes, limit=1)
        _add_if_fits(folder_summaries, take_all=True) # Always include folder summaries
        _add_if_fits(file_summaries) # Fill rest with file summaries
        
        # Only if we have massive room (very small project) do we add raw code
        _add_if_fits(raw_code)
        
        seen = set()
        deduped = []
        for h in final_hits:
            fpath = h.get("payload", {}).get("file_path")
            # If it's a folder summary, use its chunk ID instead of file_path
            if h.get("payload", {}).get("chunk_type") == "folder_summary":
                fpath = h.get("id")
            if fpath not in seen:
                seen.add(fpath)
                deduped.append(h)
        return deduped


def extract_query_symbols(query: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", query)))


STOPWORDS = {
    "add",
    "and",
    "bug",
    "code",
    "file",
    "fix",
    "for",
    "from",
    "generate",
    "into",
    "logic",
    "please",
    "refactor",
    "request",
    "show",
    "test",
    "tests",
    "the",
    "this",
    "unit",
    "with",
}


def extract_query_terms(query: str) -> list[str]:
    raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9_/-]{2,}", query.lower())
    terms: list[str] = []
    for term in raw_terms:
        for piece in re.split(r"[/_-]+", term):
            if len(piece) >= 3 and piece not in STOPWORDS:
                terms.append(piece)
        if len(term) >= 3 and term not in STOPWORDS:
            terms.append(term.replace("/", " "))
    return list(dict.fromkeys(terms))[:30]


def symbol_file_fragments(symbols: list[str]) -> set[str]:
    fragments: set[str] = set()
    for symbol in symbols:
        words = [word.lower() for word in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", symbol)]
        if not words:
            continue
        fragments.add("_".join(words))
        if words[-1] == "agent" and len(words) > 1:
            fragments.add("_".join(words[:-1]))
        for word in words:
            if word not in {"agent", "class", "function"}:
                fragments.add(word)
    return fragments


def keyword_boost(payload: dict, symbols: list[str], terms: list[str], file_fragments: set[str]) -> float:
    if not symbols and not terms and not file_fragments:
        return 0.0
    boost = 0.0
    content = str(payload.get("content", ""))
    content_lower = content.lower()
    symbol_name = str(payload.get("symbol_name") or "")
    symbol_lower = symbol_name.lower()
    file_path = str(payload.get("file_path") or "").replace("\\", "/").lower()
    folder = str(payload.get("folder") or "").replace("\\", "/").lower()
    imports = " ".join(str(item) for item in payload.get("imports") or []).lower()
    called = " ".join(str(item) for item in payload.get("called_functions") or []).lower()
    dependencies = " ".join(str(item) for item in payload.get("dependency_neighbors") or []).lower()
    frameworks = " ".join(str(item) for item in payload.get("frameworks") or []).lower()
    project_types = " ".join(str(item) for item in payload.get("project_types") or []).lower()
    for symbol in symbols:
        query_symbol_lower = symbol.lower()
        if symbol_lower == query_symbol_lower:
            boost += 0.30
        if query_symbol_lower in content_lower:
            boost += 0.20
    for term in terms:
        term_lower = term.lower()
        if term_lower == symbol_lower or term_lower in symbol_lower:
            boost += 0.28
        if term_lower in file_path or term_lower in folder:
            boost += 0.24
        if term_lower in content_lower:
            boost += 0.14
        if term_lower in imports or term_lower in called or term_lower in dependencies:
            boost += 0.12
        if term_lower in frameworks or term_lower in project_types:
            boost += 0.10
    for fragment in file_fragments:
        if fragment and fragment in file_path:
            boost += 0.18
        if fragment and fragment in symbol_lower:
            boost += 0.16
    if payload.get("is_test_file"):
        test_terms = {"test", "tests", "pytest", "unit"}
        if any(term in test_terms for term in terms):
            boost += 0.15
    if payload.get("is_config_file") and any(term in {"config", "docker", "timeout", "environment"} for term in terms):
        boost += 0.15
    return min(boost, 0.60)


def importance_boost(payload: dict) -> float:
    importance = safe_float(payload.get("importance_score", 0.0))
    boost = min(importance * 0.08, 0.20)
    if payload.get("is_entry_point"):
        boost += 0.10
    if payload.get("imported_by_count"):
        boost += min(safe_float(payload.get("imported_by_count", 0)) * 0.03, 0.15)
    return min(boost, 0.30)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def active_file_boost(payload: dict, active_file: str | None) -> float:
    if not active_file:
        return 0.0
    active = active_file.replace("\\", "/").lower()
    file_path = str(payload.get("file_path") or "").replace("\\", "/").lower()
    relative_path = str(payload.get("relative_path") or "").replace("\\", "/").lower()
    folder = str(payload.get("folder") or "").replace("\\", "/").lower()
    if active.endswith(file_path) or active.endswith(relative_path):
        return 0.25
    active_folder = "/".join(active.split("/")[:-1])
    if folder and (active_folder.endswith(folder) or folder in active_folder):
        return 0.10
    neighbors = [str(item).replace("\\", "/").lower() for item in payload.get("dependency_neighbors") or []]
    active_name = active.split("/")[-1]
    if any(active.endswith(neighbor) or neighbor.endswith(active_name) for neighbor in neighbors):
        return 0.18
    return 0.0


def _has_project_map_source(sources: list[RagSource]) -> bool:
    return any(source.metadata.get("source") == "project_map" or source.chunk_type == "project_map" for source in sources)


def _reliable_project_source_count(sources: list[RagSource]) -> int:
    return sum(1 for source in sources if _is_reliable_project_source(source))


def _is_reliable_project_source(source: RagSource) -> bool:
    if source.metadata.get("source") == "project_map" or source.chunk_type == "project_map":
        return True
    if source.score < settings.rag_threshold:
        return False
    if source.metadata.get("is_doc_file") or source.metadata.get("is_config_file") or source.metadata.get("is_entry_point"):
        return True
    if safe_float(source.metadata.get("importance_score", 0.0)) >= 0.3:
        return True
    relative = str(source.metadata.get("relative_file_path") or source.metadata.get("relative_path") or "").lower()
    if relative.startswith("readme") or relative.endswith("/readme.md"):
        return True
    return False


def project_map_score(payload: dict, query: str) -> float:
    terms = extract_query_terms(query)
    content = str(payload.get("content", "")).lower()
    project_map = payload.get("project_map") or {}
    haystack = " ".join(
        [
            content,
            " ".join(project_map.get("project_types", []) if isinstance(project_map, dict) else []),
            " ".join(project_map.get("detected_frameworks", []) if isinstance(project_map, dict) else []),
            " ".join(project_map.get("main_modules", []) if isinstance(project_map, dict) else []),
            " ".join(project_map.get("entry_points", []) if isinstance(project_map, dict) else []),
            " ".join(project_map.get("config_files", []) if isinstance(project_map, dict) else []),
            " ".join(project_map.get("important_files", []) if isinstance(project_map, dict) else []),
            " ".join(project_map.get("dependency_files", []) if isinstance(project_map, dict) else []),
        ]
    ).lower()
    matches = sum(1 for term in terms if term in haystack)
    return min(0.82, 0.64 + matches * 0.04)
