from __future__ import annotations

import logging
import re

from backend.api.schemas import RagSource
from backend.config.settings import settings
from backend.rag.embedder import LocalEmbedder
from backend.rag.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedder: LocalEmbedder | None = None, store: QdrantStore | None = None) -> None:
        self.embedder = embedder or LocalEmbedder()
        self.store = store or QdrantStore()

    def search(self, query: str, top_k: int | None = None, language: str | None = None) -> list[RagSource]:
        requested_top_k = top_k or settings.rag_top_k
        vector = self.embedder.embed(query)
        symbols = extract_query_symbols(query)
        terms = extract_query_terms(query)
        file_fragments = symbol_file_fragments(symbols + terms)
        semantic_hits = self.store.search(vector, max(requested_top_k * 20, 50))
        keyword_hits = self.store.keyword_search(symbols + terms, file_fragments, limit=max(requested_top_k * 10, 30))
        project_map_hits = self._project_map_hits(query)
        logger.info(
            (
                "Retriever candidates: semantic_hits=%s keyword_hits=%s project_map_hits=%s "
                "symbols=%s terms=%s file_fragments=%s"
            ),
            len(semantic_hits),
            len(keyword_hits),
            len(project_map_hits),
            symbols,
            terms,
            sorted(file_fragments),
        )
        hits = project_map_hits + semantic_hits + keyword_hits
        ranked_hits = self._rank_hits(query, hits, language=language)
        sources: list[RagSource] = []
        seen_locations: set[tuple[str | None, int | None, int | None]] = set()
        for hit in ranked_hits:
            payload = hit.get("payload", {})
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
            if len(sources) >= requested_top_k:
                break
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

    def _rank_hits(self, query: str, hits: list[dict], language: str | None = None) -> list[dict]:
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
            if payload.get("source") == "project_map":
                boost += 0.10
            adjusted = min(1.0, semantic_score + boost)
            ranked_hit = dict(hit)
            ranked_hit["semantic_score"] = semantic_score
            ranked_hit["keyword_boost"] = boost
            ranked_hit["language_match"] = language_match
            ranked_hit["score"] = adjusted
            ranked.append(ranked_hit)
        return sorted(ranked, key=lambda item: (item.get("score", 0.0), bool(item.get("language_match"))), reverse=True)

    def _project_map_hits(self, query: str) -> list[dict]:
        rows = self.store.scroll_payload_rows(limit=200)
        hits = []
        for row in rows:
            payload = row.get("payload", {})
            if payload.get("source") != "project_map":
                continue
            hits.append({"id": row.get("id"), "score": project_map_score(payload, query), "payload": payload})
        return hits[:1]


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
        if term_lower in imports or term_lower in called:
            boost += 0.12
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


def project_map_score(payload: dict, query: str) -> float:
    terms = extract_query_terms(query)
    content = str(payload.get("content", "")).lower()
    project_map = payload.get("project_map") or {}
    haystack = " ".join(
        [
            content,
            " ".join(project_map.get("main_modules", []) if isinstance(project_map, dict) else []),
            " ".join(project_map.get("entry_points", []) if isinstance(project_map, dict) else []),
            " ".join(project_map.get("config_files", []) if isinstance(project_map, dict) else []),
        ]
    ).lower()
    matches = sum(1 for term in terms if term in haystack)
    return min(0.80, 0.58 + matches * 0.04)
