from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import settings
from backend.rag.embedder import LocalEmbedder
from backend.rag.qdrant_store import QdrantStore
from backend.rag.retriever import Retriever


def print_config(query: str, top_k: int) -> None:
    print(f"Query: {query}")
    print(f"Qdrant URL: {settings.qdrant_url}")
    print(f"Collection: {settings.qdrant_collection}")
    print(f"Embedding model: {settings.ollama_embedding_model}")
    print(f"top_k: {top_k}")
    print(f"Similarity threshold: {settings.rag_threshold}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query local RAG index.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--project", default=None, help="Project path to scope retrieval to (required for project-scoped search).")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    store = QdrantStore()
    embedder = LocalEmbedder()

    if args.debug:
        print_config(args.query, args.top_k)

    reachable = store.ready
    if args.debug:
        print(f"Qdrant reachable: {reachable}")
    if not reachable:
        print_config(args.query, args.top_k)
        print("No RAG results found")
        print("Reason: Qdrant is not reachable.")
        return

    collection_exists = store.collection_exists()
    if args.debug:
        print(f"Collection exists: {collection_exists}")
    if not collection_exists:
        print_config(args.query, args.top_k)
        print("No RAG results found")
        print(f"Reason: Collection does not exist: {settings.qdrant_collection}")
        print("Run: python scripts/index_project.py backend --max-files 25")
        return

    point_count = store.point_count()
    collection_vector_dim = store.vector_size()
    query_vector = embedder.embed(args.query)
    query_vector_dim = len(query_vector)

    if args.debug:
        print(f"Query vector dimension: {query_vector_dim}")
        print(f"Collection vector dimension: {collection_vector_dim}")
        print(f"Total points in collection: {point_count}")

    if point_count == 0:
        print_config(args.query, args.top_k)
        print("Collection exists but contains 0 points. Run: python scripts/index_project.py backend --max-files 25")
        print("No RAG results found")
        return

    if collection_vector_dim is not None and query_vector_dim != collection_vector_dim:
        print_config(args.query, args.top_k)
        print("No RAG results found")
        print(
            "Clear error: query vector dimension does not match collection vector dimension "
            f"({query_vector_dim} != {collection_vector_dim})."
        )
        return

    raw_response = None
    if args.debug:
        try:
            raw_response = store.search_raw(query_vector, args.top_k)
            print("Raw Qdrant response:")
            print(json.dumps(raw_response, indent=2)[:10000])
        except Exception as exc:
            print(f"Raw Qdrant search failed: {type(exc).__name__}: {exc}")

    retriever = Retriever(embedder=embedder, store=store)
    results = retriever.search(args.query, top_k=args.top_k, project_path=args.project)
    print(f"Results returned: {len(results)}")

    if not results:
        print_config(args.query, args.top_k)
        print("No RAG results found")
        return

    best_score = results[0].score
    if args.debug:
        print(f"Best score: {best_score}")

    if best_score < settings.rag_threshold:
        print("Results found but below threshold.")

    for result in results:
        print(
            f"{result.score:.3f} {result.file_path}:{result.start_line}-{result.end_line} "
            f"{result.symbol_name} {result.chunk_type}"
        )
        if args.debug:
            print(
                "  "
                f"semantic_score={result.metadata.get('semantic_score')} "
                f"keyword_boost={result.metadata.get('keyword_boost')}"
            )
        print(result.content[:500])
        print()


if __name__ == "__main__":
    main()
