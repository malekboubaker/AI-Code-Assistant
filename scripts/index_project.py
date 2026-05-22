from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.rag.indexer import index_project_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a local project into local Qdrant/fallback vector store.")
    parser.add_argument("project_path")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="Re-index all scanned files instead of only changed files.")
    args = parser.parse_args()
    report = index_project_report(args.project_path, max_files=args.max_files, full=args.full)
    print("AI Code Assistant RAG indexing complete")
    print(f"Files scanned: {report.files_scanned}")
    print(f"Files skipped: {report.files_skipped}")
    if report.skipped_by_reason:
        print("Skipped by reason:")
        for reason, count in sorted(report.skipped_by_reason.items()):
            print(f"  {reason}: {count}")
    print(f"Files re-indexed: {report.files_reindexed}")
    print(f"Files unchanged: {report.files_unchanged}")
    print(f"Chunks indexed: {report.chunks_indexed}")
    print(f"Chunks deleted: {report.chunks_deleted}")
    print(f"Qdrant collection: {report.collection_name}")
    print(f"Embedding model: {report.embedding_model}")
    print(f"Project map: {report.project_map_path}")
    print(f"Total time: {report.total_time_ms} ms")


if __name__ == "__main__":
    main()
