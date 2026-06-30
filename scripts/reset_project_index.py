from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.rag.project_identity import normalize_project_path, project_id_for_path
from backend.rag.qdrant_store import QdrantStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete local RAG index entries for one project only.")
    parser.add_argument("project_path")
    args = parser.parse_args()

    project_id = project_id_for_path(args.project_path)
    project_path = normalize_project_path(args.project_path)
    store = QdrantStore()
    deleted = store.delete_by_project_id(project_id)

    print("AI Code Assistant project RAG index reset")
    print(f"Project id: {project_id}")
    print(f"Project path: {project_path}")
    print(f"Qdrant collection: {store.collection_name}")
    print(f"Deleted points: {deleted}")


if __name__ == "__main__":
    main()
