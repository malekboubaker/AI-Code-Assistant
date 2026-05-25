from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.rag.project_map import load_project_map
from backend.rag.project_identity import normalize_project_path, project_id_for_path
from backend.rag.qdrant_store import QdrantStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Show local RAG/Qdrant indexing status.")
    parser.add_argument("--project", help="Show status for a single project path.")
    args = parser.parse_args()

    store = QdrantStore()
    project_id = project_id_for_path(args.project) if args.project else None
    project_path = normalize_project_path(args.project) if args.project else None
    project_map = None if project_id else load_project_map()
    rows = store.scroll_payload_rows(limit=10000, project_id=project_id)
    project_map_rows = _project_map_rows(store, project_id)
    project_map_data = _project_map_data(project_map_rows or rows, project_map)
    project_map_exists = bool(project_map_data)
    languages = Counter()
    folders = Counter()
    for row in rows:
        payload = row.get("payload", {})
        language = payload.get("language")
        folder = payload.get("folder")
        if language:
            languages[str(language)] += 1
        if folder:
            folders[str(folder)] += 1

    print("AI Code Assistant RAG status")
    print(f"Qdrant ready: {'yes' if store.ready else 'no'}")
    print(f"Collection name: {store.collection_name}")
    if project_id:
        print(f"Project id: {project_id}")
        print(f"Project path: {project_path}")
    print(f"Point count: {store.point_count(project_id=project_id)}")
    print(f"Total chunks: {sum(1 for row in rows if row.get('payload', {}).get('source') != 'project_map')}")
    print(f"Project map exists: {'yes' if project_map_exists else 'no'}")
    print(f"Embedding dimension: {store.vector_size()}")
    if project_map_data:
        print(f"Detected project type: {', '.join(project_map_data.get('project_types', [])) or 'unknown/generic project'}")
        print(f"Detected frameworks: {', '.join(project_map_data.get('detected_frameworks', [])) or 'none detected'}")
        print(f"Last indexed time: {project_map_data.get('last_indexed_time')}")
        print(f"Embedding model: {project_map_data.get('embedding_model')}")
        print(f"Files scanned: {project_map_data.get('files_scanned')}")
        print(f"Files skipped: {project_map_data.get('files_skipped')}")
        print("Entry points:")
        for item in project_map_data.get("entry_points", [])[:15]:
            print(f"  {item}")
        print("Important files:")
        for item in project_map_data.get("important_files", [])[:15]:
            print(f"  {item}")
        print("Skipped folders:")
        for folder, count in sorted(project_map_data.get("skipped_folders", {}).items())[:15]:
            print(f"  {folder}: {count}")
    else:
        print("Detected project type: unknown")
        print("Detected frameworks: unknown")
        print("Last indexed time: unknown")

    if languages:
        print("Indexed languages:")
        for language, count in languages.most_common():
            print(f"  {language}: {count}")
    elif project_map_data:
        print("Indexed languages:")
        for language, count in project_map_data.get("detected_languages", {}).items():
            print(f"  {language}: {count}")
    else:
        print("Indexed languages: unknown")

    if folders:
        print("Indexed folders:")
        for folder, count in folders.most_common(15):
            print(f"  {folder}: {count}")
    elif project_map_data:
        print("Indexed folders:")
        for folder in project_map_data.get("folder_structure", [])[:15]:
            print(f"  {folder}")
    else:
        print("Indexed folders: unknown")


def _project_map_data(rows: list[dict], project_map) -> dict | None:
    for row in rows:
        payload = row.get("payload", {})
        if payload.get("source") == "project_map" and isinstance(payload.get("project_map"), dict):
            return payload["project_map"]
    return project_map.to_dict() if project_map else None


def _project_map_rows(store: QdrantStore, project_id: str | None) -> list[dict]:
    try:
        return store.scroll_payload_rows(limit=5, project_id=project_id, filters={"source": "project_map"})
    except TypeError:
        return store.scroll_payload_rows(limit=10000, project_id=project_id)


if __name__ == "__main__":
    main()
