from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.rag.project_map import load_project_map
from backend.rag.qdrant_store import QdrantStore


def main() -> None:
    store = QdrantStore()
    project_map = load_project_map()
    rows = store.scroll_payload_rows(limit=10000)
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
    print(f"Point count: {store.point_count()}")
    print(f"Embedding dimension: {store.vector_size()}")
    if project_map:
        print(f"Last indexed time: {project_map.last_indexed_time}")
        print(f"Embedding model: {project_map.embedding_model}")
        print(f"Files scanned: {project_map.files_scanned}")
        print(f"Files skipped: {project_map.files_skipped}")
    else:
        print("Last indexed time: unknown")

    if languages:
        print("Indexed languages:")
        for language, count in languages.most_common():
            print(f"  {language}: {count}")
    elif project_map:
        print("Indexed languages:")
        for language, count in project_map.detected_languages.items():
            print(f"  {language}: {count}")
    else:
        print("Indexed languages: unknown")

    if folders:
        print("Top indexed folders:")
        for folder, count in folders.most_common(15):
            print(f"  {folder}: {count}")
    elif project_map:
        print("Top indexed folders:")
        for folder in project_map.folder_structure[:15]:
            print(f"  {folder}")
    else:
        print("Top indexed folders: unknown")


if __name__ == "__main__":
    main()
