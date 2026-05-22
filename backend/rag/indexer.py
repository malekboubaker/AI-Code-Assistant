from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config.settings import ROOT_DIR, settings
from backend.rag.chunker import chunk_file
from backend.rag.embedder import LocalEmbedder
from backend.rag.project_map import build_project_map, project_map_payload, save_project_map
from backend.rag.qdrant_store import QdrantStore
from backend.rag.scanner import scan_project_with_stats


INDEX_STATE_PATH = ROOT_DIR / "data" / "metadata" / "rag_index_state.json"


@dataclass
class IndexReport:
    files_scanned: int
    files_skipped: int
    chunks_indexed: int
    chunks_deleted: int
    files_reindexed: int
    files_unchanged: int
    collection_name: str
    embedding_model: str
    total_time_ms: int
    project_map_path: str
    skipped_by_reason: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProjectIndexer:
    def __init__(self, embedder: LocalEmbedder | None = None, store: QdrantStore | None = None) -> None:
        self.embedder = embedder or LocalEmbedder()
        self.store = store or QdrantStore()

    def index(self, project_path: str, max_files: int | None = None, full: bool = False) -> IndexReport:
        started = time.perf_counter()
        root = Path(project_path).resolve()
        scan = scan_project_with_stats(str(root), max_files=max_files)
        state = {} if full else _load_state()
        project_key = str(root)
        project_state = state.get(project_key, {})
        previous_files: dict[str, dict[str, Any]] = dict(project_state.get("files", {}))
        current_files: dict[str, dict[str, Any]] = {}
        chunks_indexed = 0
        chunks_deleted = 0
        files_reindexed = 0
        files_unchanged = 0

        self.store.ensure_collection(self.embedder.vector_size)
        for path in scan.files:
            relative_path = _relative_path(path, root)
            file_hash = _file_hash(path)
            previous = previous_files.get(relative_path)
            current_files[relative_path] = {
                "sha256": file_hash,
                "last_modified": path.stat().st_mtime,
            }
            if previous and previous.get("sha256") == file_hash and not full:
                files_unchanged += 1
                continue

            chunks_deleted += max(0, self.store.delete_by_file_path(str(path)))
            for chunk in chunk_file(path, project_root=root):
                chunk.payload["file_sha256"] = file_hash
                vector = self.embedder.embed(_embedding_text(chunk.payload))
                point_id = _point_id(project_key, relative_path, chunk.payload)
                self.store.upsert(vector, chunk.payload, point_id=point_id)
                chunks_indexed += 1
            files_reindexed += 1

        removed_files = set(previous_files) - set(current_files)
        for relative_path in removed_files:
            chunks_deleted += max(0, self.store.delete_by_file_path(str(root / relative_path)))

        indexed_at = datetime.now(timezone.utc).isoformat()
        project_map = build_project_map(
            str(root),
            scan.files,
            skipped_files=scan.skipped_files,
            skipped_by_reason=scan.skipped_by_reason,
            last_indexed_time=indexed_at,
            embedding_model=self.embedder.model,
            collection_name=self.store.collection_name,
        )
        project_map_path = save_project_map(project_map)
        self.store.delete_by_payload("source", "project_map")
        self.store.upsert(
            self.embedder.embed(project_map.to_summary()),
            project_map_payload(project_map),
            point_id=_project_map_point_id(project_key),
        )

        state[project_key] = {
            "last_indexed_time": indexed_at,
            "collection_name": self.store.collection_name,
            "embedding_model": self.embedder.model,
            "files": current_files,
        }
        _save_state(state)

        return IndexReport(
            files_scanned=scan.files_scanned,
            files_skipped=scan.skipped_files,
            chunks_indexed=chunks_indexed,
            chunks_deleted=chunks_deleted,
            files_reindexed=files_reindexed,
            files_unchanged=files_unchanged,
            collection_name=self.store.collection_name,
            embedding_model=self.embedder.model,
            total_time_ms=round((time.perf_counter() - started) * 1000),
            project_map_path=str(project_map_path),
            skipped_by_reason=scan.skipped_by_reason,
        )


def index_project(project_path: str, max_files: int | None = None, full: bool = False) -> tuple[int, int, str]:
    report = index_project_report(project_path, max_files=max_files, full=full)
    return report.files_scanned, report.chunks_indexed, report.collection_name


def index_project_report(project_path: str, max_files: int | None = None, full: bool = False) -> IndexReport:
    indexer = ProjectIndexer()
    return indexer.index(project_path, max_files=max_files, full=full)


def _embedding_text(payload: dict[str, Any]) -> str:
    parts = [
        str(payload.get("file_path", "")),
        str(payload.get("symbol_name") or ""),
        str(payload.get("parent_scope") or ""),
        str(payload.get("chunk_type") or ""),
        str(payload.get("content") or ""),
    ]
    return "\n".join(part for part in parts if part)


def _point_id(project_key: str, relative_path: str, payload: dict[str, Any]) -> str:
    key = "|".join(
        [
            project_key,
            relative_path,
            str(payload.get("start_line")),
            str(payload.get("end_line")),
            str(payload.get("symbol_name") or ""),
            str(payload.get("file_sha256") or ""),
        ]
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _project_map_point_id(project_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{project_key}|project_map"))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return path.name


def _load_state(path: Path = INDEX_STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(state: dict[str, Any], path: Path = INDEX_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
