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
from backend.rag.project_analyzer import ProjectAnalyzer
from backend.rag.project_identity import normalize_project_path, project_id_for_path, project_name_for_path
from backend.rag.project_map import build_project_map, project_map_payload, save_project_map
from backend.rag.qdrant_store import QdrantStore
from backend.rag.scanner import scan_project_with_stats


INDEX_STATE_PATH = ROOT_DIR / "data" / "metadata" / "rag_index_state.json"
INDEX_MANIFEST_PATH = ROOT_DIR / "data" / "metadata" / "index_manifest.json"


@dataclass
class IndexReport:
    project_id: str
    project_name: str
    project_path: str
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
        normalized_project_path = normalize_project_path(root)
        project_id = project_id_for_path(root)
        project_name = project_name_for_path(root)
        scan = scan_project_with_stats(str(root), max_files=max_files)
        analysis = ProjectAnalyzer().analyze(str(root), scan.files)
        state = _load_state()
        project_key = project_id
        project_state = {} if full else state.get(project_key, {})
        previous_files: dict[str, dict[str, Any]] = dict(project_state.get("files", {}))
        current_files: dict[str, dict[str, Any]] = {}
        chunks_indexed = 0
        chunks_deleted = 0
        files_reindexed = 0
        files_unchanged = 0

        self.store.ensure_collection(self.embedder.vector_size)
        if full and hasattr(self.store, "delete_by_project_id"):
            chunks_deleted += max(0, self.store.delete_by_project_id(project_id))
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

            if not full:
                chunks_deleted += max(0, self.store.delete_by_file_path(str(path), project_id=project_id))
            file_signal = analysis.file_signals.get(relative_path)
            for chunk in chunk_file(path, project_root=root):
                chunk.payload.update(
                    {
                        "project_id": project_id,
                        "project_name": project_name,
                        "project_path": normalized_project_path,
                        "workspace_root": normalized_project_path,
                        "file_path": str(path),
                        "relative_file_path": relative_path,
                        "relative_path": relative_path,
                        "file_sha256": file_hash,
                    }
                )
                if file_signal:
                    chunk.payload.update(
                        {
                            "project_types": analysis.project_types,
                            "frameworks": analysis.detected_frameworks,
                            "importance_score": file_signal.importance_score,
                            "import_count": file_signal.import_count,
                            "imported_by_count": file_signal.imported_by_count,
                            "symbol_count": file_signal.symbol_count,
                            "dependency_neighbors": file_signal.dependency_neighbors[:30],
                            "is_entry_point": file_signal.is_entry_point,
                        }
                    )
                vector = self.embedder.embed(_embedding_text(chunk.payload))
                point_id = _point_id(project_key, relative_path, chunk.payload)
                self.store.upsert(vector, chunk.payload, point_id=point_id)
                chunks_indexed += 1
            files_reindexed += 1

        if not full:
            removed_files = set(previous_files) - set(current_files)
            for relative_path in removed_files:
                chunks_deleted += max(0, self.store.delete_by_file_path(str(root / relative_path), project_id=project_id))

        indexed_at = datetime.now(timezone.utc).isoformat()
        project_map = build_project_map(
            str(root),
            scan.files,
            skipped_files=scan.skipped_files,
            skipped_by_reason=scan.skipped_by_reason,
            skipped_folders=scan.skipped_folders,
            last_indexed_time=indexed_at,
            embedding_model=self.embedder.model,
            collection_name=self.store.collection_name,
            analysis=analysis,
        )
        project_map_path = save_project_map(project_map)
        self.store.delete_by_filter({"source": "project_map", "project_id": project_id})
        self.store.upsert(
            self.embedder.embed(project_map.to_summary()),
            project_map_payload(project_map),
            point_id=_project_map_point_id(project_key),
        )

        state[project_key] = {
            "project_id": project_id,
            "project_name": project_name,
            "project_path": normalized_project_path,
            "workspace_root": normalized_project_path,
            "last_indexed_time": indexed_at,
            "collection_name": self.store.collection_name,
            "embedding_model": self.embedder.model,
            "files": current_files,
        }
        _save_state(state)
        _save_index_manifest(
            project_id=project_id,
            project_name=project_name,
            project_path=normalized_project_path,
            indexed_at=indexed_at,
            collection_name=self.store.collection_name,
            embedding_model=self.embedder.model,
            files_scanned=scan.files_scanned,
            chunks_indexed=chunks_indexed,
        )

        return IndexReport(
            project_id=project_id,
            project_name=project_name,
            project_path=normalized_project_path,
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


def _load_index_manifest(path: Path = INDEX_MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"projects": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if "projects" not in data or not isinstance(data["projects"], dict):
        data["projects"] = {}
    return data


def _save_index_manifest(
    *,
    project_id: str,
    project_name: str,
    project_path: str,
    indexed_at: str,
    collection_name: str,
    embedding_model: str,
    files_scanned: int,
    chunks_indexed: int,
    path: Path = INDEX_MANIFEST_PATH,
) -> None:
    manifest = _load_index_manifest(path)
    manifest["projects"][project_id] = {
        "project_id": project_id,
        "project_name": project_name,
        "project_path": project_path,
        "workspace_root": project_path,
        "last_indexed_time": indexed_at,
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "files_scanned": files_scanned,
        "chunks_indexed": chunks_indexed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
