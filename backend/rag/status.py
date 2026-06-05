from __future__ import annotations

from typing import Any

from backend.rag.project_identity import normalize_project_path, project_id_for_path
from backend.rag.qdrant_store import QdrantStore


def get_project_rag_status(project_path: str, store: QdrantStore | None = None) -> dict[str, Any]:
    active_store = store or QdrantStore()
    project_id = project_id_for_path(project_path)
    normalized_path = normalize_project_path(project_path)
    point_count = int(active_store.point_count(project_id=project_id) or 0)
    project_map_payload = _project_map_payload(active_store, project_id)
    project_map = project_map_payload.get("project_map") if project_map_payload else {}
    if not isinstance(project_map, dict):
        project_map = {}

    frameworks = project_map.get("detected_frameworks", [])
    if not frameworks and project_map_payload:
        frameworks = project_map_payload.get("frameworks", [])

    return {
        "project_id": project_id,
        "project_path": normalized_path,
        "indexed": point_count > 0,
        "project_map_exists": bool(project_map_payload),
        "point_count": point_count,
        "last_indexed": project_map.get("last_indexed_time"),
        "detected_languages": project_map.get("detected_languages", {}),
        "frameworks": frameworks,
        "entry_points": project_map.get("entry_points", []),
        "qdrant_collection": active_store.collection_name,
        "qdrant_ready": active_store.ready,
    }


def project_map_exists(project_path: str, store: QdrantStore | None = None) -> bool:
    active_store = store or QdrantStore()
    return bool(_project_map_payload(active_store, project_id_for_path(project_path)))


def _project_map_payload(store: QdrantStore, project_id: str) -> dict[str, Any] | None:
    try:
        rows = store.scroll_payload_rows(limit=5, project_id=project_id, filters={"source": "project_map"})
    except TypeError:
        rows = store.scroll_payload_rows(limit=1000, project_id=project_id)
    for row in rows:
        payload = row.get("payload", {})
        if payload.get("project_id") == project_id and payload.get("source") == "project_map":
            return payload
    return None
