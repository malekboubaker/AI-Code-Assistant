from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.api import routes
from backend.main import app
from backend.rag.project_identity import project_id_for_path


client = TestClient(app)


def test_rag_status_for_indexed_project(monkeypatch, tmp_path):
    project_id = project_id_for_path(tmp_path)

    def fake_status(project_path):
        return {
            "project_id": project_id,
            "project_path": str(tmp_path),
            "indexed": True,
            "project_map_exists": True,
            "point_count": 12,
            "last_indexed": "2026-05-25T00:00:00Z",
            "detected_languages": {"python": 2},
            "frameworks": ["FastAPI"],
            "entry_points": ["main.py"],
            "qdrant_collection": "code_chunks",
            "qdrant_ready": True,
        }

    monkeypatch.setattr(routes, "get_project_rag_status", fake_status)

    response = client.get("/api/v1/rag/status", params={"project_path": str(tmp_path)})

    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == project_id
    assert body["indexed"] is True
    assert body["project_map_exists"] is True
    assert body["point_count"] == 12
    assert body["frameworks"] == ["FastAPI"]


def test_rag_status_for_unindexed_project(monkeypatch, tmp_path):
    project_id = project_id_for_path(tmp_path)

    monkeypatch.setattr(
        routes,
        "get_project_rag_status",
        lambda project_path: {
            "project_id": project_id,
            "project_path": str(tmp_path),
            "indexed": False,
            "project_map_exists": False,
            "point_count": 0,
            "last_indexed": None,
            "detected_languages": {},
            "frameworks": [],
            "entry_points": [],
            "qdrant_collection": "code_chunks",
            "qdrant_ready": True,
        },
    )

    response = client.get("/api/v1/rag/status", params={"project_path": str(tmp_path)})

    assert response.status_code == 200
    assert response.json()["indexed"] is False
    assert response.json()["project_map_exists"] is False


def test_incremental_index_endpoint(monkeypatch, tmp_path):
    calls = []
    project_id = project_id_for_path(tmp_path)

    def fake_index(project_path, full=False):
        calls.append((project_path, full))
        return SimpleNamespace(
            project_id=project_id,
            files_scanned=3,
            files_reindexed=1,
            files_skipped=1,
            chunks_indexed=4,
            total_time_ms=25,
        )

    monkeypatch.setattr(routes, "index_project_report", fake_index)
    monkeypatch.setattr(routes, "project_map_exists", lambda project_path: True)

    response = client.post("/api/v1/rag/index", json={"project_path": str(tmp_path), "mode": "incremental"})

    assert response.status_code == 200
    assert calls == [(str(tmp_path), False)]
    body = response.json()
    assert body["status"] == "success"
    assert body["files_indexed"] == 1
    assert body["chunks_stored"] == 4
    assert body["project_map_exists"] is True


def test_full_index_endpoint_requests_full_project_rebuild(monkeypatch, tmp_path):
    calls = []
    project_id = project_id_for_path(tmp_path)

    def fake_index(project_path, full=False):
        calls.append((project_path, full))
        return SimpleNamespace(
            project_id=project_id,
            files_scanned=5,
            files_reindexed=5,
            files_skipped=0,
            chunks_indexed=8,
            total_time_ms=50,
        )

    monkeypatch.setattr(routes, "index_project_report", fake_index)
    monkeypatch.setattr(routes, "project_map_exists", lambda project_path: True)

    response = client.post("/api/v1/rag/index", json={"project_path": str(tmp_path), "mode": "full"})

    assert response.status_code == 200
    assert calls == [(str(tmp_path), True)]
    assert response.json()["chunks_created"] == 8


def test_reset_endpoint_deletes_only_current_project(monkeypatch, tmp_path):
    calls = []
    project_id = project_id_for_path(tmp_path)

    def fake_reset(project_path):
        calls.append(project_path)
        return project_id, 7

    monkeypatch.setattr(routes, "reset_project_index", fake_reset)

    response = client.post("/api/v1/rag/reset", json={"project_path": str(tmp_path)})

    assert response.status_code == 200
    assert calls == [str(tmp_path)]
    assert response.json() == {"status": "success", "project_id": project_id, "deleted_points": 7}
