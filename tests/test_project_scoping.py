from __future__ import annotations

import sys
from pathlib import Path

from backend.agent.orchestrator import (
    AgentOrchestrator,
    INSUFFICIENT_PROJECT_CONTEXT_MESSAGE,
    MISSING_PROJECT_MAP_MESSAGE,
)
from backend.agent.rag_controller import RagControllerAgent
from backend.api.schemas import GenerateRequest, RagSource
from backend.rag.embedder import LocalEmbedder
from backend.rag.indexer import ProjectIndexer
from backend.rag.project_identity import normalize_project_path, project_id_for_path
from backend.rag.qdrant_store import QdrantStore
from backend.rag.retriever import Retriever


class CapturingStore:
    collection_name = "test_chunks"

    def __init__(self):
        self.payloads = []

    def ensure_collection(self, vector_size):
        return None

    def delete_by_file_path(self, file_path, project_id=None):
        return 0

    def delete_by_filter(self, filters):
        return 0

    def upsert(self, vector, payload, point_id=None):
        self.payloads.append(payload)
        return point_id or "point"


def test_indexed_chunks_contain_project_identity(tmp_path: Path):
    (tmp_path / "main.py").write_text("def run():\n    return 'ok'\n", encoding="utf-8")
    store = CapturingStore()

    report = ProjectIndexer(embedder=LocalEmbedder(vector_size=16), store=store).index(str(tmp_path), full=True)

    assert report.project_id == project_id_for_path(tmp_path)
    assert store.payloads
    for payload in store.payloads:
        assert payload["project_id"] == report.project_id
        assert payload["project_name"] == tmp_path.name
        assert payload["project_path"] == normalize_project_path(tmp_path)
        assert payload["workspace_root"] == normalize_project_path(tmp_path)
        assert payload["file_path"]
        assert payload["relative_file_path"]


def test_full_index_creates_project_map_payload(tmp_path: Path):
    (tmp_path / "README.md").write_text("# Travel Planner\nPlans trips.\n", encoding="utf-8")
    (tmp_path / "main.py").write_text("def run():\n    return 'ok'\n", encoding="utf-8")
    store = CapturingStore()

    report = ProjectIndexer(embedder=LocalEmbedder(vector_size=16), store=store).index(str(tmp_path), full=True)

    project_maps = [payload for payload in store.payloads if payload.get("source") == "project_map"]
    assert len(project_maps) == 1
    project_map = project_maps[0]
    assert project_map["project_id"] == report.project_id
    assert project_map["project_path"] == normalize_project_path(tmp_path)
    assert project_map["project_name"] == tmp_path.name
    assert project_map["detected_languages"]["python"] == 1
    assert isinstance(project_map["frameworks"], list)
    assert isinstance(project_map["entry_points"], list)
    assert isinstance(project_map["important_files"], list)
    assert isinstance(project_map["indexed_folders"], list)
    assert "Project map summary" in project_map["summary"]


def test_retrieval_filters_by_project_id_and_blocks_cross_project_chunks(tmp_path: Path):
    project_a = tmp_path / "AI Travel Planner"
    project_b = tmp_path / "AI Code Assistant"
    project_a.mkdir()
    project_b.mkdir()
    store = QdrantStore()
    store._client = None
    store.fallback_path = tmp_path / "vectors.jsonl"
    embedder = LocalEmbedder(vector_size=32)
    project_a_id = project_id_for_path(project_a)
    project_b_id = project_id_for_path(project_b)

    store.upsert(
        embedder.embed("travel itinerary planner"),
        {
            "content": "Travel itinerary planner",
            "language": "python",
            "project_id": project_a_id,
            "file_path": str(project_a / "planner.py"),
            "relative_file_path": "planner.py",
            "source": "project_code",
        },
    )
    store.upsert(
        embedder.embed("Qdrant RagControllerAgent QdrantStore"),
        {
            "content": "Qdrant RagControllerAgent QdrantStore",
            "language": "python",
            "project_id": project_b_id,
            "file_path": str(project_b / "rag.py"),
            "relative_file_path": "rag.py",
            "source": "project_code",
        },
    )

    results = Retriever(embedder=embedder, store=store).search(
        "Qdrant RagControllerAgent QdrantStore",
        top_k=5,
        project_path=str(project_a),
    )

    assert results
    assert all(source.metadata["project_id"] == project_a_id for source in results)
    assert all("Qdrant" not in source.content for source in results)


class NoIndexRetriever:
    def count_project_points(self, project_id):
        return 0


class RaisingModel:
    name = "raising"

    def generate(self, prompt, options=None):
        raise AssertionError("Model should not be called without indexed project context")


def test_project_explain_refuses_when_project_is_not_indexed(tmp_path: Path):
    orchestrator = AgentOrchestrator()
    orchestrator.model_provider = RaisingModel()
    orchestrator.rag_controller = RagControllerAgent(retriever=NoIndexRetriever())

    response = orchestrator.run(
        GenerateRequest(
            task="project_explain",
            instruction="Explain this project",
            language="python",
            project_path=str(tmp_path),
            use_rag=True,
        )
    )

    assert response.explanation == INSUFFICIENT_PROJECT_CONTEXT_MESSAGE
    assert response.used_rag is False
    assert response.metadata["rag_skip_reason"] == "project_not_indexed"


class HallucinatingModel:
    name = "hallucinating"

    def generate(self, prompt, options=None):
        return "This project uses Qdrant, RAG, vector databases, RagControllerAgent, and QdrantStore for retrieval."


class GroundedRetriever:
    def count_project_points(self, project_id):
        return 2

    def search(self, query, top_k=None, project_path=None, **kwargs):
        project_id = project_id_for_path(project_path)
        return [
            RagSource(
                content="Project map summary:\n- Project type: Python\n- Entry points: planner.py",
                score=0.72,
                language="text",
                file_path="project_map.json",
                start_line=1,
                end_line=2,
                chunk_type="project_map",
                metadata={"source": "project_map", "project_id": project_id},
            ),
            RagSource(
                content="The project helps build travel itineraries from local planner data.",
                score=0.68,
                language="markdown",
                file_path="README.md",
                start_line=1,
                end_line=1,
                chunk_type="doc",
                metadata={"project_id": project_id},
            ),
        ]


def test_project_explain_blocks_technologies_absent_from_retrieved_context(tmp_path: Path):
    orchestrator = AgentOrchestrator()
    orchestrator.model_provider = HallucinatingModel()
    orchestrator.rag_controller = RagControllerAgent(retriever=GroundedRetriever())

    response = orchestrator.run(
        GenerateRequest(
            task="project_explain",
            instruction="Explain this project",
            language="python",
            project_path=str(tmp_path),
            use_rag=True,
        )
    )

    assert "Qdrant" not in response.explanation
    assert response.explanation == INSUFFICIENT_PROJECT_CONTEXT_MESSAGE
    assert "qdrant" in response.metadata["grounding_blocked_terms"]
    assert "rag" in response.metadata["grounding_blocked_terms"]
    assert "vector databases" in response.metadata["grounding_blocked_terms"]


class GroundedExplanationModel:
    name = "grounded"

    def __init__(self):
        self.prompt = ""

    def generate(self, prompt, options=None):
        self.prompt = prompt
        return "This project helps build travel itineraries from local planner data."


def test_project_explain_uses_project_map_when_available(tmp_path: Path):
    orchestrator = AgentOrchestrator()
    model = GroundedExplanationModel()
    orchestrator.model_provider = model
    orchestrator.rag_controller = RagControllerAgent(retriever=GroundedRetriever())

    response = orchestrator.run(
        GenerateRequest(
            task="project_explain",
            instruction="Explain this project",
            language="python",
            project_path=str(tmp_path),
            use_rag=True,
        )
    )

    assert response.used_rag is True
    assert response.metadata["project_map_exists"] is True
    assert response.explanation.startswith("This project helps")
    assert "Project map summary" in model.prompt


class ReliableChunksNoMapRetriever:
    def count_project_points(self, project_id):
        return 3

    def search(self, query, top_k=None, project_path=None, **kwargs):
        project_id = project_id_for_path(project_path)
        return [
            RagSource(
                content="# Travel Planner\nBuilds local travel itineraries.",
                score=0.72,
                language="markdown",
                file_path="README.md",
                start_line=1,
                end_line=2,
                chunk_type="doc",
                metadata={
                    "project_id": project_id,
                    "is_doc_file": True,
                    "relative_file_path": "README.md",
                },
            )
        ]


def test_project_explain_does_not_refuse_when_reliable_sources_exist_without_map(tmp_path: Path):
    orchestrator = AgentOrchestrator()
    model = GroundedExplanationModel()
    orchestrator.model_provider = model
    orchestrator.rag_controller = RagControllerAgent(retriever=ReliableChunksNoMapRetriever())

    response = orchestrator.run(
        GenerateRequest(
            task="project_explain",
            instruction="Explain this project",
            language="python",
            project_path=str(tmp_path),
            use_rag=True,
        )
    )

    assert response.used_rag is True
    assert response.metadata["project_map_exists"] is False
    assert response.metadata["reliable_source_count"] == 1
    assert response.explanation != MISSING_PROJECT_MAP_MESSAGE


class UnreliableChunksNoMapRetriever:
    def count_project_points(self, project_id):
        return 3

    def search(self, query, top_k=None, project_path=None, **kwargs):
        project_id = project_id_for_path(project_path)
        return [
            RagSource(
                content="def helper():\n    return 1",
                score=0.72,
                language="python",
                file_path="misc.py",
                start_line=1,
                end_line=2,
                chunk_type="function",
                metadata={"project_id": project_id, "relative_file_path": "misc.py"},
            )
        ]


def test_project_explain_with_chunks_but_missing_map_returns_clear_message(tmp_path: Path):
    orchestrator = AgentOrchestrator()
    orchestrator.model_provider = RaisingModel()
    orchestrator.rag_controller = RagControllerAgent(retriever=UnreliableChunksNoMapRetriever())

    response = orchestrator.run(
        GenerateRequest(
            task="project_explain",
            instruction="Explain this project",
            language="python",
            project_path=str(tmp_path),
            use_rag=True,
        )
    )

    assert response.used_rag is False
    assert response.explanation == MISSING_PROJECT_MAP_MESSAGE
    assert response.metadata["rag_skip_reason"] == "missing_project_map_full_index_required"


def test_rag_status_reports_project_specific_point_count(tmp_path: Path, monkeypatch, capsys):
    import scripts.rag_status as rag_status

    project_id = project_id_for_path(tmp_path)

    class FakeStatusStore:
        ready = True
        collection_name = "test_chunks"

        def scroll_payload_rows(self, limit=10000, project_id=None):
            assert project_id == project_id_for_path(tmp_path)
            return [
                {
                    "payload": {
                        "source": "project_map",
                        "project_id": project_id,
                        "project_map": {
                            "project_types": ["Python"],
                            "detected_frameworks": [],
                            "entry_points": ["main.py"],
                            "important_files": ["main.py"],
                            "detected_languages": {"python": 1},
                            "folder_structure": ["."],
                            "last_indexed_time": "now",
                            "embedding_model": "local",
                            "files_scanned": 1,
                            "files_skipped": 0,
                            "skipped_folders": {},
                        },
                    }
                }
            ]

        def point_count(self, project_id=None):
            return 7

        def vector_size(self):
            return 32

    monkeypatch.setattr(rag_status, "QdrantStore", FakeStatusStore)
    monkeypatch.setattr(sys, "argv", ["rag_status.py", "--project", str(tmp_path)])

    rag_status.main()

    output = capsys.readouterr().out
    assert f"Project id: {project_id}" in output
    assert "Point count: 7" in output
    assert "Project map exists: yes" in output
    assert "Detected project type: Python" in output
