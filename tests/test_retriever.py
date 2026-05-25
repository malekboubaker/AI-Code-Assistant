from backend.rag.embedder import LocalEmbedder
from backend.rag.qdrant_store import QdrantStore
from backend.rag.retriever import Retriever
from backend.rag.project_identity import normalize_project_path, project_id_for_path
from backend.api.schemas import RagSource
from backend.agent.rag_controller import RagControllerAgent


def test_retriever_fallback_store(tmp_path, monkeypatch):
    store = QdrantStore()
    store._client = None
    store.fallback_path = tmp_path / "vectors.jsonl"
    embedder = LocalEmbedder(vector_size=32)
    vector = embedder.embed("calculate hash")
    project_id = project_id_for_path(tmp_path)
    store.upsert(
        vector,
        {
            "content": "def calculate_hash(): pass",
            "language": "python",
            "project_id": project_id,
            "project_path": normalize_project_path(tmp_path),
            "file_path": "x.py",
        },
    )
    results = Retriever(embedder=embedder, store=store).search("calculate hash", top_k=1, project_path=str(tmp_path))
    assert results
    assert results[0].score > 0


class FakeRetriever:
    def search(self, query, top_k=None, project_path=None, **kwargs):
        project_id = project_id_for_path(project_path)
        return [
            RagSource(
                content="class AgentOrchestrator:\n    pass",
                score=0.653,
                language="python",
                file_path="backend/agent/orchestrator.py",
                start_line=1,
                end_line=2,
                metadata={"project_id": project_id},
            )
        ]


def test_rag_controller_enables_rag_above_local_threshold():
    decision = RagControllerAgent(retriever=FakeRetriever()).decide(
        "orchestrator task router",
        enabled=True,
        project_path=".",
    )
    assert decision.use_rag is True
    assert decision.best_score == 0.653
    assert decision.threshold == 0.60
    assert decision.sources
    assert "Retrieved chunk" in decision.context


class HybridStore:
    collection_name = "test"

    def search(self, vector, top_k=5, project_id=None):
        return []

    def keyword_search(self, symbols, file_fragments, limit=50, project_id=None):
        return [
            {
                "id": "auth",
                "score": 0.55,
                "payload": {
                    "content": "def authenticate_user():\n    return database.connect()",
                    "language": "python",
                    "project_id": project_id,
                    "file_path": "backend/authentication/service.py",
                    "start_line": 10,
                    "end_line": 12,
                    "chunk_type": "function",
                    "symbol_name": "authenticate_user",
                    "folder": "backend/authentication",
                    "called_functions": ["connect"],
                },
            }
        ]

    def _rows(self, project_id):
        return [
            {
                "id": "map",
                "payload": {
                    "source": "project_map",
                    "content": "Project map summary:\n- Main modules: backend/authentication/service.py",
                    "project_id": project_id,
                    "file_path": "project_map.json",
                    "language": "text",
                    "start_line": 1,
                    "end_line": 2,
                    "chunk_type": "project_map",
                    "symbol_name": "project_map",
                    "project_map": {"main_modules": ["backend/authentication/service.py"]},
                },
            }
        ]

    def scroll_payload_rows(self, limit=200, project_id=None):
        return self._rows(project_id)


def test_retriever_uses_keyword_and_project_map_for_concepts():
    embedder = LocalEmbedder(vector_size=32)
    results = Retriever(embedder=embedder, store=HybridStore()).search(
        "Optimize the authentication database logic",
        top_k=2,
        project_path=".",
    )

    assert any(result.file_path == "backend/authentication/service.py" for result in results)
    assert any(result.metadata.get("source") == "project_map" for result in results)


def test_retriever_boosts_matching_language():
    retriever = Retriever(embedder=LocalEmbedder(vector_size=32), store=HybridStore())
    hits = [
        {"score": 0.50, "payload": {"content": "function run() {}", "language": "javascript", "file_path": "src/run.js"}},
        {"score": 0.50, "payload": {"content": "def run(): pass", "language": "python", "file_path": "src/run.py"}},
    ]

    ranked = retriever._rank_hits("run", hits, language="python")

    assert ranked[0]["payload"]["language"] == "python"
