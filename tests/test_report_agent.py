from fastapi.testclient import TestClient

from backend.agent.report_agent import ProjectReportAgent
from backend.api import routes
from backend.api.schemas import ProjectReportRequest, ProjectReportResponse, RagSource
from backend.main import app
from backend.rag.project_identity import project_id_for_path


client = TestClient(app)


class FakeReportStore:
    def scroll_payload_rows(self, limit=5, project_id=None, filters=None):
        return [
            {
                "id": "map",
                "payload": {
                    "project_id": project_id,
                    "source": "project_map",
                    "detected_languages": {"python": 10},
                    "detected_frameworks": ["FastAPI"],
                    "entry_points": ["backend/main.py"],
                    "important_files": ["backend/main.py"],
                    "important_files": ["backend/main.py"],
                    "project_map": {
                        "important_files": ["backend/main.py"],
                        "source_folders": ["backend/agent", "backend/api"],
                        "dependency_files": ["requirements.txt"],
                        "graph": {
                            "report": {
                                "most_connected_modules": ["backend/agent"],
                                "critical_files": ["backend/main.py"],
                                "hotspots": ["backend/agent/orchestrator.py"],
                                "edge_counts": {"FILE_IMPORTS_FILE": 12},
                            },
                            "file_relations": {
                                "backend/main.py": {"imports": ["backend/api/routes.py"]}
                            },
                            "api_routes": [
                                {
                                    "method": "POST",
                                    "path": "/api/v1/generate",
                                    "handler": "generate",
                                    "file": "backend/api/routes.py",
                                }
                            ],
                        },
                    },
                },
            }
        ]


class FakeReportRetriever:
    def __init__(self):
        self.store = FakeReportStore()

    def count_project_points(self, project_id):
        return 25

    def search(self, query, top_k=None, project_path=None, task=None, explanation_scope=None, **kwargs):
        project_id = project_id_for_path(project_path)
        return [
            RagSource(
                content="Project map summary:\n- Project type: Python",
                score=0.82,
                language="text",
                file_path="project_map.json",
                start_line=1,
                end_line=2,
                chunk_type="project_map",
                symbol_name="project_map",
                metadata={"source": "project_map", "project_id": project_id, "relative_file_path": "project_map.json"},
            ),
            RagSource(
                content="from fastapi import FastAPI\napp = FastAPI()",
                score=0.74,
                language="python",
                file_path="backend/main.py",
                start_line=1,
                end_line=2,
                chunk_type="file",
                metadata={"project_id": project_id, "relative_file_path": "backend/main.py"},
            ),
            RagSource(
                content="def generate(): ...",
                score=0.7,
                language="python",
                file_path="backend/api/routes.py",
                start_line=1,
                end_line=2,
                chunk_type="function",
                symbol_name="generate",
                metadata={"project_id": project_id, "relative_file_path": "backend/api/routes.py"},
            ),
        ]


class FakeReportModel:
    name = "fake-report"

    def __init__(self):
        self.prompt = ""

    def generate(self, prompt, options=None):
        self.prompt = prompt
        return (
            "## 1. Executive Summary\n"
            "This project is a local AI code assistant backend.\n\n"
            "## 2. Project Purpose\n"
            "It helps developers generate and explain code locally.\n\n"
            "## 12. Suggested Improvements\n"
            "- High: add more tests.\n"
        )


class NotIndexedRetriever:
    def __init__(self):
        self.store = FakeReportStore()

    def count_project_points(self, project_id):
        return 0

    def search(self, *args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("search should not run for an unindexed project")


def test_report_agent_generates_structured_grounded_report():
    model = FakeReportModel()
    agent = ProjectReportAgent(retriever=FakeReportRetriever(), model_provider=model)

    result = agent.generate(ProjectReportRequest(project_path="."))

    assert result.status == "success"
    assert result.markdown.startswith("# Project Report")
    assert "## 1. Executive Summary" in result.markdown
    assert "## 12. Suggested Improvements" in result.markdown
    assert result.files_analyzed == 2
    assert "backend/main.py" in result.source_files
    assert "project_map.json" not in result.source_files
    assert result.project_map_used is True
    assert result.rag_enabled is True
    assert "## 13. Sources & Evidence" in result.markdown
    assert "Files analyzed: 2" in result.markdown
    assert "Source diversity score:" in result.markdown
    assert "AI code assistant" in result.summary
    # The grounded facts (technologies, endpoints, dependency arrows) reach the prompt.
    assert "FastAPI" in model.prompt
    assert "## 1. Executive Summary" in model.prompt
    assert "POST /api/v1/generate" in model.prompt
    assert "backend/main.py → backend/api/routes.py" in model.prompt
    assert "[API endpoints]" in model.prompt
    assert "[Dependency relationships]" in model.prompt
    # The deprecated maturity section is gone from the new structure.
    assert "Project Maturity Assessment" not in model.prompt


def test_report_agent_reports_unindexed_project():
    agent = ProjectReportAgent(retriever=NotIndexedRetriever(), model_provider=FakeReportModel())

    result = agent.generate(ProjectReportRequest(project_path="."))

    assert result.status == "not_indexed"
    assert result.markdown == ""
    assert "indexing" in result.message.lower()


def test_report_endpoint_returns_report(monkeypatch, tmp_path):
    def fake_generate(request):
        return ProjectReportResponse(
            status="success",
            markdown="# Project Report\n## 1. Executive Summary\nDemo.",
            summary="Demo.",
            project_name="demo",
            project_path=str(tmp_path),
            files_analyzed=4,
            source_files=["a.py", "b.py"],
            project_map_used=True,
            rag_enabled=True,
            generated_at="2026-06-17T10:00:00Z",
            duration_ms=12,
        )

    monkeypatch.setattr(routes.report_agent, "generate", fake_generate)

    response = client.post("/api/v1/report", json={"project_path": str(tmp_path)})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["files_analyzed"] == 4
    assert body["project_map_used"] is True
    assert "# Project Report" in body["markdown"]
