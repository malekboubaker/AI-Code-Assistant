from pathlib import Path

from backend.rag.project_analyzer import ProjectAnalyzer
from backend.rag.project_map import build_project_map
from backend.rag.retriever import Retriever
from backend.rag.embedder import LocalEmbedder


def analyze_tmp(tmp_path: Path):
    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    return ProjectAnalyzer().analyze(str(tmp_path), files), files


def test_python_project_detection(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["fastapi"]\n', encoding="utf-8")
    (tmp_path / "app.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n", encoding="utf-8")

    analysis, _ = analyze_tmp(tmp_path)

    assert "Python" in analysis.project_types
    assert "FastAPI" in analysis.project_types
    assert "FastAPI" in analysis.detected_frameworks
    assert "app.py" in analysis.entry_points


def test_node_react_project_detection(tmp_path: Path):
    (tmp_path / "package.json").write_text(
        '{"dependencies":{"react":"latest","vite":"latest"},"scripts":{"start":"vite --host 0.0.0.0"}}',
        encoding="utf-8",
    )
    (tmp_path / "main.tsx").write_text("import React from 'react';\n", encoding="utf-8")

    analysis, _ = analyze_tmp(tmp_path)

    assert "Node.js" in analysis.project_types
    assert "React" in analysis.project_types
    assert "React" in analysis.detected_frameworks
    assert "Vite" in analysis.detected_frameworks


def test_java_maven_project_detection(tmp_path: Path):
    (tmp_path / "pom.xml").write_text("<project><dependencies></dependencies></project>", encoding="utf-8")
    (tmp_path / "Main.java").write_text("public class Main { public static void main(String[] args) {} }", encoding="utf-8")

    analysis, _ = analyze_tmp(tmp_path)

    assert "Java" in analysis.project_types
    assert "Java Maven" in analysis.project_types


def test_csharp_project_detection(tmp_path: Path):
    (tmp_path / "App.csproj").write_text("<Project Sdk=\"Microsoft.NET.Sdk\"></Project>", encoding="utf-8")
    (tmp_path / "Program.cs").write_text("var builder = WebApplication.CreateBuilder(args);", encoding="utf-8")

    analysis, _ = analyze_tmp(tmp_path)

    assert "C# .NET" in analysis.project_types
    assert "Program.cs" in analysis.entry_points


def test_rust_project_detection(tmp_path: Path):
    (tmp_path / "Cargo.toml").write_text("[package]\nname='demo'\nversion='0.1.0'\n", encoding="utf-8")
    (tmp_path / "main.rs").write_text("fn main() {}\n", encoding="utf-8")

    analysis, _ = analyze_tmp(tmp_path)

    assert "Rust" in analysis.project_types
    assert "main.rs" in analysis.entry_points


def test_unknown_project_fallback(tmp_path: Path):
    (tmp_path / "README.md").write_text("# Notes\nNo obvious framework.\n", encoding="utf-8")

    analysis, _ = analyze_tmp(tmp_path)

    assert "unknown/generic project" in analysis.project_types
    assert analysis.workspace_summary


def test_mixed_project_detection_with_nested_manifests(tmp_path: Path):
    api = tmp_path / "api"
    web = tmp_path / "web"
    api.mkdir()
    web.mkdir()
    (api / "pyproject.toml").write_text('[project]\ndependencies = ["fastapi"]\n', encoding="utf-8")
    (api / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n", encoding="utf-8")
    (web / "package.json").write_text(
        '{"main":"src/main.tsx","dependencies":{"react":"latest","vite":"latest"}}',
        encoding="utf-8",
    )
    (web / "src").mkdir()
    (web / "src" / "main.tsx").write_text("import React from 'react';\n", encoding="utf-8")

    analysis, _ = analyze_tmp(tmp_path)

    assert "mixed full-stack project" in analysis.project_types
    assert "FastAPI" in analysis.detected_frameworks
    assert "React" in analysis.detected_frameworks
    assert "web/src/main.tsx" in analysis.entry_points


def test_project_map_creation_uses_analysis(tmp_path: Path):
    (tmp_path / "package.json").write_text('{"dependencies":{"vue":"latest"}}', encoding="utf-8")
    (tmp_path / "widget.js").write_text("export function widget() { return true; }", encoding="utf-8")
    analysis, files = analyze_tmp(tmp_path)

    project_map = build_project_map(str(tmp_path), files, analysis=analysis)

    assert "Vue" in project_map.detected_frameworks
    assert "Node.js" in project_map.project_types
    assert project_map.workspace_summary
    assert "package.json" in project_map.dependency_files


def test_retrieval_boosts_important_files_without_fixed_folder_names():
    retriever = Retriever(embedder=LocalEmbedder(vector_size=16), store=object())
    hits = [
        {
            "score": 0.50,
            "payload": {
                "content": "def helper(): pass",
                "language": "python",
                "file_path": "src/helper.py",
                "relative_path": "src/helper.py",
                "importance_score": 0.0,
            },
        },
        {
            "score": 0.50,
            "payload": {
                "content": "def helper(): pass",
                "language": "python",
                "file_path": "banana_tree/central_logic.py",
                "relative_path": "banana_tree/central_logic.py",
                "importance_score": 2.0,
                "imported_by_count": 4,
            },
        },
    ]

    ranked = retriever._rank_hits("helper", hits, language="python")

    assert ranked[0]["payload"]["relative_path"] == "banana_tree/central_logic.py"
