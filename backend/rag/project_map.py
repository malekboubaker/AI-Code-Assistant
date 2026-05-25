from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.config.settings import ROOT_DIR
from backend.rag.project_identity import normalize_project_path, project_id_for_path, project_name_for_path
from backend.rag.project_analyzer import ProjectAnalysis, ProjectAnalyzer
from backend.tools.language_detector import detect_language


PROJECT_MAP_PATH = ROOT_DIR / "data" / "metadata" / "project_map.json"


@dataclass
class ProjectMap:
    project_id: str
    project_name: str
    project_path: str
    workspace_root: str
    folder_structure: list[str]
    important_files: list[str]
    detected_languages: dict[str, int]
    main_modules: list[str]
    entry_points: list[str]
    tests_folders: list[str]
    config_files: list[str]
    documentation_files: list[str]
    readme_summary: str
    files_scanned: int
    files_skipped: int
    skipped_by_reason: dict[str, int]
    project_types: list[str] = field(default_factory=list)
    detected_frameworks: list[str] = field(default_factory=list)
    source_folders: list[str] = field(default_factory=list)
    dependency_files: list[str] = field(default_factory=list)
    workspace_summary: str = ""
    dependency_graph: dict[str, list[str]] = field(default_factory=dict)
    skipped_folders: dict[str, int] = field(default_factory=dict)
    last_indexed_time: str | None = None
    embedding_model: str | None = None
    collection_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_summary(self, max_items: int = 12) -> str:
        lines = [
            "Project map summary:",
            f"- Project name: {self.project_name}",
            f"- Project id: {self.project_id}",
            f"- Project path: {self.project_path}",
            f"- Project type: {', '.join(self.project_types[:max_items]) or 'unknown/generic project'}",
            f"- Languages: {_format_counts(self.detected_languages)}",
            f"- Frameworks: {', '.join(self.detected_frameworks[:max_items]) or 'none detected'}",
            f"- Source folders: {', '.join(self.source_folders[:max_items]) or 'unknown'}",
            f"- Entry points: {', '.join(self.entry_points[:max_items]) or 'unknown'}",
            f"- Important files: {', '.join(self.important_files[:max_items]) or 'unknown'}",
            f"- Dependency files: {', '.join(self.dependency_files[:max_items]) or 'unknown'}",
            f"- Tests: {', '.join(self.tests_folders[:max_items]) or 'unknown'}",
            f"- Config: {', '.join(self.config_files[:max_items]) or 'unknown'}",
        ]
        if self.workspace_summary:
            lines.append(f"- Workspace: {self.workspace_summary}")
        if self.readme_summary:
            lines.append(f"- README/docs: {self.readme_summary}")
        return "\n".join(lines)


def build_project_map(
    project_path: str,
    files: list[Path],
    *,
    skipped_files: int = 0,
    skipped_by_reason: dict[str, int] | None = None,
    skipped_folders: dict[str, int] | None = None,
    last_indexed_time: str | None = None,
    embedding_model: str | None = None,
    collection_name: str | None = None,
    analysis: ProjectAnalysis | None = None,
) -> ProjectMap:
    root = Path(project_path).resolve()
    analysis = analysis or ProjectAnalyzer().analyze(str(root), files)
    languages: Counter[str] = Counter()
    folders: Counter[str] = Counter()

    for path in files:
        relative = _relative_path(path, root)
        language = detect_language(str(path))
        languages[language] += 1
        folder = str(Path(relative).parent).replace("\\", "/")
        folders[folder] += 1

    folder_structure = _unique_limited(analysis.source_folders + [folder for folder, _ in folders.most_common(80)], 80)
    readme_summary = _readme_summary(root, analysis.documentation_files)

    return ProjectMap(
        project_id=project_id_for_path(root),
        project_name=project_name_for_path(root),
        project_path=normalize_project_path(root),
        workspace_root=normalize_project_path(root),
        folder_structure=folder_structure,
        important_files=analysis.important_files,
        detected_languages=dict(languages.most_common()),
        main_modules=analysis.important_files[:40],
        entry_points=analysis.entry_points,
        tests_folders=analysis.test_folders,
        config_files=analysis.config_files,
        documentation_files=analysis.documentation_files,
        readme_summary=readme_summary,
        files_scanned=len(files),
        files_skipped=skipped_files,
        skipped_by_reason=skipped_by_reason or {},
        skipped_folders=skipped_folders or {},
        project_types=analysis.project_types,
        detected_frameworks=analysis.detected_frameworks,
        source_folders=analysis.source_folders,
        dependency_files=analysis.dependency_files,
        workspace_summary=analysis.workspace_summary,
        dependency_graph=analysis.dependency_graph,
        last_indexed_time=last_indexed_time,
        embedding_model=embedding_model,
        collection_name=collection_name,
        extra={"file_signals": analysis.to_dict().get("file_signals", {})},
    )


def save_project_map(project_map: ProjectMap, path: Path = PROJECT_MAP_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(project_map.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_project_map(path: Path = PROJECT_MAP_PATH) -> ProjectMap | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    project_path = data.get("project_path") or data.get("workspace_root") or ""
    if project_path:
        data.setdefault("project_id", project_id_for_path(project_path))
        data.setdefault("project_name", project_name_for_path(project_path))
        data.setdefault("workspace_root", str(Path(project_path).resolve()))
    return ProjectMap(**data)


def project_map_payload(project_map: ProjectMap) -> dict[str, Any]:
    relative_file_path = "project_map.json"
    summary = project_map.to_summary()
    return {
        "content": summary,
        "summary": summary,
        "language": "text",
        "project_id": project_map.project_id,
        "project_name": project_map.project_name,
        "project_path": project_map.project_path,
        "workspace_root": project_map.workspace_root,
        "detected_languages": project_map.detected_languages,
        "frameworks": project_map.detected_frameworks,
        "detected_frameworks": project_map.detected_frameworks,
        "project_types": project_map.project_types,
        "entry_points": project_map.entry_points,
        "important_files": project_map.important_files,
        "indexed_folders": project_map.folder_structure,
        "source_folders": project_map.source_folders,
        "file_path": str(Path(project_map.project_path) / relative_file_path),
        "relative_path": relative_file_path,
        "relative_file_path": relative_file_path,
        "start_line": 1,
        "end_line": len(summary.splitlines()),
        "chunk_type": "project_map",
        "symbol_name": "project_map",
        "parent_scope": None,
        "imports": [],
        "called_functions": [],
        "folder": ".",
        "is_test_file": False,
        "is_config_file": True,
        "is_doc_file": True,
        "source": "project_map",
        "validated": True,
        "created_by": "indexer",
        "project_map": project_map.to_dict(),
    }


def _readme_summary(root: Path, documentation_files: list[str]) -> str:
    readmes = [path for path in documentation_files if Path(path).name.lower().startswith("readme")]
    candidates = readmes or documentation_files[:3]
    snippets: list[str] = []
    for relative in candidates[:3]:
        path = root / relative
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        snippets.append(_summarize_text(text))
    return " ".join(snippet for snippet in snippets if snippet)[:1200]


def _summarize_text(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            lines.append(line.lstrip("#").strip())
        elif len(lines) < 6:
            lines.append(line)
        if len(lines) >= 8:
            break
    return " ".join(lines)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return path.name


def _unique_limited(items: list[str], limit: int) -> list[str]:
    return list(dict.fromkeys(items))[:limit]


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{language}={count}" for language, count in counts.items()) or "unknown"
