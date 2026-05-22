from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.config.settings import ROOT_DIR
from backend.rag.metadata_extractor import is_config_file, is_doc_file, is_test_file
from backend.tools.language_detector import detect_language


PROJECT_MAP_PATH = ROOT_DIR / "data" / "metadata" / "project_map.json"


@dataclass
class ProjectMap:
    project_path: str
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
    last_indexed_time: str | None = None
    embedding_model: str | None = None
    collection_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_summary(self, max_items: int = 12) -> str:
        lines = [
            "Project map summary:",
            f"- Project path: {self.project_path}",
            f"- Languages: {_format_counts(self.detected_languages)}",
            f"- Main modules: {', '.join(self.main_modules[:max_items]) or 'unknown'}",
            f"- Entry points: {', '.join(self.entry_points[:max_items]) or 'unknown'}",
            f"- Tests: {', '.join(self.tests_folders[:max_items]) or 'unknown'}",
            f"- Config: {', '.join(self.config_files[:max_items]) or 'unknown'}",
        ]
        if self.readme_summary:
            lines.append(f"- README/docs: {self.readme_summary}")
        return "\n".join(lines)


def build_project_map(
    project_path: str,
    files: list[Path],
    *,
    skipped_files: int = 0,
    skipped_by_reason: dict[str, int] | None = None,
    last_indexed_time: str | None = None,
    embedding_model: str | None = None,
    collection_name: str | None = None,
) -> ProjectMap:
    root = Path(project_path).resolve()
    languages: Counter[str] = Counter()
    folders: Counter[str] = Counter()
    important_files: list[str] = []
    main_modules: list[str] = []
    entry_points: list[str] = []
    tests_folders: set[str] = set()
    config_files: list[str] = []
    documentation_files: list[str] = []

    for path in files:
        relative = _relative_path(path, root)
        language = detect_language(str(path))
        languages[language] += 1
        folder = str(Path(relative).parent).replace("\\", "/")
        folders[folder] += 1
        if is_test_file(path):
            tests_folders.add(folder)
        if is_config_file(path, language):
            config_files.append(relative)
        if is_doc_file(path, language):
            documentation_files.append(relative)
        if _is_important_file(path, language):
            important_files.append(relative)
        if _is_main_module(path, relative):
            main_modules.append(relative)
        if _is_entry_point(path, relative):
            entry_points.append(relative)

    folder_structure = [folder for folder, _ in folders.most_common(80)]
    readme_summary = _readme_summary(root, documentation_files)

    return ProjectMap(
        project_path=str(root),
        folder_structure=folder_structure,
        important_files=_unique_limited(important_files, 80),
        detected_languages=dict(languages.most_common()),
        main_modules=_unique_limited(main_modules, 40),
        entry_points=_unique_limited(entry_points, 40),
        tests_folders=sorted(tests_folders)[:40],
        config_files=_unique_limited(config_files, 60),
        documentation_files=_unique_limited(documentation_files, 40),
        readme_summary=readme_summary,
        files_scanned=len(files),
        files_skipped=skipped_files,
        skipped_by_reason=skipped_by_reason or {},
        last_indexed_time=last_indexed_time,
        embedding_model=embedding_model,
        collection_name=collection_name,
    )


def save_project_map(project_map: ProjectMap, path: Path = PROJECT_MAP_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(project_map.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_project_map(path: Path = PROJECT_MAP_PATH) -> ProjectMap | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ProjectMap(**data)


def project_map_payload(project_map: ProjectMap) -> dict[str, Any]:
    return {
        "content": project_map.to_summary(),
        "language": "text",
        "file_path": "project_map.json",
        "relative_path": "project_map.json",
        "start_line": 1,
        "end_line": len(project_map.to_summary().splitlines()),
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


def _is_important_file(path: Path, language: str) -> bool:
    name = path.name.lower()
    return (
        name.startswith("readme")
        or name in {"package.json", "pyproject.toml", "cargo.toml", "dockerfile", "docker-compose.yml", "docker-compose.yaml"}
        or is_config_file(path, language)
        or _is_entry_point(path, str(path))
    )


def _is_main_module(path: Path, relative: str) -> bool:
    name = path.name.lower()
    parts = Path(relative).parts
    return (
        name in {"main.py", "app.py", "server.py", "index.js", "index.ts", "main.rs", "program.cs"}
        or "src" in parts
        or "backend" in parts
    )


def _is_entry_point(path: Path, relative: str) -> bool:
    name = path.name.lower()
    return name in {
        "main.py",
        "app.py",
        "server.py",
        "index.js",
        "index.ts",
        "main.ts",
        "main.rs",
        "program.cs",
        "dockerfile",
        "package.json",
    } or relative.endswith("/main.py")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return path.name


def _unique_limited(items: list[str], limit: int) -> list[str]:
    return list(dict.fromkeys(items))[:limit]


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{language}={count}" for language, count in counts.items()) or "unknown"
