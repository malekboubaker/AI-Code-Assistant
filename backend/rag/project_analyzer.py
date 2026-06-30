from __future__ import annotations

import json
import re
import tomllib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from backend.rag.dependency_graph import build_dependency_graph, extract_routes, extract_structure
from backend.rag.metadata_extractor import infer_called_functions, is_config_file, is_doc_file, is_test_file
from backend.tools.language_detector import detect_language


MANIFEST_NAMES = {
    "package.json",
    "angular.json",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "cmakelists.txt",
    "cargo.toml",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
}


@dataclass
class FileSignal:
    relative_path: str
    language: str
    import_count: int = 0
    imported_by_count: int = 0
    symbol_count: int = 0
    is_entry_point: bool = False
    is_config_file: bool = False
    is_test_file: bool = False
    is_doc_file: bool = False
    dependency_neighbors: list[str] = field(default_factory=list)
    importance_score: float = 0.0


@dataclass
class ProjectAnalysis:
    project_path: str
    project_types: list[str]
    detected_languages: dict[str, int]
    detected_frameworks: list[str]
    entry_points: list[str]
    important_files: list[str]
    config_files: list[str]
    test_folders: list[str]
    source_folders: list[str]
    documentation_files: list[str]
    dependency_files: list[str]
    dependency_graph: dict[str, list[str]]
    reverse_dependency_count: dict[str, int]
    file_signals: dict[str, FileSignal]
    workspace_summary: str
    structures: dict[str, dict[str, Any]] = field(default_factory=dict)
    lines_of_code: dict[str, int] = field(default_factory=dict)
    graph: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["file_signals"] = {path: asdict(signal) for path, signal in self.file_signals.items()}
        return data


class ProjectAnalyzer:
    def analyze(self, project_path: str, files: list[Path]) -> ProjectAnalysis:
        root = Path(project_path).resolve()
        relative_files = [_relative_path(path, root) for path in files]
        file_set = set(relative_files)
        language_counts: Counter[str] = Counter()
        frameworks: set[str] = set()
        project_types: set[str] = set()
        config_files: list[str] = []
        documentation_files: list[str] = []
        dependency_files: list[str] = []
        test_folders: set[str] = set()
        folder_code_counts: Counter[str] = Counter()
        symbol_counts: dict[str, int] = {}
        dependency_graph: dict[str, list[str]] = {}
        explicit_entry_points: set[str] = set()
        structures: dict[str, dict[str, Any]] = {}
        calls_by_file: dict[str, list[str]] = {}
        routes_by_file: dict[str, list[dict[str, str]]] = {}
        lines_of_code: dict[str, int] = {}

        manifests: dict[str, list[Path]] = defaultdict(list)
        for relative in relative_files:
            manifests[Path(relative).name.lower()].append(root / relative)
        manifest_info = self._inspect_manifests(root, manifests)
        frameworks.update(manifest_info["frameworks"])
        project_types.update(manifest_info["project_types"])
        dependency_files.extend(manifest_info["dependency_files"])
        explicit_entry_points.update(manifest_info["entry_points"])

        for path, relative in zip(files, relative_files):
            language = detect_language(str(path))
            language_counts[language] += 1
            if language in {"python", "javascript", "typescript", "java", "cpp", "csharp", "rust"} and not is_test_file(path):
                folder_code_counts[str(Path(relative).parent).replace("\\", "/")] += 1
            if is_test_file(path):
                test_folders.add(str(Path(relative).parent).replace("\\", "/"))
            if is_config_file(path, language):
                config_files.append(relative)
            if is_doc_file(path, language):
                documentation_files.append(relative)
            if Path(relative).name.lower() in MANIFEST_NAMES:
                dependency_files.append(relative)
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                content = ""
            symbol_counts[relative] = _symbol_count(content, language)
            lines_of_code[relative] = len(content.splitlines())
            dependency_graph[relative] = self._resolve_imports(path, relative, content, language, root, file_set)
            frameworks.update(_frameworks_from_content(content, language))
            structures[relative] = extract_structure(content, language)
            calls_by_file[relative] = infer_called_functions(content, language)
            routes = extract_routes(content, language)
            if routes:
                routes_by_file[relative] = routes
            if _generic_entrypoint_signal(relative, content, language):
                explicit_entry_points.add(relative)

        project_types.update(_project_types_from_languages(language_counts))
        if _looks_mixed_project(project_types, language_counts):
            project_types.add("mixed full-stack project")
        if not project_types:
            project_types.add("unknown/generic project")

        reverse_counts: Counter[str] = Counter()
        for source, targets in dependency_graph.items():
            for target in targets:
                reverse_counts[target] += 1

        file_signals: dict[str, FileSignal] = {}
        for path, relative in zip(files, relative_files):
            language = detect_language(str(path))
            related = sorted(set(dependency_graph.get(relative, [])))
            score = _importance_score(
                relative=relative,
                import_count=len(related),
                imported_by_count=reverse_counts.get(relative, 0),
                symbol_count=symbol_counts.get(relative, 0),
                is_entry_point=relative in explicit_entry_points,
                is_config=relative in config_files or relative in dependency_files,
                is_doc=relative in documentation_files,
                is_test=is_test_file(path),
            )
            file_signals[relative] = FileSignal(
                relative_path=relative,
                language=language,
                import_count=len(related),
                imported_by_count=reverse_counts.get(relative, 0),
                symbol_count=symbol_counts.get(relative, 0),
                is_entry_point=relative in explicit_entry_points,
                is_config_file=relative in config_files or relative in dependency_files,
                is_test_file=is_test_file(path),
                is_doc_file=relative in documentation_files,
                dependency_neighbors=related + _reverse_neighbors(relative, dependency_graph),
                importance_score=score,
            )

        important_files = [
            relative
            for relative, _ in sorted(
                ((relative, signal.importance_score) for relative, signal in file_signals.items()),
                key=lambda item: item[1],
                reverse=True,
            )
            if file_signals[relative].importance_score > 0
        ][:80]
        source_folders = [folder for folder, _ in folder_code_counts.most_common(40)]

        graph = build_dependency_graph(
            files=relative_files,
            dependency_graph={key: sorted(set(value)) for key, value in dependency_graph.items() if value},
            reverse_counts=dict(reverse_counts),
            structures=structures,
            calls_by_file=calls_by_file,
            routes_by_file=routes_by_file,
            test_files={relative for relative, signal in file_signals.items() if signal.is_test_file},
            entry_points=_unique_sorted(explicit_entry_points),
            importance={relative: signal.importance_score for relative, signal in file_signals.items()},
            imported_by_counts={relative: signal.imported_by_count for relative, signal in file_signals.items()},
        )

        return ProjectAnalysis(
            project_path=str(root),
            project_types=list(project_types),
            detected_languages=dict(language_counts),
            detected_frameworks=list(frameworks),
            entry_points=list(_unique_sorted(explicit_entry_points)),
            important_files=list(important_files),
            config_files=_unique_limited(config_files, 80),
            test_folders=list(test_folders),
            source_folders=source_folders,
            documentation_files=_unique_limited(documentation_files, 60),
            dependency_files=_unique_limited(dependency_files, 60),
            dependency_graph={key: sorted(set(value)) for key, value in dependency_graph.items() if value},
            reverse_dependency_count=dict(reverse_counts),
            file_signals=file_signals,
            workspace_summary=_workspace_summary(project_types, frameworks, language_counts),
            structures=structures,
            lines_of_code=lines_of_code,
            graph=graph,
        )

    def _inspect_manifests(self, root: Path, manifests: dict[str, list[Path]]) -> dict[str, Any]:
        frameworks: set[str] = set()
        project_types: set[str] = set()
        dependency_files: list[str] = []
        entry_points: set[str] = set()

        for package_json in manifests.get("package.json", []):
            dependency_files.append(_relative_path(package_json, root))
            project_types.add("Node.js")
            try:
                package = json.loads(package_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                package = {}
            deps = {
                **package.get("dependencies", {}),
                **package.get("devDependencies", {}),
                **package.get("peerDependencies", {}),
            }
            dep_names = set(deps)
            if "react" in dep_names:
                frameworks.add("React")
                project_types.add("React")
            if "@angular/core" in dep_names or "angular.json" in manifests:
                frameworks.add("Angular")
                project_types.add("Angular")
            if "vue" in dep_names:
                frameworks.add("Vue")
                project_types.add("Vue")
            if "vite" in dep_names or any(name.startswith("vite.config") for name in manifests):
                frameworks.add("Vite")
            if "express" in dep_names:
                frameworks.add("Express")
            main = package.get("main")
            if isinstance(main, str):
                entry_points.add(_manifest_relative_ref(package_json, root, main))
            scripts = package.get("scripts", {})
            if isinstance(scripts, dict):
                for script in scripts.values():
                    entry_points.update(_manifest_relative_ref(package_json, root, ref) for ref in _script_file_refs(str(script)))

        for angular in manifests.get("angular.json", []):
            frameworks.add("Angular")
            project_types.add("Angular")
            dependency_files.append(_relative_path(angular, root))

        for name, paths in manifests.items():
            if name.startswith("vite.config"):
                frameworks.add("Vite")
                dependency_files.extend(_relative_path(path, root) for path in paths)

        pyprojects = manifests.get("pyproject.toml", [])
        requirements_files = manifests.get("requirements.txt", [])
        setup_files = manifests.get("setup.py", [])
        if pyprojects or requirements_files or setup_files:
            project_types.add("Python")
        dependency_files.extend(_relative_path(path, root) for path in [*pyprojects, *requirements_files, *setup_files])
        python_deps: set[str] = set()
        for path in pyprojects:
            python_deps.update(_python_dependencies(path, None, None))
        for path in requirements_files:
            python_deps.update(_python_dependencies(None, path, None))
        for path in setup_files:
            python_deps.update(_python_dependencies(None, None, path))
        if "fastapi" in python_deps:
            frameworks.add("FastAPI")
            project_types.add("FastAPI")
        if "flask" in python_deps:
            frameworks.add("Flask")
            project_types.add("Flask")
        if "django" in python_deps:
            frameworks.add("Django")
            project_types.add("Django")

        build_files = manifests.get("pom.xml", []) + manifests.get("build.gradle", []) + manifests.get("build.gradle.kts", [])
        if build_files:
            project_types.add("Java")
            for build_file in build_files:
                dependency_files.append(_relative_path(build_file, root))
                text = build_file.read_text(encoding="utf-8", errors="ignore").lower()
                if build_file.name.lower() == "pom.xml":
                    project_types.add("Java Maven")
                if "spring-boot" in text or "org.springframework" in text:
                    frameworks.add("Spring")
                    project_types.add("Java Spring")

        csproj_files = [path for name, paths in manifests.items() if name.endswith(".csproj") for path in paths]
        if csproj_files:
            project_types.add("C# .NET")
            dependency_files.extend(_relative_path(path, root) for path in csproj_files)

        for cmake in manifests.get("cmakelists.txt", []):
            project_types.add("C++")
            frameworks.add("CMake")
            dependency_files.append(_relative_path(cmake, root))

        for cargo in manifests.get("cargo.toml", []):
            project_types.add("Rust")
            dependency_files.append(_relative_path(cargo, root))
            try:
                data = tomllib.loads(cargo.read_text(encoding="utf-8"))
                bins = data.get("bin", [])
                if isinstance(bins, list):
                    for item in bins:
                        if isinstance(item, dict) and isinstance(item.get("path"), str):
                            entry_points.add(_manifest_relative_ref(cargo, root, item["path"]))
            except tomllib.TOMLDecodeError:
                pass

        for name in ("dockerfile", "docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"):
            for manifest in manifests.get(name, []):
                frameworks.add("Docker")
                dependency_files.append(_relative_path(manifest, root))

        return {
            "frameworks": frameworks,
            "project_types": project_types,
            "dependency_files": _unique_limited(dependency_files, 80),
            "entry_points": entry_points,
        }

    def _resolve_imports(
        self,
        path: Path,
        relative: str,
        content: str,
        language: str,
        root: Path,
        file_set: set[str],
    ) -> list[str]:
        refs = _import_references(content, language)
        resolved: set[str] = set()
        for ref in refs:
            target = _resolve_reference(path, ref, root, file_set, language)
            if target and target != relative:
                resolved.add(target)
        return sorted(resolved)


def _python_dependencies(pyproject: Path | None, requirements: Path | None, setup_py: Path | None) -> set[str]:
    deps: set[str] = set()
    if requirements and requirements.exists():
        for line in requirements.read_text(encoding="utf-8", errors="ignore").splitlines():
            name = re.split(r"[<>=~!;\[]", line.strip())[0].lower()
            if name and not name.startswith("#"):
                deps.add(name)
    if pyproject and pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            data = {}
        for dep in data.get("project", {}).get("dependencies", []) or []:
            deps.add(re.split(r"[<>=~!;\[]", str(dep).lower())[0])
        poetry = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
        deps.update(str(name).lower() for name in poetry if name.lower() != "python")
    if setup_py and setup_py.exists():
        text = setup_py.read_text(encoding="utf-8", errors="ignore").lower()
        for name in ("fastapi", "flask", "django"):
            if name in text:
                deps.add(name)
    return deps


def _frameworks_from_content(content: str, language: str) -> set[str]:
    lowered = content.lower()
    frameworks: set[str] = set()
    if language == "python":
        if "fastapi" in lowered:
            frameworks.add("FastAPI")
        if "flask" in lowered:
            frameworks.add("Flask")
        if "django" in lowered:
            frameworks.add("Django")
    if language in {"javascript", "typescript"}:
        if "from 'react'" in lowered or 'from "react"' in lowered:
            frameworks.add("React")
        if "from 'vue'" in lowered or 'from "vue"' in lowered:
            frameworks.add("Vue")
        if "express()" in lowered or "from 'express'" in lowered or 'from "express"' in lowered:
            frameworks.add("Express")
    if language == "java" and "org.springframework" in lowered:
        frameworks.add("Spring")
    return frameworks


def _project_types_from_languages(language_counts: Counter[str]) -> set[str]:
    mapping = {
        "python": "Python",
        "javascript": "Node.js",
        "typescript": "Node.js",
        "java": "Java",
        "csharp": "C# .NET",
        "cpp": "C++",
        "rust": "Rust",
    }
    return {mapping[language] for language in language_counts if language in mapping}


def _looks_mixed_project(project_types: set[str], language_counts: Counter[str]) -> bool:
    ecosystems = 0
    if {"Python", "FastAPI", "Flask", "Django"} & project_types:
        ecosystems += 1
    if {"Node.js", "React", "Angular", "Vue"} & project_types or language_counts.get("javascript") or language_counts.get("typescript"):
        ecosystems += 1
    if {"Java", "Java Spring", "C# .NET", "C++", "Rust"} & project_types:
        ecosystems += 1
    return ecosystems >= 2


def _generic_entrypoint_signal(relative: str, content: str, language: str) -> bool:
    name = Path(relative).name.lower()
    if name in {"main.py", "app.py", "server.py", "index.js", "index.ts", "main.ts", "main.rs", "program.cs"}:
        return True
    if language == "python" and re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", content):
        return True
    if language in {"javascript", "typescript"} and re.search(r"\b(listen|createServer)\s*\(", content):
        return True
    if language == "java" and "public static void main" in content:
        return True
    if language == "csharp" and ("static void main" in content.lower() or "webapplication.createbuilder" in content.lower()):
        return True
    if language == "cpp" and re.search(r"\bint\s+main\s*\(", content):
        return True
    if language == "rust" and re.search(r"\bfn\s+main\s*\(", content):
        return True
    return False


def _symbol_count(content: str, language: str) -> int:
    patterns = [
        r"\bclass\s+[A-Za-z_]\w*",
        r"\bdef\s+[A-Za-z_]\w*\s*\(",
        r"\bfunction\s+[A-Za-z_]\w*\s*\(",
        r"\bfn\s+[A-Za-z_]\w*\s*\(",
        r"\b(?:const|let|var)\s+[A-Za-z_]\w*\s*=\s*(?:async\s*)?\(",
        r"\b(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+[A-Za-z_]\w*\s*\(",
    ]
    return sum(len(re.findall(pattern, content)) for pattern in patterns)


def _importance_score(
    *,
    relative: str,
    import_count: int,
    imported_by_count: int,
    symbol_count: int,
    is_entry_point: bool,
    is_config: bool,
    is_doc: bool,
    is_test: bool,
) -> float:
    score = 0.0
    score += min(imported_by_count * 0.22, 1.2)
    score += min(import_count * 0.08, 0.5)
    score += min(symbol_count * 0.05, 0.5)
    if is_entry_point:
        score += 1.0
    if is_config:
        score += 0.35
    if is_doc:
        score += 0.2
    if is_test:
        score += 0.1
    if Path(relative).name.lower().startswith("readme"):
        score += 0.25
    return round(score, 4)


def _import_references(content: str, language: str) -> list[str]:
    refs: list[str] = []
    if language == "python":
        refs.extend(match.group(1) for match in re.finditer(r"^\s*import\s+([A-Za-z_][\w.]*)", content, re.MULTILINE))
        refs.extend(match.group(1) for match in re.finditer(r"^\s*from\s+([A-Za-z_.][\w.]*)\s+import", content, re.MULTILINE))
    elif language in {"javascript", "typescript"}:
        refs.extend(match.group(1) for match in re.finditer(r"from\s+['\"]([^'\"]+)['\"]", content))
        refs.extend(match.group(1) for match in re.finditer(r"require\(['\"]([^'\"]+)['\"]\)", content))
    elif language == "java":
        refs.extend(match.group(1).split(".")[-1] for match in re.finditer(r"^\s*import\s+([\w.]+);", content, re.MULTILINE))
    elif language == "cpp":
        refs.extend(match.group(1) for match in re.finditer(r"#include\s+[<\"]([^>\"]+)[>\"]", content))
    elif language == "csharp":
        refs.extend(match.group(1).split(".")[-1] for match in re.finditer(r"^\s*using\s+([\w.]+);", content, re.MULTILINE))
    elif language == "rust":
        refs.extend(match.group(1).replace("::", "/") for match in re.finditer(r"^\s*use\s+([\w:]+)", content, re.MULTILINE))
        refs.extend(match.group(1) for match in re.finditer(r"^\s*mod\s+([A-Za-z_]\w*)\s*;", content, re.MULTILINE))
    return list(dict.fromkeys(refs))[:100]


def _resolve_reference(path: Path, ref: str, root: Path, file_set: set[str], language: str) -> str | None:
    candidates: list[str] = []
    current_dir = path.parent
    if ref.startswith(".") or ref.startswith("/"):
        base = (current_dir / ref).resolve() if ref.startswith(".") else (root / ref.lstrip("/")).resolve()
        candidates.extend(_candidate_paths(_relative_path(base, root)))
    else:
        normalized = ref.replace(".", "/").replace("\\", "/")
        candidates.extend(_candidate_paths(normalized))
        candidates.extend(_candidate_paths(normalized.split("/")[-1]))
    for candidate in candidates:
        if candidate in file_set:
            return candidate
    lowered = ref.lower().split("/")[-1].split(".")[0]
    for candidate in file_set:
        stem = Path(candidate).stem.lower()
        if stem == lowered:
            return candidate
    return None


def _candidate_paths(base: str) -> list[str]:
    clean = base.strip("./").replace("\\", "/")
    suffixes = ["", ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".cs", ".cpp", ".hpp", ".h", ".rs", "/index.js", "/index.ts", "/mod.rs"]
    return [clean + suffix for suffix in suffixes if clean]


def _reverse_neighbors(relative: str, dependency_graph: dict[str, list[str]]) -> list[str]:
    return sorted(source for source, targets in dependency_graph.items() if relative in targets)


def _script_file_refs(script: str) -> set[str]:
    refs: set[str] = set()
    for token in re.findall(r"[\w./\\-]+\.(?:js|ts|py|mjs|cjs)", script):
        refs.add(_normalize_relative(token))
    return refs


def _manifest_relative_ref(manifest_path: Path, root: Path, value: str) -> str:
    normalized = _normalize_relative(value)
    manifest_dir = _relative_path(manifest_path.parent, root)
    if manifest_dir in {"", "."}:
        return normalized
    if normalized.startswith(f"{manifest_dir}/"):
        return normalized
    return f"{manifest_dir}/{normalized}"


def _workspace_summary(project_types: set[str], frameworks: set[str], language_counts: Counter[str]) -> str:
    project_label = ", ".join(sorted(project_types)) or "unknown/generic project"
    framework_label = ", ".join(sorted(frameworks)) or "none detected"
    language_label = ", ".join(f"{language}={count}" for language, count in language_counts.most_common()) or "unknown"
    return f"{project_label}. Frameworks: {framework_label}. Languages: {language_label}."


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _normalize_relative(value: str) -> str:
    return value.strip("./").replace("\\", "/")


def _unique_limited(items: list[str], limit: int) -> list[str]:
    return list(dict.fromkeys(items))[:limit]


def _unique_sorted(items: set[str]) -> list[str]:
    return sorted(item for item in items if item)
