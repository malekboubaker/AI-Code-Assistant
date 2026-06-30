from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Relationship types. Kept as plain string constants so the graph serializes to
# JSON and stays easy to extend for future features (impact analysis, change
# planning, architecture visualization, code navigation, doc generation).
# ---------------------------------------------------------------------------
FILE_IMPORTS_FILE = "FILE_IMPORTS_FILE"
FILE_DEPENDS_ON_FILE = "FILE_DEPENDS_ON_FILE"
CLASS_INHERITS_CLASS = "CLASS_INHERITS_CLASS"
CLASS_USES_CLASS = "CLASS_USES_CLASS"
FUNCTION_CALLS_FUNCTION = "FUNCTION_CALLS_FUNCTION"
FUNCTION_BELONGS_TO_FILE = "FUNCTION_BELONGS_TO_FILE"
TEST_COVERS_FILE = "TEST_COVERS_FILE"
TEST_COVERS_FUNCTION = "TEST_COVERS_FUNCTION"
ENTRYPOINT_STARTS_MODULE = "ENTRYPOINT_STARTS_MODULE"
MODULE_DEPENDS_ON_MODULE = "MODULE_DEPENDS_ON_MODULE"
API_ROUTE_CALLS_HANDLER = "API_ROUTE_CALLS_HANDLER"

MAX_EDGES = 4000
MAX_RELATED_FILES = 15
MAX_RELATED_TESTS = 10
MAX_RELATED_SYMBOLS = 20


def extract_structure(content: str, language: str) -> dict[str, Any]:
    """Return classes (with base classes/interfaces) and functions defined in a file."""
    classes: list[tuple[str, list[str]]] = []
    functions: list[str] = []
    if language == "python":
        try:
            tree = ast.parse(content)
        except SyntaxError:
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    bases = [name for name in (_ast_name(base) for base in node.bases) if name]
                    classes.append((node.name, bases))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.append(node.name)
            return {"classes": classes, "functions": list(dict.fromkeys(functions))}

    for match in re.finditer(
        r"\bclass\s+([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s*(?:extends\s+([A-Za-z_][\w.]*))?\s*(?:implements\s+([^\{]+))?",
        content,
    ):
        name = match.group(1)
        bases: list[str] = []
        if match.group(2):
            bases += _split_names(match.group(2))
        if match.group(3):
            bases.append(match.group(3).split(".")[-1])
        if match.group(4):
            bases += _split_names(match.group(4))
        classes.append((name, bases))
    for match in re.finditer(r"\b(?:struct|enum|trait|interface)\s+([A-Za-z_]\w*)", content):
        classes.append((match.group(1), []))
    for match in re.finditer(r"\bimpl\s+([A-Za-z_]\w*)\s+for\s+([A-Za-z_]\w*)", content):
        classes.append((match.group(2), [match.group(1)]))
    for match in re.finditer(
        r"\b(?:function\s+([A-Za-z_]\w*)|fn\s+([A-Za-z_]\w*)|def\s+([A-Za-z_]\w*))", content
    ):
        name = match.group(1) or match.group(2) or match.group(3)
        if name:
            functions.append(name)
    return {"classes": classes, "functions": list(dict.fromkeys(functions))}


def extract_routes(content: str, language: str) -> list[dict[str, str]]:
    """Best-effort detection of API route handlers (FastAPI/Flask, Express)."""
    routes: list[dict[str, str]] = []
    if language == "python":
        lines = content.splitlines()
        for index, line in enumerate(lines):
            match = re.search(r"@\w+\.(get|post|put|patch|delete|route)\(\s*['\"]([^'\"]+)['\"]", line)
            if not match:
                continue
            for ahead in lines[index + 1 : index + 5]:
                handler = re.search(r"\b(?:async\s+)?def\s+([A-Za-z_]\w*)", ahead)
                if handler:
                    routes.append({"method": match.group(1).upper(), "path": match.group(2), "handler": handler.group(1)})
                    break
    elif language in {"javascript", "typescript"}:
        for match in re.finditer(
            r"\b(?:app|router)\.(get|post|put|patch|delete)\(\s*['\"]([^'\"]+)['\"]\s*,\s*([A-Za-z_]\w*)",
            content,
        ):
            routes.append({"method": match.group(1).upper(), "path": match.group(2), "handler": match.group(3)})
    return routes


def build_dependency_graph(
    *,
    files: list[str],
    dependency_graph: dict[str, list[str]],
    reverse_counts: dict[str, int],
    structures: dict[str, dict[str, Any]],
    calls_by_file: dict[str, list[str]],
    routes_by_file: dict[str, list[dict[str, str]]],
    test_files: set[str],
    entry_points: list[str],
    importance: dict[str, float],
    imported_by_counts: dict[str, int],
) -> dict[str, Any]:
    """Build a typed dependency graph and per-file relation index from analysis data."""
    edges: list[list[str]] = []
    edge_counts: dict[str, int] = defaultdict(int)

    def add_edge(source: str, target: str, relation: str) -> None:
        if not source or not target or source == target:
            return
        edge_counts[relation] += 1
        if len(edges) < MAX_EDGES:
            edges.append([source, target, relation])

    # Symbol indexes for cross-file resolution.
    class_file: dict[str, str] = {}
    file_classes: dict[str, list[str]] = {}
    file_functions: dict[str, list[str]] = {}
    func_file: dict[str, str] = {}
    func_ambiguous: set[str] = set()
    for relative, struct in structures.items():
        names = []
        for class_name, _bases in struct.get("classes", []):
            class_file.setdefault(class_name, relative)
            names.append(class_name)
        file_classes[relative] = names
        functions = list(struct.get("functions", []))
        file_functions[relative] = functions
        for function in functions:
            if function in func_file and func_file[function] != relative:
                func_ambiguous.add(function)
            else:
                func_file.setdefault(function, relative)

    # FILE_IMPORTS_FILE + MODULE_DEPENDS_ON_MODULE
    imported_by: dict[str, list[str]] = defaultdict(list)
    module_deps: dict[str, set[str]] = defaultdict(set)
    for source, targets in dependency_graph.items():
        for target in targets:
            add_edge(source, target, FILE_IMPORTS_FILE)
            imported_by[target].append(source)
            source_module, target_module = _module_of(source), _module_of(target)
            if source_module != target_module:
                module_deps[source_module].add(target_module)
    for source_module, targets in module_deps.items():
        for target_module in targets:
            add_edge(source_module, target_module, MODULE_DEPENDS_ON_MODULE)

    # Inheritance / interface implementation.
    class_parents: dict[str, list[str]] = defaultdict(list)
    class_children: dict[str, list[str]] = defaultdict(list)
    file_parent_files: dict[str, set[str]] = defaultdict(set)
    file_child_files: dict[str, set[str]] = defaultdict(set)
    for relative, struct in structures.items():
        for class_name, bases in struct.get("classes", []):
            for base in bases:
                add_edge(class_name, base, CLASS_INHERITS_CLASS)
                class_parents[class_name].append(base)
                class_children[base].append(class_name)
                base_file = class_file.get(base)
                if base_file and base_file != relative:
                    file_parent_files[relative].add(base_file)
                    file_child_files[base_file].add(relative)

    # FUNCTION_BELONGS_TO_FILE
    for relative, functions in file_functions.items():
        for function in functions:
            add_edge(function, relative, FUNCTION_BELONGS_TO_FILE)

    # FUNCTION_CALLS_FUNCTION (file -> function), with call target files for expansion.
    call_target_files: dict[str, set[str]] = defaultdict(set)
    for relative, called in calls_by_file.items():
        for name in called:
            if name in func_ambiguous:
                continue
            target_file = func_file.get(name)
            if target_file and target_file != relative:
                add_edge(relative, name, FUNCTION_CALLS_FUNCTION)
                call_target_files[relative].add(target_file)

    # Tests.
    file_tests: dict[str, list[str]] = defaultdict(list)
    for relative in test_files:
        targets = _resolve_test_targets(relative, dependency_graph.get(relative, []), test_files, files)
        for target in targets:
            add_edge(relative, target, TEST_COVERS_FILE)
            file_tests[target].append(relative)
            for function in file_functions.get(target, []):
                if function in calls_by_file.get(relative, []):
                    add_edge(relative, function, TEST_COVERS_FUNCTION)

    # Entry points -> modules.
    for entry in entry_points:
        add_edge(entry, _module_of(entry), ENTRYPOINT_STARTS_MODULE)

    # API routes.
    api_routes: list[dict[str, str]] = []
    for relative, routes in routes_by_file.items():
        for route in routes:
            add_edge(f"{route['method']} {route['path']}", route["handler"], API_ROUTE_CALLS_HANDLER)
            api_routes.append({**route, "file": relative})

    # Per-file relation index used by the indexer + retrieval expansion.
    file_relations: dict[str, dict[str, Any]] = {}
    for relative in files:
        imports = list(dict.fromkeys(dependency_graph.get(relative, [])))
        importers = list(dict.fromkeys(imported_by.get(relative, [])))
        tests = list(dict.fromkeys(file_tests.get(relative, [])))
        parent_files = sorted(file_parent_files.get(relative, set()))
        child_files = sorted(file_child_files.get(relative, set()))
        call_files = sorted(call_target_files.get(relative, set()))
        related = _ordered_unique(
            imports + importers + tests + parent_files + child_files + call_files, exclude=relative
        )
        parents = sorted({base for name in file_classes.get(relative, []) for base in class_parents.get(name, [])})
        children = sorted({child for name in file_classes.get(relative, []) for child in class_children.get(name, [])})
        symbols = _ordered_unique(file_classes.get(relative, []) + file_functions.get(relative, []) + parents)
        file_relations[relative] = {
            "imports": imports[:MAX_RELATED_FILES],
            "imported_by": importers[:MAX_RELATED_FILES],
            "related_files": related[:MAX_RELATED_FILES],
            "related_tests": tests[:MAX_RELATED_TESTS],
            "related_symbols": symbols[:MAX_RELATED_SYMBOLS],
            "dependency_count": len(imports) + len(importers),
            "call_count": len(calls_by_file.get(relative, [])),
            "classes": file_classes.get(relative, []),
            "functions": file_functions.get(relative, [])[:30],
            "parent_classes": parents,
            "child_classes": children,
        }

    report = _build_report(files, dependency_graph, imported_by, importance, imported_by_counts, entry_points, edge_counts)

    return {
        "edges": edges,
        "edge_counts": dict(edge_counts),
        "file_relations": file_relations,
        "api_routes": api_routes[:80],
        "report": report,
    }


def related_files_from_payload(payload: dict[str, Any]) -> list[str]:
    """Collect graph-related files from a chunk payload (used for retrieval expansion)."""
    related: list[str] = []
    for key in ("related_files", "related_tests"):
        for value in payload.get(key) or []:
            normalized = str(value).replace("\\", "/")
            if normalized and normalized not in related:
                related.append(normalized)
    for value in payload.get("dependency_neighbors") or []:
        normalized = str(value).replace("\\", "/")
        if normalized and normalized not in related:
            related.append(normalized)
    return related


def _build_report(
    files: list[str],
    dependency_graph: dict[str, list[str]],
    imported_by: dict[str, list[str]],
    importance: dict[str, float],
    imported_by_counts: dict[str, int],
    entry_points: list[str],
    edge_counts: dict[str, int],
) -> dict[str, Any]:
    degree = {
        relative: len(dependency_graph.get(relative, [])) + len(imported_by.get(relative, []))
        for relative in files
    }
    module_degree: dict[str, int] = defaultdict(int)
    for relative, value in degree.items():
        module_degree[_module_of(relative)] += value
    most_connected_modules = [
        module for module, _ in sorted(module_degree.items(), key=lambda item: item[1], reverse=True) if module
    ][:8]
    critical_files = _top_files(
        files, key=lambda relative: (len(imported_by.get(relative, [])), importance.get(relative, 0.0))
    )
    hotspots = _top_files(files, key=lambda relative: (degree.get(relative, 0), importance.get(relative, 0.0)))
    return {
        "most_connected_modules": most_connected_modules,
        "critical_files": critical_files,
        "hotspots": hotspots,
        "entry_points": list(entry_points)[:8],
        "edge_counts": dict(edge_counts),
        "file_count": len(files),
        "edge_count": sum(edge_counts.values()),
    }


def _top_files(files: list[str], key, limit: int = 8) -> list[str]:
    ranked = sorted(files, key=key, reverse=True)
    result = [relative for relative in ranked if key(relative)[0] > 0]
    return result[:limit]


def _resolve_test_targets(
    test_relative: str, imports: list[str], test_files: set[str], files: list[str]
) -> list[str]:
    targets = [target for target in imports if target not in test_files]
    if targets:
        return list(dict.fromkeys(targets))
    stem = Path(test_relative).name.lower()
    stem = re.sub(r"\.(py|js|ts|tsx|jsx|java|cs|rs|cpp|cc)$", "", stem)
    stem = re.sub(r"(^test_|_test$|\.test$|\.spec$|^test)", "", stem)
    stem = stem.strip("_.")
    if not stem:
        return []
    matches = [
        candidate
        for candidate in files
        if candidate not in test_files and Path(candidate).stem.lower() == stem
    ]
    return matches[:3]


def _ordered_unique(items: list[str], exclude: str | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not item or item == exclude or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _module_of(relative: str) -> str:
    parent = str(Path(relative).parent).replace("\\", "/")
    return "(root)" if parent in {"", "."} else parent


def _ast_name(node: Any) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _ast_name(node.value)
    return None


def _split_names(text: str) -> list[str]:
    names: list[str] = []
    for raw in text.split(","):
        cleaned = re.sub(r"<.*?>", "", raw).strip()
        cleaned = cleaned.split("=")[-1].strip()
        cleaned = cleaned.split(".")[-1].strip()
        match = re.match(r"[A-Za-z_]\w*", cleaned)
        if match:
            names.append(match.group(0))
    return names
