from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from backend.tools.language_detector import detect_language


IMPORT_PATTERNS = {
    "python": re.compile(r"^\s*(?:import|from)\s+.+", re.MULTILINE),
    "javascript": re.compile(r"^\s*import\s+.+|^\s*const\s+.+require\(.+", re.MULTILINE),
    "typescript": re.compile(r"^\s*import\s+.+|^\s*const\s+.+require\(.+", re.MULTILINE),
    "java": re.compile(r"^\s*import\s+.+;", re.MULTILINE),
    "cpp": re.compile(r"^\s*#include\s+.+", re.MULTILINE),
    "csharp": re.compile(r"^\s*using\s+.+;", re.MULTILINE),
    "rust": re.compile(r"^\s*(?:use|extern crate)\s+.+", re.MULTILINE),
}


def extract_metadata(
    path: Path,
    content: str,
    start_line: int,
    end_line: int,
    chunk_type: str,
    *,
    project_root: Path | None = None,
    symbol_name: str | None = None,
    parent_scope: str | None = None,
    file_imports: list[str] | None = None,
) -> dict[str, Any]:
    language = detect_language(str(path), content)
    imports = file_imports or [match.group(0).strip() for match in IMPORT_PATTERNS.get(language, re.compile("$^")).finditer(content)]
    symbol_name = symbol_name or infer_symbol_name(content, language)
    relative_path = _relative_path(path, project_root)
    return {
        "language": language,
        "file_path": str(path),
        "relative_path": relative_path,
        "start_line": start_line,
        "end_line": end_line,
        "chunk_type": chunk_type,
        "symbol_name": symbol_name,
        "parent_scope": parent_scope,
        "imports": imports[:20],
        "called_functions": infer_called_functions(content, language)[:50],
        "folder": str(Path(relative_path).parent).replace("\\", "/") if relative_path else str(path.parent),
        "is_test_file": is_test_file(path),
        "is_config_file": is_config_file(path, language),
        "is_doc_file": is_doc_file(path, language),
        "task_tags": ["code_gen", "bug_fix", "bug_detection", "refactoring"],
        "source": "project_code",
        "validated": True,
        "created_by": "indexer",
    }


def infer_symbol_name(content: str, language: str) -> str | None:
    patterns = [
        r"class\s+([A-Za-z_]\w*)",
        r"def\s+([A-Za-z_]\w*)\s*\(",
        r"fn\s+([A-Za-z_]\w*)\s*\(",
        r"function\s+([A-Za-z_]\w*)\s*\(",
        r"(?:export\s+)?(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\(",
        r"(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+([A-Za-z_]\w*)\s*\(",
    ]
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            return match.group(1)
    return None


def infer_called_functions(content: str, language: str) -> list[str]:
    if language == "python":
        try:
            import ast

            tree = ast.parse(content)
            names: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = _python_call_name(node.func)
                    if name:
                        names.append(name)
            return list(dict.fromkeys(names))
        except SyntaxError:
            pass
    matches = re.findall(r"\b([A-Za-z_]\w*)\s*\(", content)
    excluded = {
        "if",
        "for",
        "while",
        "switch",
        "catch",
        "function",
        "return",
        "class",
        "def",
        "fn",
        "new",
    }
    return list(dict.fromkeys(name for name in matches if name not in excluded))


def is_test_file(path: Path) -> bool:
    normalized = str(path).replace("\\", "/").lower()
    name = path.name.lower()
    return (
        "/test/" in normalized
        or "/tests/" in normalized
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".test.js")
        or name.endswith(".spec.js")
        or name.endswith(".test.ts")
        or name.endswith(".spec.ts")
    )


def is_config_file(path: Path, language: str | None = None) -> bool:
    name = path.name.lower()
    suffix = path.suffix.lower()
    return (language == "config") or (language in {"yaml", "json", "dockerfile"} and name != "readme.md") or suffix in {
        ".toml",
        ".ini",
        ".cfg",
        ".conf",
        ".env",
        ".properties",
        ".gradle",
        ".xml",
    }


def is_doc_file(path: Path, language: str | None = None) -> bool:
    name = path.name.lower()
    return language == "markdown" or name.startswith("readme") or "/docs/" in str(path).replace("\\", "/").lower()


def _python_call_name(node: Any) -> str | None:
    if hasattr(node, "id"):
        return str(node.id)
    if hasattr(node, "attr"):
        return str(node.attr)
    return None


def _relative_path(path: Path, project_root: Path | None) -> str:
    if project_root is None:
        return path.name
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        return path.name
