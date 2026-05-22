from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from backend.rag.metadata_extractor import extract_metadata
from backend.tools.language_detector import detect_language


@dataclass
class CodeChunk:
    content: str
    payload: dict


@dataclass
class ChunkCandidate:
    content: str
    start_line: int
    end_line: int
    chunk_type: str
    symbol_name: str | None = None
    parent_scope: str | None = None


MAX_CHUNK_LINES = 120
WINDOW_OVERLAP_LINES = 12


def chunk_file(
    path: Path,
    max_lines: int = MAX_CHUNK_LINES,
    overlap: int = WINDOW_OVERLAP_LINES,
    project_root: Path | None = None,
) -> list[CodeChunk]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    language = detect_language(str(path), content)
    file_imports = extract_file_imports(content, language)
    candidates = _symbol_chunks(content, language, max_lines=max_lines, overlap=overlap)
    if not candidates:
        chunk_type = "doc" if language == "markdown" else "config" if language in {"yaml", "json", "dockerfile", "config"} else "window"
        candidates = _line_window_chunks(content, max_lines=max_lines, overlap=overlap, chunk_type=chunk_type)

    result: list[CodeChunk] = []
    for candidate in candidates:
        payload = extract_metadata(
            path,
            candidate.content,
            candidate.start_line,
            candidate.end_line,
            candidate.chunk_type,
            project_root=project_root,
            symbol_name=candidate.symbol_name,
            parent_scope=candidate.parent_scope,
            file_imports=file_imports,
        )
        payload["content"] = candidate.content
        result.append(CodeChunk(content=candidate.content, payload=payload))
    return result


def _symbol_chunks(content: str, language: str, max_lines: int, overlap: int) -> list[ChunkCandidate]:
    if language == "python":
        return _python_chunks(content, max_lines=max_lines, overlap=overlap)
    if language in {"javascript", "typescript", "java", "cpp", "csharp", "rust"}:
        return _brace_symbol_chunks(content, language, max_lines=max_lines, overlap=overlap)
    if language == "markdown":
        return _markdown_chunks(content, max_lines=max_lines, overlap=overlap)
    return []


def _python_chunks(content: str, max_lines: int, overlap: int) -> list[ChunkCandidate]:
    try:
        import ast

        tree = ast.parse(content)
    except SyntaxError:
        return []

    lines = content.splitlines()
    candidates: list[ChunkCandidate] = []

    def visit(body: list, parent_scope: str | None = None) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno
                end = getattr(node, "end_lineno", start)
                chunk_type = "class" if isinstance(node, ast.ClassDef) else "function"
                symbol_name = node.name
                line_count = end - start + 1
                if line_count <= max_lines:
                    candidates.append(
                        ChunkCandidate(
                            content="\n".join(lines[start - 1 : end]),
                            start_line=start,
                            end_line=end,
                            chunk_type=chunk_type,
                            symbol_name=symbol_name,
                            parent_scope=parent_scope,
                        )
                    )
                else:
                    if isinstance(node, ast.ClassDef):
                        visit(node.body, parent_scope=symbol_name)
                    else:
                        for window in _line_window_chunks(
                            "\n".join(lines[start - 1 : end]),
                            max_lines=max_lines,
                            overlap=overlap,
                            chunk_type="function_window",
                            line_offset=start - 1,
                        ):
                            window.symbol_name = symbol_name
                            window.parent_scope = parent_scope
                            candidates.append(window)

    visit(tree.body)
    return sorted(candidates, key=lambda chunk: (chunk.start_line, chunk.end_line))


def _brace_symbol_chunks(content: str, language: str, max_lines: int, overlap: int) -> list[ChunkCandidate]:
    lines = content.splitlines()
    candidates: list[ChunkCandidate] = []
    patterns = _symbol_patterns(language)
    index = 0
    while index < len(lines):
        line = lines[index]
        match = None
        chunk_type = "symbol"
        for candidate_type, pattern in patterns:
            match = pattern.search(line)
            if match:
                chunk_type = candidate_type
                break
        if not match:
            index += 1
            continue

        start_index = index
        symbol_name = match.group(1) if match.groups() else None
        end_index = _find_brace_block_end(lines, start_index)
        if end_index is None:
            index += 1
            continue
        raw = "\n".join(lines[start_index : end_index + 1])
        if end_index - start_index + 1 <= max_lines:
            candidates.append(
                ChunkCandidate(
                    content=raw,
                    start_line=start_index + 1,
                    end_line=end_index + 1,
                    chunk_type=chunk_type,
                    symbol_name=symbol_name,
                )
            )
        else:
            for window in _line_window_chunks(
                raw,
                max_lines=max_lines,
                overlap=overlap,
                chunk_type=f"{chunk_type}_window",
                line_offset=start_index,
            ):
                window.symbol_name = symbol_name
                candidates.append(window)
        index = end_index + 1
    return candidates


def _markdown_chunks(content: str, max_lines: int, overlap: int) -> list[ChunkCandidate]:
    lines = content.splitlines()
    if not lines:
        return []
    heading_indexes = [index for index, line in enumerate(lines) if line.startswith("#")]
    if not heading_indexes:
        return _line_window_chunks(content, max_lines=max_lines, overlap=overlap, chunk_type="doc")
    candidates: list[ChunkCandidate] = []
    for pos, start_index in enumerate(heading_indexes):
        end_index = heading_indexes[pos + 1] - 1 if pos + 1 < len(heading_indexes) else len(lines) - 1
        title = lines[start_index].lstrip("#").strip() or "documentation"
        section = "\n".join(lines[start_index : end_index + 1])
        if end_index - start_index + 1 <= max_lines:
            candidates.append(
                ChunkCandidate(
                    content=section,
                    start_line=start_index + 1,
                    end_line=end_index + 1,
                    chunk_type="doc_section",
                    symbol_name=title[:80],
                )
            )
        else:
            candidates.extend(
                _line_window_chunks(
                    section,
                    max_lines=max_lines,
                    overlap=overlap,
                    chunk_type="doc_window",
                    line_offset=start_index,
                )
            )
    return candidates


def _line_window_chunks(
    content: str,
    max_lines: int,
    overlap: int,
    chunk_type: str,
    line_offset: int = 0,
) -> list[ChunkCandidate]:
    lines = content.splitlines()
    if not lines:
        return []
    chunks: list[ChunkCandidate] = []
    step = max(1, max_lines - overlap)
    for start_index in range(0, len(lines), step):
        end_index = min(len(lines), start_index + max_lines)
        chunks.append(
            ChunkCandidate(
                content="\n".join(lines[start_index:end_index]),
                start_line=line_offset + start_index + 1,
                end_line=line_offset + end_index,
                chunk_type=chunk_type,
            )
        )
        if end_index == len(lines):
            break
    return chunks


def _symbol_patterns(language: str) -> list[tuple[str, re.Pattern[str]]]:
    common = [
        ("class", re.compile(r"\bclass\s+([A-Za-z_]\w*)")),
        ("function", re.compile(r"\bfunction\s+([A-Za-z_]\w*)\s*\(")),
        ("function", re.compile(r"\b(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?\(")),
    ]
    if language == "rust":
        return [
            ("function", re.compile(r"\bfn\s+([A-Za-z_]\w*)\s*\(")),
            ("class", re.compile(r"\b(?:struct|enum|trait|impl)\s+([A-Za-z_]\w*)")),
        ]
    if language in {"java", "cpp", "csharp"}:
        return [
            ("class", re.compile(r"\b(?:class|interface|struct|enum)\s+([A-Za-z_]\w*)")),
            ("function", re.compile(r"\b(?:public|private|protected|static|virtual|async|final|override|\s)*[\w:<>,~*&\[\]]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:const\s*)?\{")),
        ]
    return common


def _find_brace_block_end(lines: list[str], start_index: int) -> int | None:
    depth = 0
    seen_open = False
    for index in range(start_index, len(lines)):
        stripped = _strip_line_literals(lines[index])
        depth += stripped.count("{")
        if "{" in stripped:
            seen_open = True
        depth -= stripped.count("}")
        if seen_open and depth <= 0:
            return index
    return None


def _strip_line_literals(line: str) -> str:
    line = re.sub(r"//.*", "", line)
    line = re.sub(r"#.*", "", line)
    line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
    line = re.sub(r"'(?:\\.|[^'\\])*'", "''", line)
    return line


def extract_file_imports(content: str, language: str) -> list[str]:
    from backend.rag.metadata_extractor import IMPORT_PATTERNS

    pattern = IMPORT_PATTERNS.get(language)
    if pattern is None:
        return []
    return [match.group(0).strip() for match in pattern.finditer(content)][:20]
