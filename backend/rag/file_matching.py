from __future__ import annotations

import re
from dataclasses import dataclass

# Detect explicit file references such as a2a_client.py, src/foo/bar.ts, ./pkg/mod.rs.
# A reference is only "explicit" when it carries a known source/config extension; bare
# module names without an extension stay in the semantic-retrieval path on purpose.
FILE_REFERENCE_RE = re.compile(
    r"(?:[A-Za-z0-9_.\-]+[/\\])*[A-Za-z0-9_.\-]+\."
    r"(?:py|pyi|js|jsx|mjs|cjs|ts|tsx|java|cpp|cc|cxx|hpp|hh|h|cs|rs|go|rb|php|kt|swift|scala"
    r"|md|json|ya?ml|toml|cfg|ini|xml|gradle)"
    r"\b"
)


def extract_file_references(text: str) -> list[str]:
    """Return unique, normalized file references explicitly mentioned in ``text``."""
    if not text:
        return []
    references: list[str] = []
    for match in FILE_REFERENCE_RE.findall(text):
        cleaned = match.replace("\\", "/").lstrip("./").strip()
        if cleaned:
            references.append(cleaned)
    return list(dict.fromkeys(references))


def matches_file_reference(payload: dict, references: list[str]) -> str | None:
    """Return the reference that exactly identifies ``payload``'s file, else ``None``.

    Matching is intentionally strict so that similarly named files (for example
    ``a2a_client.py`` versus ``a2a_server.py``) never cross-match: a bare filename
    must equal the candidate basename, and a path-qualified reference must be a
    real path suffix of the candidate.
    """
    if not references:
        return None
    candidates = _payload_path_candidates(payload)
    if not candidates:
        return None
    for reference in references:
        ref_norm = reference.replace("\\", "/").lower().strip("/")
        if not ref_norm:
            continue
        ref_base = ref_norm.rsplit("/", 1)[-1]
        ref_has_dir = "/" in ref_norm
        for candidate in candidates:
            cand_norm = candidate.replace("\\", "/").lower().strip("/")
            if not cand_norm:
                continue
            cand_base = cand_norm.rsplit("/", 1)[-1]
            if cand_norm == ref_norm:
                return reference
            if ref_has_dir:
                if cand_norm.endswith("/" + ref_norm):
                    return reference
            elif cand_base == ref_base:
                return reference
    return None


def _payload_path_candidates(payload: dict) -> list[str]:
    candidates: list[str] = []
    for key in ("relative_file_path", "relative_path", "file_path"):
        value = payload.get(key)
        if value:
            candidates.append(str(value))
    return candidates


# ---------------------------------------------------------------------------
# Explicit entity references (files, symbols, folders) named directly in the
# user's prompt. These must override semantic retrieval, so they are detected
# precisely to avoid false positives on ordinary prose.
# ---------------------------------------------------------------------------

# Multi-hump CamelCase (AgentOrchestrator, FlightAgent) — avoids matching single
# capitalized words like "Explain", "What", "Difference".
CLASS_RE = re.compile(r"\b([A-Z][a-z0-9]+(?:[A-Z][A-Za-z0-9]*)+)\b")
# An identifier immediately followed by "(" (generate_embedding(), build(args)).
FUNCTION_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\([^)]*\)")
# A path that ends with a slash and is not followed by a filename (src/agent/).
FOLDER_RE = re.compile(r"\b([A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*)/(?![A-Za-z0-9_.])")


@dataclass(frozen=True)
class RequestedEntity:
    name: str
    kind: str  # "file" | "symbol" | "folder"


def extract_requested_entities(text: str) -> list[RequestedEntity]:
    """Extract files, symbols (classes/functions), and folders explicitly named in ``text``."""
    if not text:
        return []
    entities: list[RequestedEntity] = []
    seen: set[tuple[str, str]] = set()

    def add(name: str, kind: str) -> None:
        cleaned = name.strip()
        key = (kind, cleaned.lower())
        if cleaned and key not in seen:
            seen.add(key)
            entities.append(RequestedEntity(name=cleaned, kind=kind))

    for reference in extract_file_references(text):
        add(reference, "file")
    for match in FOLDER_RE.finditer(text):
        folder = match.group(1).replace("\\", "/").strip("/")
        if folder:
            add(folder, "folder")
    for match in FUNCTION_RE.finditer(text):
        add(match.group(1), "symbol")
    for match in CLASS_RE.finditer(text):
        add(match.group(1), "symbol")
    return entities


def matches_entity(payload: dict, entity: RequestedEntity) -> bool:
    """Return True when an indexed chunk payload exactly satisfies a requested entity."""
    if entity.kind == "file":
        return matches_file_reference(payload, [entity.name]) is not None
    if entity.kind == "symbol":
        target = entity.name.rstrip("()")
        if not target:
            return False
        symbol = str(payload.get("symbol_name") or "")
        if symbol.lower() == target.lower():
            return True
        content = str(payload.get("content") or "")
        return bool(
            re.search(
                r"\b(?:def|function|fn|class|struct|interface|trait|enum)\s+" + re.escape(target) + r"\b",
                content,
            )
        )
    if entity.kind == "folder":
        target = entity.name.replace("\\", "/").lower().strip("/")
        if not target:
            return False
        folder = str(payload.get("folder") or "").replace("\\", "/").lower().strip("/")
        relative = str(payload.get("relative_file_path") or payload.get("relative_path") or "").replace("\\", "/").lower()
        return folder == target or folder.startswith(target + "/") or relative.startswith(target + "/")
    return False
