from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from time import perf_counter

from backend.api.schemas import ProjectReportRequest, ProjectReportResponse, RagSource
from backend.model.generation_config import default_generation_options
from backend.model.model_factory import create_model_provider
from backend.rag.project_identity import normalize_project_path, project_id_for_path, project_name_for_path
from backend.rag.retriever import Retriever
from backend.rag.status import _project_map_payload
from backend.tools.formatter import THINK_RE

logger = logging.getLogger(__name__)

REPORT_TOP_K = 14
REPORT_CONTEXT_BUDGET = 14000
REPORT_QUERY = (
    "Project architecture report: purpose, main components, technologies, frameworks, "
    "data flow, entry points, modules, responsibilities, tests, and code quality."
)

INSUFFICIENT_MESSAGE = (
    "There is not enough indexed context to generate a project report. "
    "Please run indexing first, then try again."
)


class ProjectReportAgent:
    """Generates a structured, source-grounded project report.

    Reuses the existing retriever (project-level explanation retrieval), the project
    map, and the local model provider. It does not change the agent pipeline.
    """

    def __init__(self, retriever: Retriever | None = None, model_provider=None) -> None:
        self.retriever = retriever or Retriever()
        self.model_provider = model_provider or create_model_provider()

    def generate(self, request: ProjectReportRequest) -> ProjectReportResponse:
        started = perf_counter()
        project_path = request.project_path
        project_id = project_id_for_path(project_path)
        normalized_path = normalize_project_path(project_path)
        project_name = project_name_for_path(project_path)
        generated_at = datetime.now(timezone.utc).isoformat()

        point_count = 0
        if hasattr(self.retriever, "count_project_points"):
            point_count = self.retriever.count_project_points(project_id)
        if point_count <= 0:
            return ProjectReportResponse(
                status="not_indexed",
                message=INSUFFICIENT_MESSAGE,
                project_id=project_id,
                project_name=project_name,
                project_path=normalized_path,
                generated_at=generated_at,
                duration_ms=_elapsed_ms(started),
            )

        map_payload = _project_map_payload(self.retriever.store, project_id)
        try:
            sources = self.retriever.search(
                REPORT_QUERY,
                top_k=REPORT_TOP_K,
                project_path=project_path,
                task="project_explain",
                explanation_scope="project",
            )
        except TypeError:
            sources = self.retriever.search(REPORT_QUERY, top_k=REPORT_TOP_K)

        source_files = _source_files(sources)
        project_map_used = _uses_project_map(sources) or bool(map_payload)
        if not sources and not map_payload:
            return ProjectReportResponse(
                status="insufficient_context",
                message=INSUFFICIENT_MESSAGE,
                project_id=project_id,
                project_name=project_name,
                project_path=normalized_path,
                generated_at=generated_at,
                duration_ms=_elapsed_ms(started),
            )

        prompt = _build_report_prompt(sources, map_payload)
        options = default_generation_options(task="project_explain", prompt=prompt)
        raw = self.model_provider.generate(prompt, options)
        body = THINK_RE.sub("", raw).strip()
        if not body:
            return ProjectReportResponse(
                status="insufficient_context",
                message="The local model returned an empty report. Please try again.",
                project_id=project_id,
                project_name=project_name,
                project_path=normalized_path,
                files_analyzed=len(source_files),
                source_files=source_files[:60],
                project_map_used=project_map_used,
                generated_at=generated_at,
                duration_ms=_elapsed_ms(started),
            )

        chunk_count = len(sources)
        source_diversity = round(len(source_files) / chunk_count, 3) if chunk_count else 0.0
        markdown = _wrap_report(
            body=body,
            project_name=project_name,
            project_path=normalized_path,
            generated_at=generated_at,
            files_analyzed=len(source_files),
            source_files=source_files,
            project_map_used=project_map_used,
            chunk_count=chunk_count,
            source_diversity=source_diversity,
        )
        summary = _extract_summary(body)
        logger.info(
            "Project report generated: project_id=%s files_analyzed=%s project_map_used=%s duration_ms=%s",
            project_id,
            len(source_files),
            project_map_used,
            _elapsed_ms(started),
        )
        return ProjectReportResponse(
            status="success",
            markdown=markdown,
            summary=summary,
            project_id=project_id,
            project_name=project_name,
            project_path=normalized_path,
            files_analyzed=len(source_files),
            source_files=source_files[:60],
            project_map_used=project_map_used,
            rag_enabled=True,
            generated_at=generated_at,
            duration_ms=_elapsed_ms(started),
        )


def _elapsed_ms(start: float) -> int:
    return round((perf_counter() - start) * 1000)


def _source_files(sources: list[RagSource]) -> list[str]:
    files: list[str] = []
    for source in sources:
        if source.metadata.get("source") == "project_map" or source.chunk_type == "project_map":
            continue
        relative = (
            source.metadata.get("relative_file_path")
            or source.metadata.get("relative_path")
            or source.file_path
        )
        if relative and relative not in files:
            files.append(str(relative))
    return files


def _uses_project_map(sources: list[RagSource]) -> bool:
    return any(source.metadata.get("source") == "project_map" or source.chunk_type == "project_map" for source in sources)


def _build_report_prompt(sources: list[RagSource], map_payload: dict | None) -> str:
    parts = [
        "[TASK: project_report]",
        "You are a senior software engineer writing an architecture and project-understanding report",
        "after analyzing a repository. The report is used for onboarding, code reviews, and presentations.",
        "Prioritize UNDERSTANDING (what the project does, how it works, dependencies, risks, improvements)",
        "over listing files. Be concise but informative.",
        "",
        "Grounding rules:",
        "- Use ONLY the project facts and retrieved code context below; both come from the real repository.",
        "- Never invent technologies, frameworks, APIs, endpoints, or architecture.",
        "- Do not mention a technology unless it appears in the project facts or retrieved code.",
        "- Do not paste raw file contents, raw chunks, or long file dumps into the report.",
        "- For sections 3, 8, 9, 10 use the provided fact lists; do not add items beyond them.",
        "- If a section has no supporting evidence, write exactly: \"Insufficient evidence to determine this.\"",
        "",
        "Write the report in clean Markdown using EXACTLY these headings, in order:",
        "## 1. Executive Summary",
        "(2-4 sentences: what the project does, its main objective, primary use case, and the main technologies.)",
        "## 2. Project Purpose",
        "(Why it exists, the problem it solves, who the users are, expected outputs. Avoid implementation details.)",
        "## 3. Technologies Used",
        "(Group the detected languages, frameworks, databases, libraries, build/containerization tools under short labels.)",
        "## 4. Architecture Overview",
        "(A readable high-level architecture using only discovered components. A simple top-down arrow flow is encouraged.)",
        "## 5. Main Workflow",
        "(The main execution flow: input -> processing steps / important modules -> output. Explain how it actually works.)",
        "## 6. Folder Structure",
        "(List the important folders and give each a one-line purpose inferred from its files.)",
        "## 7. Important Files",
        "(For each important file: path, role, and why it matters. Keep to the listed important files.)",
        "## 8. Dependency Graph Summary",
        "(A concise, readable summary of key relationships using the provided arrows. Do not dump the full graph.)",
        "## 9. Entry Points",
        "(List the detected entry points and explain each one's role.)",
        "## 10. API Endpoints",
        "(List detected endpoints as method, path, and purpose. If none are detected, say so.)",
        "## 11. Detected Issues",
        "(Only evidence-backed issues, e.g. no tests, missing README/env template, large files, tight coupling, duplicate or unused modules.)",
        "## 12. Suggested Improvements",
        "(Practical, prioritized recommendations grounded in the evidence. Mark each Critical / High / Medium / Low.)",
        "",
        "Do not include a top-level \"# Project Report\" heading and do not write a Sources section;",
        "both are added automatically. Start your answer at \"## 1. Executive Summary\".",
    ]
    facts = _grounded_facts(sources, map_payload)
    parts.extend(["", "PROJECT FACTS (ground truth from the project map and dependency graph):", facts])
    parts.extend(["", "RETRIEVED CODE CONTEXT (evidence only — summarize, never paste verbatim):", _format_sources(sources)])
    parts.extend(["", "Return only the Markdown report. Avoid external APIs or cloud services."])
    return "\n".join(parts)


def _grounded_facts(sources: list[RagSource], map_payload: dict | None) -> str:
    project_map = {}
    if map_payload and isinstance(map_payload.get("project_map"), dict):
        project_map = map_payload["project_map"]
    graph = project_map.get("graph", {}) if isinstance(project_map, dict) else {}
    report = graph.get("report", {}) if isinstance(graph, dict) else {}

    languages = (map_payload or {}).get("detected_languages") or project_map.get("detected_languages") or {}
    frameworks = (map_payload or {}).get("detected_frameworks") or (map_payload or {}).get("frameworks") or []
    project_types = (map_payload or {}).get("project_types") or project_map.get("project_types") or []
    entry_points = (map_payload or {}).get("entry_points") or project_map.get("entry_points") or []
    important_files = (map_payload or {}).get("important_files") or project_map.get("important_files") or []
    source_folders = project_map.get("source_folders") or (map_payload or {}).get("source_folders") or []
    dependency_files = project_map.get("dependency_files") or []
    readme = project_map.get("readme_summary") or ""

    blocks = [
        _fact_block("Languages", [f"{name}={count}" for name, count in languages.items()] if isinstance(languages, dict) else []),
        _fact_block("Frameworks & tools", [str(item) for item in frameworks]),
        _fact_block("Project types", [str(item) for item in project_types]),
        _fact_block("Manifests / build files", [str(item) for item in dependency_files]),
        _fact_block("Source folders", [str(item) for item in source_folders[:14]]),
        _fact_block("Important files", [str(item) for item in important_files[:12]]),
        _fact_block("Entry points", [str(item) for item in entry_points[:12]]),
        _fact_block("Dependency relationships", _dependency_relationships(graph, important_files)),
        _fact_block("API endpoints", _api_endpoints(graph)),
        _fact_block(
            "Graph signals",
            _graph_signal_lines(report),
        ),
    ]
    if readme:
        blocks.append(_fact_block("README/docs excerpt", [readme[:400]]))
    if not map_payload:
        blocks.append("[Project map]\n(none — project map unavailable; rely on retrieved code context)")
    return "\n\n".join(blocks)


def _fact_block(label: str, items: list[str]) -> str:
    if not items:
        return f"[{label}]\n(none detected)"
    return f"[{label}]\n" + "\n".join(f"- {item}" for item in items)


def _graph_signal_lines(report: dict) -> list[str]:
    lines: list[str] = []
    if report.get("most_connected_modules"):
        lines.append("Most connected modules: " + ", ".join(map(str, report["most_connected_modules"][:8])))
    if report.get("critical_files"):
        lines.append("Critical files: " + ", ".join(map(str, report["critical_files"][:8])))
    if report.get("hotspots"):
        lines.append("Architecture hotspots: " + ", ".join(map(str, report["hotspots"][:8])))
    if report.get("edge_counts"):
        lines.append("Relationship counts: " + ", ".join(f"{name}={count}" for name, count in sorted(report["edge_counts"].items())))
    return lines


def _dependency_relationships(graph: dict, important_files: list, limit: int = 14) -> list[str]:
    relations: list[str] = []
    seen: set[tuple[str, str]] = set()
    file_relations = graph.get("file_relations", {}) if isinstance(graph, dict) else {}
    ordered_files = [str(item) for item in important_files] + [
        key for key in file_relations if str(key) not in {str(item) for item in important_files}
    ]
    for relative in ordered_files:
        info = file_relations.get(relative)
        if not isinstance(info, dict):
            continue
        for target in (info.get("imports") or [])[:3]:
            pair = (str(relative), str(target))
            if pair in seen:
                continue
            seen.add(pair)
            relations.append(f"{relative} → {target}")
            if len(relations) >= limit:
                return relations
    if relations:
        return relations
    for edge in graph.get("edges", []) if isinstance(graph, dict) else []:
        if len(edge) >= 3 and edge[2] == "FILE_IMPORTS_FILE":
            pair = (str(edge[0]), str(edge[1]))
            if pair in seen:
                continue
            seen.add(pair)
            relations.append(f"{edge[0]} → {edge[1]}")
            if len(relations) >= limit:
                break
    return relations


def _api_endpoints(graph: dict) -> list[str]:
    routes = graph.get("api_routes", []) if isinstance(graph, dict) else []
    endpoints: list[str] = []
    for route in routes[:30]:
        if not isinstance(route, dict):
            continue
        method = route.get("method", "")
        path = route.get("path", "")
        handler = route.get("handler", "")
        file_ref = route.get("file", "")
        suffix = f" → {handler}" if handler else ""
        location = f" ({file_ref})" if file_ref else ""
        endpoints.append(f"{method} {path}{suffix}{location}".strip())
    return endpoints


def _format_sources(sources: list[RagSource]) -> str:
    blocks: list[str] = []
    budget = REPORT_CONTEXT_BUDGET
    for source in sources:
        if budget <= 0:
            break
        if source.metadata.get("source") == "project_map" or source.chunk_type == "project_map":
            content = _trim(source.content, min(2400, budget))
            blocks.append(f"### Project map summary\n{content}")
            budget -= len(content)
            continue
        relative = (
            source.metadata.get("relative_file_path")
            or source.metadata.get("relative_path")
            or source.file_path
            or "unknown"
        )
        content = _trim(source.content, min(1500, budget))
        blocks.append(
            f"### {relative} (type={source.chunk_type or 'unknown'}, symbol={source.symbol_name or 'n/a'})\n"
            f"```{source.language or ''}\n{content}\n```"
        )
        budget -= len(content)
    return "\n\n".join(blocks) if blocks else "(no project sources retrieved)"


def _wrap_report(
    *,
    body: str,
    project_name: str,
    project_path: str,
    generated_at: str,
    files_analyzed: int,
    source_files: list[str],
    project_map_used: bool,
    chunk_count: int,
    source_diversity: float,
) -> str:
    display_date = generated_at.replace("T", " ")[:16]
    header = [
        "# Project Report",
        "",
        f"**Project:** {project_name}",
        f"**Path:** {project_path}",
        f"**Generated:** {display_date} UTC",
        f"**Files analyzed:** {files_analyzed} · **RAG:** enabled · "
        f"**Project map:** {'used' if project_map_used else 'not used'}",
        "",
        body.strip(),
        "",
        "---",
        "",
        "## 13. Sources & Evidence",
        f"- Files analyzed: {files_analyzed}",
        f"- Chunks analyzed: {chunk_count}",
        f"- Project map used: {'yes' if project_map_used else 'no'}",
        f"- Source diversity score: {source_diversity}",
        "- Source files used:",
    ]
    if source_files:
        header.extend(f"  - {relative}" for relative in source_files[:60])
    else:
        header.append("  - (no source files retrieved)")
    return "\n".join(header) + "\n"


def _extract_summary(body: str) -> str:
    match = re.search(
        r"##\s*1\.?\s*Executive Summary\s*(.+?)(?:\n##\s|\Z)", body, re.IGNORECASE | re.DOTALL
    )
    text = match.group(1) if match else body
    lines = [re.sub(r"^[\s\-*#>]+", "", line).strip() for line in text.splitlines()]
    joined = " ".join(line for line in lines if line)
    if len(joined) > 500:
        return joined[:500].rstrip() + "…"
    return joined


def _trim(content: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(content) <= max_chars:
        return content
    return content[:max_chars].rstrip() + "\n...[trimmed]"
