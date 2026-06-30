from __future__ import annotations

import re

from backend.api.schemas import TaskName
from backend.rag.file_matching import extract_file_references


EXPLAIN_PREFIXES = (
    "explain",
    "describe",
    "how does",
    "how do",
    "what is the role of",
    "what does",
)


KEYWORDS: list[tuple[str, TaskName]] = [
    ("compare", "compare"),
    ("difference", "compare"),
    ("autocomplete", "auto_complete"),
    ("complete", "auto_complete"),
    ("fix", "bug_fix"),
    ("bug", "bug_detection"),
    ("error", "bug_detection"),
    ("test", "test_gen"),
    ("refactor", "refactoring"),
    ("optimize", "perf_opt"),
    ("performance", "perf_opt"),
    ("generate", "code_gen"),
    ("write", "code_gen"),
    ("explain", "project_explain"),
]

SOURCE_LISTING_PATTERNS = (
    r"\blist\s+(?:the\s+)?(?:source\s+files|sources)\b",
    r"\bshow\s+(?:the\s+)?(?:source\s+files|sources)\b",
    r"\bwhich\s+files\s+did\s+you\s+use\b",
    r"\bwhat\s+(?:source\s+files|sources)\s+did\s+you\s+use\b",
    r"\bsources\s+used\b",
)


class TaskRouterAgent:
    def detect(self, instruction: str, explicit_task: TaskName | None = None) -> TaskName:
        if explicit_task:
            if explicit_task == "explain":
                return "project_explain"
            return explicit_task
        lower = instruction.lower()
        detected_task = "code_gen"
        if lower.strip().startswith(EXPLAIN_PREFIXES) or any(re.search(pattern, lower) for pattern in SOURCE_LISTING_PATTERNS):
            detected_task = "project_explain"
        else:
            for keyword, task in KEYWORDS:
                if keyword in lower:
                    detected_task = task
                    break
        
        if detected_task == "project_explain":
            if "this file" in lower or "the file" in lower or "this class" in lower or "this function" in lower:
                return "file_explain"
            if len(extract_file_references(instruction)) > 0:
                return "file_explain"
                
        return detected_task
