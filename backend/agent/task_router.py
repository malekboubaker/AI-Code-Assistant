from __future__ import annotations

import re

from backend.api.schemas import TaskName


EXPLAIN_PREFIXES = (
    "explain",
    "describe",
    "how does",
    "how do",
    "what is the role of",
    "what does",
)


KEYWORDS: list[tuple[str, TaskName]] = [
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
        if any(re.search(pattern, lower) for pattern in SOURCE_LISTING_PATTERNS):
            return "project_explain"
        if lower.strip().startswith(EXPLAIN_PREFIXES):
            return "project_explain"
        for keyword, task in KEYWORDS:
            if keyword in lower:
                return task
        return "code_gen"
