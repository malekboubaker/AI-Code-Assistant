from __future__ import annotations

import logging

from backend.agent.context_agent import RequestContext
from backend.agent.rag_controller import RagDecision

logger = logging.getLogger(__name__)

LANGUAGE_LABELS = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "java": "Java",
    "cpp": "C++",
    "csharp": "C#",
    "rust": "Rust",
}

TASK_PREFIXES = {
    "auto_complete": "[TASK: auto_complete]",
    "code_gen": "[TASK: code_gen]",
    "bug_detection": "[TASK: bug_detection]",
    "bug_fix": "[TASK: bug_fix]",
    "perf_opt": "[TASK: perf_opt]",
    "test_gen": "[TASK: test_gen]",
    "refactoring": "[TASK: refactoring]",
    "project_explain": "[TASK: project_explain]",
}


class PromptBuilderAgent:
    def build(self, context: RequestContext, rag: RagDecision) -> str:
        language_label = LANGUAGE_LABELS.get(context.language, context.language)
        parts = [
            TASK_PREFIXES[context.task],
            f"Language: {language_label}",
            f"File path: {context.file_path or 'unknown'}",
            "",
            "You are a local AI code assistant.",
            f"Use idiomatic {language_label} and preserve the surrounding language/runtime conventions.",
        ]
        if context.task == "project_explain":
            parts.extend(_project_explain_instructions(context.explanation_scope))
        elif context.task == "auto_complete":
            parts.extend(
                [
                    "Complete only the missing code at the cursor.",
                    "Return only the code that should be inserted.",
                    "Do not repeat the existing code.",
                    "Do not include Markdown fences.",
                    "Do not include explanation, headings, comments about the answer, or alternatives.",
                    "Keep the completion minimal and syntactically compatible with the current code.",
                ]
            )
        else:
            parts.extend(
                [
                    "Return executable code only.",
                    "Do not use Markdown fences.",
                    "Do not include explanation inside the code block/text.",
                    "Do not include headings such as **Explanation:** in the code output.",
                    "If explanation is needed, put it after a line that says exactly: Explanation:",
                ]
            )

        if context.task == "test_gen":
            parts.extend(
                [
                    "",
                    "Test generation requirements:",
                    f"- Return valid {language_label} test code only.",
                    "- Include imports.",
                    "- Write concrete tests for the visible function and expected behavior.",
                    f"- Prefer specific test names that match common {language_label} conventions.",
                    f"- Use the test framework implied by the existing {language_label} code or project context.",
                    "- If no file/module name is known, write tests that can be adapted and keep them syntactically valid.",
                    "- Do not return an empty response.",
                    "- Do not wrap the answer in Markdown fences.",
                    "- Do not use placeholder tests such as assert true or assert True.",
                ]
            )
        elif context.task == "refactoring":
            parts.extend(["", "Refactoring requirements:", "- Return only the refactored code.", "- Preserve behavior."])
        elif context.task == "perf_opt":
            parts.extend(["", "Performance optimization requirements:", "- Return only the optimized code.", "- Preserve behavior."])
        if context.selected_code_primary and context.task != "project_explain":
            parts.extend(
                [
                    "",
                    "Selected-code requirements:",
                    "- Treat the selected code as the primary source of truth.",
                    "- Use retrieved project context only as supporting context.",
                    "- Do not replace the user's selected-code request with a whole-project answer.",
                ]
            )
        if rag.use_rag:
            parts.extend(
                [
                    "",
                    "Relevant local project context follows.",
                    "Use it only if it is truly relevant; ignore it if it conflicts with the user request.",
                    rag.context,
                ]
            )
        code_label, current_code = _code_section(context)
        parts.extend(
            [
                "",
                "User instruction:",
                context.instruction,
                "",
                code_label,
                current_code,
            ]
        )
        if context.selected_code_primary and context.surrounding_context.strip():
            parts.extend(
                [
                    "",
                    "Nearby file context (supporting only):",
                    f"```{context.language}\n{context.surrounding_context}\n```",
                ]
            )
        parts.extend(
            [
                "",
                (
                    "Return the final answer as prose explanation only. Avoid external APIs or cloud services."
                    if context.task == "project_explain"
                    else (
                        "Return only the missing code to insert. Avoid external APIs or cloud services."
                        if context.task == "auto_complete"
                        else "Return the final answer as code first. Avoid external APIs or cloud services."
                    )
                ),
            ]
        )
        prompt = "\n".join(parts)
        logger.debug("Final prompt sent to model for task=%s:\n%s", context.task, prompt)
        return prompt


def _project_explain_instructions(scope: str) -> list[str]:
    common = [
        f"Explanation scope: {scope}",
        "Do not generate code.",
        "Do not invent classes/functions.",
        "Put the answer in explanation.",
        "Return prose explanation only.",
    ]
    if scope == "project":
        return common + [
            "Use only the retrieved project context.",
            "Do not guess missing architecture.",
            "Do not mention technologies unless they appear in the retrieved files.",
            "If the retrieved context is insufficient, say that clearly.",
            "Write a project-level explanation, not a single-file walkthrough.",
            "Prioritize project map, README/docs, manifests/config files, entry points, important files, and representative modules.",
            "Do not let the active editor file dominate the explanation unless the retrieved sources show it is the main project entry point.",
            "Use this structure exactly:",
            "A. Project goal",
            "B. Main components",
            "C. How it works",
            "D. Main technologies",
            "E. Expected input/output",
            "F. Source-based notes",
            "Grounding rules:",
            "- Use only retrieved project context.",
            "- Do not invent architecture.",
            "- Do not mention a technology unless it appears in retrieved sources.",
            "- If only one file was retrieved, say: \"Based on the available indexed context...\"",
            "- If the project overview is incomplete, say what context is missing.",
            "- Do not mention Qdrant, RAG, vector databases, or the AI Code Assistant infrastructure unless those technologies exist in the target project sources.",
        ]
    if scope == "file":
        return common + [
            "You are explaining the current file, not the whole project.",
            "The current file/context is the primary source of truth.",
            "Use retrieved project context only as supporting context.",
            "Explain the current file using retrieved context only when it clarifies imports, dependencies, or surrounding project behavior.",
            "Focus on the file purpose, key symbols, and how it relates to the project.",
            "Do not turn this into a whole-project explanation unless the user asks for the project.",
            "If no retrieved context is included, base the answer on the current file/context.",
        ]
    return common + [
        "You are explaining selected code from a specific file.",
        "The selected code is the primary source of truth.",
        "Explain exactly what this selected code does and why it exists in this file.",
        "Use retrieved project context only as supporting context.",
        "Do not turn this into a whole-project explanation unless the user asks for the project.",
        "Explain the retrieved code.",
        "Focus on the selected code or requested symbol.",
        "Use retrieved context only when it clarifies dependencies or callers.",
        "Explain what the selected code does.",
        "Explain visible inputs and outputs.",
        "Explain important functions, classes, variables, and control flow.",
        "Explain the selected code's role in the current file.",
        "Mention project-level architecture only if retrieved context clearly supports it.",
        "If no retrieved context is included, say: \"Based on the selected code...\"",
    ]


def _code_section(context: RequestContext) -> tuple[str, str]:
    if context.task == "project_explain" and context.explanation_scope == "project":
        return (
            "Current code:",
            "(not included for project-level explanation; rely on retrieved project sources above)",
        )
    label = "Selected code (primary source of truth):" if context.selected_code_primary else "Current code/context:"
    code = f"```{context.language}\n{context.code}\n```" if context.code else "(none)"
    return label, code
