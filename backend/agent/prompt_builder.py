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
            parts.extend(
                [
                    "Explain the retrieved code.",
                    "Do not generate code.",
                    "Do not invent classes/functions.",
                    "Use only the retrieved project context when it is relevant.",
                    "Put the answer in explanation.",
                    "If the context is insufficient, say what is missing.",
                    "Return prose explanation only.",
                ]
            )
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
        if rag.use_rag:
            parts.extend(
                [
                    "",
                    "Relevant local project context from Qdrant follows.",
                    "Use it only if it is truly relevant; ignore it if it conflicts with the user request.",
                    rag.context,
                ]
            )
        parts.extend(
            [
                "",
                "User instruction:",
                context.instruction,
                "",
                "Current code:",
                f"```{context.language}\n{context.code}\n```" if context.code else "(none)",
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
