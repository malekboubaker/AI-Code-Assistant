from backend.agent.context_agent import RequestContext
from backend.agent.prompt_builder import PromptBuilderAgent
from backend.agent.rag_controller import RagDecision


def test_prompt_builder_includes_task_prefix():
    context = RequestContext(
        task="code_gen",
        language="python",
        instruction="write add",
        code="",
        file_path=None,
        project_path=None,
        imports=[],
    )
    prompt = PromptBuilderAgent().build(context, RagDecision(use_rag=False))
    assert "[TASK: code_gen]" in prompt
    assert "write add" in prompt


def test_prompt_builder_explain_task_forbids_code_generation():
    context = RequestContext(
        task="project_explain",
        language="python",
        instruction="Explain how AgentOrchestrator works",
        code="",
        file_path=None,
        project_path=None,
        imports=[],
        explanation_scope="project",
    )
    prompt = PromptBuilderAgent().build(context, RagDecision(use_rag=False))
    assert "[TASK: project_explain]" in prompt
    assert "Write a project-level explanation, not a single-file walkthrough." in prompt
    assert "Do not generate code." in prompt
    assert "Do not invent classes/functions." in prompt
    assert "Use only the retrieved project context." in prompt
    assert "Do not guess missing architecture." in prompt
    assert "Do not mention technologies unless they appear in the retrieved files." in prompt
    assert "If the retrieved context is insufficient, say that clearly." in prompt
    assert "Put the answer in explanation." in prompt
    assert "Return prose explanation only" in prompt
    assert "Return executable code only" not in prompt
    assert "Return the final answer as code first" not in prompt


def test_prompt_builder_selection_explanation_makes_selected_code_primary():
    context = RequestContext(
        task="project_explain",
        language="python",
        instruction="Explain this code",
        code="def calculate_total(items):\n    return sum(items)\n",
        file_path="billing.py",
        project_path=".",
        imports=[],
        explanation_scope="selection",
        has_selection=True,
        selected_code_primary=True,
        active_file_path="billing.py",
        surrounding_context="class Invoice:\n    def calculate_total(self, items):\n        return sum(items)\n",
    )

    prompt = PromptBuilderAgent().build(context, RagDecision(use_rag=False))

    assert "You are explaining selected code from a specific file." in prompt
    assert "The selected code is the primary source of truth." in prompt
    assert "Explain exactly what this selected code does and why it exists in this file." in prompt
    assert "Use retrieved project context only as supporting context." in prompt
    assert "Do not turn this into a whole-project explanation unless the user asks for the project." in prompt
    assert "Selected code (primary source of truth):" in prompt
    assert "Nearby file context (supporting only):" in prompt
    assert "A. Project goal" not in prompt


def test_prompt_builder_project_scope_uses_project_explanation_structure():
    context = RequestContext(
        task="project_explain",
        language="python",
        instruction="Explain this project",
        code="def active_file_only():\n    return 'do not focus here'\n",
        file_path="agents/flight_agent/agent.py",
        project_path=".",
        imports=[],
        explanation_scope="project",
    )

    prompt = PromptBuilderAgent().build(context, RagDecision(use_rag=False))

    assert "Explanation scope: project" in prompt
    assert "A. Project goal" in prompt
    assert "B. Main components" in prompt
    assert "C. How it works" in prompt
    assert "D. Main technologies" in prompt
    assert "E. Expected input/output" in prompt
    assert "F. Source-based notes" in prompt
    assert "Do not mention Qdrant, RAG, vector databases, or the AI Code Assistant infrastructure" in prompt
    assert "def active_file_only" not in prompt
    assert "(not included for project-level explanation" in prompt


def test_prompt_builder_auto_complete_returns_only_missing_code():
    context = RequestContext(
        task="auto_complete",
        language="python",
        instruction="complete",
        code="def add(a, b):\n    return",
        file_path=None,
        project_path=None,
        imports=[],
    )
    prompt = PromptBuilderAgent().build(context, RagDecision(use_rag=False))
    assert "Complete only the missing code at the cursor." in prompt
    assert "Return only the code that should be inserted." in prompt
    assert "Do not repeat the existing code." in prompt
    assert "Return only the missing code to insert." in prompt
    assert "Return executable code only." not in prompt
    assert "Return the final answer as code first" not in prompt


def test_prompt_builder_test_gen_is_language_aware_for_typescript():
    context = RequestContext(
        task="test_gen",
        language="typescript",
        instruction="generate tests",
        code="export function add(a: number, b: number): number { return a + b; }",
        file_path="src/add.ts",
        project_path=None,
        imports=[],
    )
    prompt = PromptBuilderAgent().build(context, RagDecision(use_rag=False))

    assert "Language: TypeScript" in prompt
    assert "Return valid TypeScript test code only." in prompt
    assert "common TypeScript conventions" in prompt
    assert "pytest" not in prompt.lower()
