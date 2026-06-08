from backend.agent.context_agent import ContextAgent
from backend.api.schemas import GenerateRequest


def test_context_agent_detects_project_explanation_scope():
    context = ContextAgent().build(
        GenerateRequest(
            instruction="What does this project do?",
            code="def flight_agent():\n    pass\n",
            language="python",
            file_path="agents/flight_agent/agent.py",
            project_path=".",
        ),
        "project_explain",
    )

    assert context.explanation_scope == "project"


def test_context_agent_detects_file_explanation_scope():
    context = ContextAgent().build(
        GenerateRequest(
            instruction="Explain this file",
            code="export function run() {}",
            language="typescript",
            file_path="src/run.ts",
            project_path=".",
        ),
        "project_explain",
    )

    assert context.explanation_scope == "file"


def test_context_agent_detects_selection_explanation_scope():
    context = ContextAgent().build(
        GenerateRequest(
            instruction="Explain this function",
            code="def calculate_total(items):\n    return sum(items)\n",
            language="python",
            file_path="billing.py",
            project_path=".",
            has_selection=True,
            surrounding_context="class Invoice:\n    def calculate_total(self, items):\n        return sum(items)\n",
        ),
        "project_explain",
    )

    assert context.explanation_scope == "selection"
    assert context.has_selection is True
    assert context.selected_code_primary is True
    assert context.active_file_path == "billing.py"
    assert "Invoice" in context.surrounding_context


def test_context_agent_treats_explicit_file_explanation_as_file_even_with_code():
    context = ContextAgent().build(
        GenerateRequest(
            instruction="Explain this file",
            code="def helper():\n    return 1\n",
            language="python",
            file_path="helpers.py",
            project_path=".",
            has_selection=False,
        ),
        "project_explain",
    )

    assert context.explanation_scope == "file"
    assert context.has_selection is False
    assert context.selected_code_primary is False


def test_context_agent_preserves_chat_history():
    context = ContextAgent().build(
        GenerateRequest(
            instruction="Now refactor the second function",
            code="def first(): pass\n\ndef second(): pass\n",
            language="python",
            file_path="helpers.py",
            project_path=".",
            chat_history=[
                {"role": "user", "content": "Explain this file."},
                {"role": "assistant", "content": "The second function is second()."},
            ],
        ),
        "refactoring",
    )

    assert context.chat_history
    assert context.chat_history[-1].role == "assistant"
    assert "second()" in context.chat_history[-1].content
