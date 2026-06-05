from backend.agent.orchestrator import AgentOrchestrator
from backend.agent.rag_controller import RagControllerAgent
from backend.api.schemas import GenerateRequest, RagSource
from backend.rag.project_identity import project_id_for_path


class FakeModel:
    name = "fake"

    def generate(self, prompt, options=None):
        return "def add(a, b):\n    return a + b\n"


class TaskAwareCodeModel:
    name = "task-aware-code"

    def generate(self, prompt, options=None):
        if "[TASK: bug_fix]" in prompt:
            return "def divide(a, b):\n    if b == 0:\n        return None\n    return a / b\n"
        if "[TASK: refactoring]" in prompt:
            return "def calc(x):\n    return [i * i for i in x]\n"
        if "[TASK: perf_opt]" in prompt:
            return "result = [x * 2 for x in items]\n"
        if "[TASK: test_gen]" in prompt:
            return "def test_add_positive_numbers():\n    assert add(2, 3) == 5\n"
        return "def generated():\n    return True\n"


class MultiLanguageModel:
    name = "multi-language"

    def generate(self, prompt, options=None):
        if "[TASK: refactoring]" in prompt and "Language: JavaScript" in prompt:
            return "function sum(items) {\n  return items.reduce((total, item) => total + item, 0);\n}\n"
        if "[TASK: test_gen]" in prompt and "Language: TypeScript" in prompt:
            return (
                "import { add } from './add';\n\n"
                "describe('add', () => {\n"
                "  it('adds numbers', () => {\n"
                "    expect(add(2, 3)).toBe(5);\n"
                "  });\n"
                "});\n"
            )
        if "[TASK: bug_fix]" in prompt and "Language: Java" in prompt:
            return "public int divide(int a, int b) {\n    if (b == 0) { return 0; }\n    return a / b;\n}\n"
        if "[TASK: perf_opt]" in prompt and "Language: C++" in prompt:
            return "std::vector<int> doubled;\ndoubled.reserve(items.size());\nfor (int item : items) {\n    doubled.push_back(item * 2);\n}\n"
        if "[TASK: refactoring]" in prompt and "Language: C#" in prompt:
            return "public int Calculate(int value)\n{\n    return value * value;\n}\n"
        return ""


class FakeMemory:
    def maybe_store(self, *args, **kwargs):
        return False


class OptionCapturingModel:
    name = "option-capture"

    def __init__(self):
        self.options = None

    def generate(self, prompt, options=None):
        self.options = options
        return "a + b\n"


class FailingRetriever:
    def search(self, query, top_k=None):
        raise AssertionError("RAG should be disabled for default auto_complete requests")


class NoResultsRetriever:
    def count_project_points(self, project_id):
        return 3

    def search(self, query, top_k=None, project_path=None, **kwargs):
        return []


class SupportingSelectionRetriever:
    def count_project_points(self, project_id):
        return 2

    def search(self, query, top_k=None, project_path=None, **kwargs):
        project_id = project_id_for_path(project_path)
        return [
            RagSource(
                content="Invoice.calculate_total is used by the billing file to total visible invoice items.",
                score=0.72,
                language="python",
                file_path="billing.py",
                start_line=1,
                end_line=4,
                chunk_type="function",
                symbol_name="calculate_total",
                metadata={"project_id": project_id, "relative_file_path": "billing.py"},
            )
        ]


class AutoCompleteRelevantRetriever:
    def count_project_points(self, project_id):
        return 1

    def search(self, query, top_k=None, project_path=None, **kwargs):
        project_id = project_id_for_path(project_path)
        return [
            RagSource(
                content="def add(a, b):\n    return a + b",
                score=0.71,
                language="python",
                file_path="math_utils.py",
                start_line=1,
                end_line=2,
                chunk_type="function",
                symbol_name="add",
                metadata={"project_id": project_id, "relative_file_path": "math_utils.py"},
            )
        ]


def test_orchestrator_runs_without_rag():
    orchestrator = AgentOrchestrator()
    orchestrator.model_provider = FakeModel()
    orchestrator.memory_writer = FakeMemory()
    response = orchestrator.run(
        GenerateRequest(instruction="write add function", language="python", use_rag=False)
    )
    assert response.task == "code_gen"
    assert response.validation.syntax_valid is True
    assert "def add" in response.generated_code
    assert response.metadata["model_name"] == "fake"
    assert response.metadata["prompt_length_chars"] > 0
    assert response.metadata["generated_length_chars"] > 0
    assert response.metadata["timing_total_ms"] >= 0
    assert response.metadata["timing_rag_ms"] >= 0
    assert response.metadata["timing_model_ms"] >= 0
    assert response.metadata["timing_validation_ms"] >= 0
    assert response.metadata["validator_used"] == "python"
    assert response.metadata["validation_duration_ms"] >= 0
    assert response.metadata["validation_errors"] == []
    assert isinstance(response.metadata["validation_warnings"], list)


def test_auto_complete_uses_short_generation_and_skips_default_rag():
    orchestrator = AgentOrchestrator()
    model = OptionCapturingModel()
    orchestrator.model_provider = model
    orchestrator.memory_writer = FakeMemory()
    orchestrator.rag_controller = RagControllerAgent(retriever=FailingRetriever())

    response = orchestrator.run(
        GenerateRequest(
            task="auto_complete",
            instruction="complete this line",
            language="python",
            code="def add(a, b):\n    return",
        )
    )

    assert response.used_rag is False
    assert model.options.max_tokens == 64
    assert model.options.temperature == 0.1


def test_auto_complete_uses_rag_when_explicitly_enabled():
    orchestrator = AgentOrchestrator()
    model = OptionCapturingModel()
    orchestrator.model_provider = model
    orchestrator.memory_writer = FakeMemory()
    orchestrator.rag_controller = RagControllerAgent(retriever=AutoCompleteRelevantRetriever())

    response = orchestrator.run(
        GenerateRequest(
            task="auto_complete",
            instruction="complete this line",
            language="python",
            code="def add(a, b):\n    return",
            file_path="math_utils.py",
            project_path=".",
            use_rag=True,
        )
    )

    assert response.used_rag is True
    assert response.metadata["rag_context_available"] is True
    assert response.metadata["rag_fallback_used"] is False
    assert model.options.max_tokens == 64


def test_code_task_continues_when_rag_has_no_results():
    orchestrator = AgentOrchestrator()
    orchestrator.model_provider = TaskAwareCodeModel()
    orchestrator.memory_writer = FakeMemory()
    orchestrator.rag_controller = RagControllerAgent(retriever=NoResultsRetriever())

    response = orchestrator.run(
        GenerateRequest(
            task="refactoring",
            instruction="Refactor this function",
            language="python",
            code="def calc(x):\n    y=[]\n    [y.append(i*i) for i in x]\n    return y\n",
            file_path="calc.py",
            project_path=".",
            has_selection=True,
            use_rag=True,
        )
    )

    assert response.generated_code.strip()
    assert response.used_rag is False
    assert response.metadata["rag_skip_reason"] == "no_results"
    assert response.metadata["rag_fallback_used"] is True
    assert response.metadata["rag_context_available"] is False
    assert response.metadata["selected_code_primary"] is True


def test_code_generation_tasks_still_return_valid_code():
    cases = [
        ("code_gen", "write a function"),
        ("bug_fix", "fix this function"),
        ("refactoring", "refactor this function"),
        ("perf_opt", "optimize this loop"),
        ("test_gen", "write pytest tests"),
    ]
    for task, instruction in cases:
        orchestrator = AgentOrchestrator()
        orchestrator.model_provider = TaskAwareCodeModel()
        orchestrator.memory_writer = FakeMemory()
        response = orchestrator.run(
            GenerateRequest(
                task=task,
                instruction=instruction,
                language="python",
                code="def add(a, b):\n    return a + b\n",
                use_rag=False,
            )
        )
        assert response.generated_code.strip()
        assert response.validation.valid is True
        assert response.validation.syntax_valid is True


def test_multilanguage_code_tasks_return_language_specific_outputs():
    cases = [
        (
            "refactoring",
            "javascript",
            "function sum(items) { let total = 0; for (const item of items) total += item; return total; }",
            "reduce",
        ),
        (
            "test_gen",
            "typescript",
            "export function add(a: number, b: number): number { return a + b; }",
            "describe('add'",
        ),
        ("bug_fix", "java", "public int divide(int a, int b) { return a / b; }", "if (b == 0)"),
        ("perf_opt", "cpp", "std::vector<int> doubled; for (int item : items) doubled.push_back(item * 2);", "reserve"),
        ("refactoring", "csharp", "public int Calculate(int value) { var x = value * value; return x; }", "Calculate"),
    ]

    for task, language, code, expected in cases:
        orchestrator = AgentOrchestrator()
        orchestrator.model_provider = MultiLanguageModel()
        orchestrator.memory_writer = FakeMemory()
        response = orchestrator.run(
            GenerateRequest(
                task=task,
                instruction="improve this code",
                language=language,
                code=code,
                use_rag=False,
            )
        )

        assert response.language == language
        assert expected in response.generated_code
        assert response.validation.valid is True


class GoodTestModel:
    name = "good-test"

    def generate(self, prompt, options=None):
        return (
            "def test_add_positive_numbers():\n"
            "    assert add(2, 3) == 5\n\n"
            "def test_add_negative_numbers():\n"
            "    assert add(-2, -3) == -5\n"
        )


def test_orchestrator_test_gen_returns_pytest_code():
    orchestrator = AgentOrchestrator()
    orchestrator.model_provider = GoodTestModel()
    orchestrator.memory_writer = FakeMemory()
    response = orchestrator.run(
        GenerateRequest(
            task="test_gen",
            instruction="generate pytest tests",
            language="python",
            code="def add(a, b):\n    return a + b\n",
            use_rag=False,
        )
    )
    assert response.task == "test_gen"
    assert response.generated_code.startswith("def test_add")
    assert response.validation.syntax_valid is True


class MarkdownModel:
    name = "markdown"

    def generate(self, prompt, options=None):
        return "```python\ndef calc(x):\n    return [i * i for i in x]\n```\n\n**Explanation:** cleaner"


def test_orchestrator_extracts_code_from_markdown():
    orchestrator = AgentOrchestrator()
    orchestrator.model_provider = MarkdownModel()
    orchestrator.memory_writer = FakeMemory()
    response = orchestrator.run(
        GenerateRequest(
            task="refactoring",
            instruction="refactor",
            language="python",
            code="def calc(x):\n    y=[]\n    [y.append(i*i) for i in x]\n    return y\n",
            use_rag=False,
        )
    )
    assert response.generated_code == "def calc(x):\n    return [i * i for i in x]"
    assert "Explanation" not in response.generated_code
    assert response.explanation == "cleaner"
    assert response.validation.syntax_valid is True


class EmptyModel:
    name = "empty"

    def generate(self, prompt, options=None):
        return ""


def test_orchestrator_empty_output_is_invalid_and_not_fallback():
    orchestrator = AgentOrchestrator()
    orchestrator.model_provider = EmptyModel()
    orchestrator.memory_writer = FakeMemory()
    response = orchestrator.run(
        GenerateRequest(task="test_gen", instruction="generate tests", language="python", use_rag=False)
    )
    assert response.generated_code == ""
    assert response.validation.valid is False
    assert response.metadata["fallback_output"] is False
    assert response.metadata["stored_in_memory"] is False


class PromptCapturingModel:
    name = "capture"

    def __init__(self):
        self.prompt = ""

    def generate(self, prompt, options=None):
        self.prompt = prompt
        return "def routed():\n    return True\n"


class RelevantRetriever:
    def search(self, query, top_k=None, project_path=None, **kwargs):
        project_id = project_id_for_path(project_path)
        return [
            RagSource(
                content="class AgentOrchestrator:\n    def run(self, request): pass",
                score=0.653,
                language="python",
                file_path="backend/agent/orchestrator.py",
                start_line=1,
                end_line=2,
                symbol_name="AgentOrchestrator",
                metadata={"project_id": project_id},
            ),
            RagSource(
                content="Project map summary:\n- Project type: Python\n- Entry points: backend/agent/orchestrator.py",
                score=0.62,
                language="text",
                file_path="project_map.json",
                start_line=1,
                end_line=2,
                chunk_type="project_map",
                symbol_name="project_map",
                metadata={"source": "project_map", "project_id": project_id},
            ),
        ]


class TypeScriptRetriever:
    def search(self, query, top_k=None, language=None, project_path=None, **kwargs):
        project_id = project_id_for_path(project_path)
        return [
            RagSource(
                content="export function requestHandler(req: Request) {\n  return handle(req);\n}",
                score=0.72,
                language="typescript",
                file_path="src/requestHandler.ts",
                start_line=1,
                end_line=3,
                symbol_name="requestHandler",
                metadata={"project_id": project_id},
            ),
            RagSource(
                content="Project map summary:\n- Project type: Node.js\n- Entry points: src/requestHandler.ts",
                score=0.61,
                language="text",
                file_path="project_map.json",
                start_line=1,
                end_line=2,
                chunk_type="project_map",
                symbol_name="project_map",
                metadata={"source": "project_map", "project_id": project_id},
            ),
        ]


def test_orchestrator_injects_rag_context_above_threshold():
    orchestrator = AgentOrchestrator()
    model = PromptCapturingModel()
    orchestrator.model_provider = model
    orchestrator.memory_writer = FakeMemory()
    orchestrator.rag_controller = RagControllerAgent(retriever=RelevantRetriever())

    response = orchestrator.run(
        GenerateRequest(
            instruction="orchestrator task router prompt builder",
            language="python",
            project_path=".",
            use_rag=True,
        )
    )

    assert response.used_rag is True
    assert response.rag_sources
    assert response.metadata["rag_best_score"] == 0.653
    assert "Retrieved chunk" in model.prompt
    assert "AgentOrchestrator" in model.prompt


class ExplanationModel:
    name = "explain"

    def __init__(self):
        self.prompt = ""

    def generate(self, prompt, options=None):
        self.prompt = prompt
        return (
            "AgentOrchestrator coordinates the workflow by first asking TaskRouterAgent "
            "to classify the request, then using ContextAgent and RAG, and finally passing "
            "the assembled context to PromptBuilderAgent to create the prompt."
        )


class PlainExplanationModel:
    name = "plain-explain"

    def __init__(self):
        self.prompt = ""

    def generate(self, prompt, options=None):
        self.prompt = prompt
        return "The requested code is explained using the provided local context."


class CapturingProjectRetriever:
    def __init__(self):
        self.query = ""
        self.kwargs = {}

    def count_project_points(self, project_id):
        return 4

    def search(self, query, top_k=None, project_path=None, **kwargs):
        self.query = query
        self.kwargs = kwargs
        project_id = project_id_for_path(project_path)
        return [
            RagSource(
                content=(
                    "Project map summary:\n"
                    "- Project type: Python\n"
                    "- Entry points: agents/host_agent/__main__.py\n"
                    "- Important files: agents/flight_agent/agent.py, agents/stay_agent/agent.py"
                ),
                score=0.78,
                language="text",
                file_path="project_map.json",
                start_line=1,
                end_line=4,
                chunk_type="project_map",
                symbol_name="project_map",
                metadata={
                    "source": "project_map",
                    "project_id": project_id,
                    "relative_file_path": "project_map.json",
                },
            ),
            RagSource(
                content="# Travel Planner\nPlans travel options from user requests.",
                score=0.72,
                language="markdown",
                file_path="README.md",
                start_line=1,
                end_line=2,
                chunk_type="doc",
                metadata={
                    "project_id": project_id,
                    "relative_file_path": "README.md",
                    "is_doc_file": True,
                },
            ),
            RagSource(
                content="def find_flights(request):\n    return []",
                score=0.68,
                language="python",
                file_path="agents/flight_agent/agent.py",
                start_line=1,
                end_line=2,
                chunk_type="function",
                symbol_name="find_flights",
                metadata={"project_id": project_id, "relative_file_path": "agents/flight_agent/agent.py"},
            ),
            RagSource(
                content="def find_stays(request):\n    return []",
                score=0.67,
                language="python",
                file_path="agents/stay_agent/agent.py",
                start_line=1,
                end_line=2,
                chunk_type="function",
                symbol_name="find_stays",
                metadata={"project_id": project_id, "relative_file_path": "agents/stay_agent/agent.py"},
            ),
        ]


def test_orchestrator_project_explain_routes_to_project_scope_without_active_code_dominance(tmp_path):
    orchestrator = AgentOrchestrator()
    model = PlainExplanationModel()
    retriever = CapturingProjectRetriever()
    orchestrator.model_provider = model
    orchestrator.memory_writer = FakeMemory()
    orchestrator.rag_controller = RagControllerAgent(retriever=retriever)

    response = orchestrator.run(
        GenerateRequest(
            instruction="Explain this project",
            language="python",
            code="def active_flight_only():\n    return 'narrow file context'\n",
            file_path=str(tmp_path / "agents" / "flight_agent" / "agent.py"),
            project_path=str(tmp_path),
            use_rag=True,
        )
    )

    assert response.task == "project_explain"
    assert response.metadata["explanation_scope"] == "project"
    assert retriever.kwargs["explanation_scope"] == "project"
    assert "active_flight_only" not in retriever.query
    assert "active_flight_only" not in model.prompt
    assert "A. Project goal" in model.prompt
    assert response.metadata["project_map_used"] is True
    assert response.metadata["source_file_count"] >= 3
    assert "README.md" in response.metadata["source_files_used"]
    assert response.metadata["selected_code_primary"] is False
    assert response.metadata["rag_context_available"] is True
    assert response.metadata["rag_fallback_used"] is False


def test_orchestrator_file_level_explanation_still_works_without_rag():
    orchestrator = AgentOrchestrator()
    model = PlainExplanationModel()
    orchestrator.model_provider = model
    orchestrator.memory_writer = FakeMemory()

    response = orchestrator.run(
        GenerateRequest(
            instruction="Explain this file",
            language="typescript",
            code="export function requestHandler(req: Request) {\n  return handle(req);\n}",
            file_path="src/requestHandler.ts",
            use_rag=False,
        )
    )

    assert response.task == "project_explain"
    assert response.metadata["explanation_scope"] == "file"
    assert response.metadata["selected_code_primary"] is False
    assert response.explanation.startswith("The requested code")
    assert "requestHandler" in model.prompt
    assert "You are explaining the current file, not the whole project." in model.prompt
    assert "A. Project goal" not in model.prompt


def test_orchestrator_selected_code_explanation_still_works_without_rag():
    orchestrator = AgentOrchestrator()
    model = PlainExplanationModel()
    orchestrator.model_provider = model
    orchestrator.memory_writer = FakeMemory()
    orchestrator.rag_controller = RagControllerAgent(retriever=NoResultsRetriever())

    response = orchestrator.run(
        GenerateRequest(
            instruction="Explain this code",
            language="python",
            code="def calculate_total(items):\n    return sum(items)\n",
            file_path="billing.py",
            project_path=".",
            has_selection=True,
            use_rag=True,
        )
    )

    assert response.task == "project_explain"
    assert response.metadata["explanation_scope"] == "selection"
    assert response.metadata["selected_code_primary"] is True
    assert response.metadata["rag_fallback_used"] is True
    assert response.metadata["rag_context_available"] is False
    assert response.explanation.startswith("The requested code")
    assert "calculate_total" in model.prompt
    assert "Selected code (primary source of truth):" in model.prompt
    assert "A. Project goal" not in model.prompt


def test_orchestrator_selected_code_explanation_uses_rag_as_supporting_context():
    orchestrator = AgentOrchestrator()
    model = PlainExplanationModel()
    orchestrator.model_provider = model
    orchestrator.memory_writer = FakeMemory()
    orchestrator.rag_controller = RagControllerAgent(retriever=SupportingSelectionRetriever())

    response = orchestrator.run(
        GenerateRequest(
            instruction="Explain this code",
            language="python",
            code="def calculate_total(items):\n    return sum(items)\n",
            file_path="billing.py",
            project_path=".",
            has_selection=True,
            surrounding_context="class Invoice:\n    def calculate_total(self, items):\n        return sum(items)\n",
            use_rag=True,
        )
    )

    assert response.used_rag is True
    assert response.metadata["explanation_scope"] == "selection"
    assert response.metadata["selected_code_primary"] is True
    assert response.metadata["rag_fallback_used"] is False
    assert response.metadata["rag_context_available"] is True
    assert "Selected code (primary source of truth):" in model.prompt
    assert "Nearby file context (supporting only):" in model.prompt
    assert "Retrieved chunk" in model.prompt
    assert "A. Project goal" not in model.prompt


def test_orchestrator_project_explain_uses_rag_without_generated_code():
    orchestrator = AgentOrchestrator()
    model = ExplanationModel()
    orchestrator.model_provider = model
    orchestrator.memory_writer = FakeMemory()
    orchestrator.rag_controller = RagControllerAgent(retriever=RelevantRetriever())

    response = orchestrator.run(
        GenerateRequest(
            instruction=(
                "Explain how AgentOrchestrator coordinates TaskRouterAgent and "
                "PromptBuilderAgent in this project. Do not generate code."
                ),
                language="python",
                project_path=".",
                use_rag=True,
            )
    )

    assert response.task == "project_explain"
    assert response.used_rag is True
    assert response.rag_sources
    assert response.explanation
    assert response.generated_code == ""
    assert "class AgentOrchestrator" not in response.generated_code
    assert any(
        source.file_path and any(name in source.file_path for name in ("orchestrator.py", "task_router.py", "prompt_builder.py"))
        for source in response.rag_sources
    )
    assert "Do not invent classes/functions." in model.prompt
    assert "Return executable code only" not in model.prompt
    assert "Return the final answer as code first" not in model.prompt


def test_orchestrator_project_explain_supports_non_python_files():
    orchestrator = AgentOrchestrator()
    model = ExplanationModel()
    orchestrator.model_provider = model
    orchestrator.memory_writer = FakeMemory()
    orchestrator.rag_controller = RagControllerAgent(retriever=TypeScriptRetriever())

    response = orchestrator.run(
        GenerateRequest(
            task="project_explain",
            instruction="Explain this request handler",
            language="typescript",
            file_path="src/requestHandler.ts",
            project_path=".",
            use_rag=True,
        )
    )

    assert response.language == "typescript"
    assert response.used_rag is True
    assert response.rag_sources[0].language == "typescript"
    assert response.generated_code == ""
    assert "Language: TypeScript" in model.prompt
