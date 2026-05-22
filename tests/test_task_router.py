from backend.agent.task_router import TaskRouterAgent


def test_task_router_detects_bug_fix():
    assert TaskRouterAgent().detect("please fix this broken function") == "bug_fix"


def test_task_router_uses_explicit_task():
    assert TaskRouterAgent().detect("anything", "test_gen") == "test_gen"


def test_task_router_detects_explanation_intent():
    task = TaskRouterAgent().detect(
        "Explain how AgentOrchestrator coordinates TaskRouterAgent and PromptBuilderAgent"
    )
    assert task == "project_explain"
