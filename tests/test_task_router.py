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


def test_task_router_does_not_treat_source_listing_as_code_generation():
    router = TaskRouterAgent()
    assert router.detect("List the source files you used for this explanation") == "project_explain"
    assert router.detect("Which files did you use?") == "project_explain"
    assert router.detect("Show sources") == "project_explain"
