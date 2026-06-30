from backend.rag.file_matching import (
    extract_file_references,
    extract_requested_entities,
    matches_entity,
    matches_file_reference,
)


def _kinds(text):
    return {(entity.kind, entity.name) for entity in extract_requested_entities(text)}


def test_extract_requested_entities_detects_files_classes_functions_folders():
    entities = _kinds("Compare test_agent.py and AgentOrchestrator and generate_embedding() in src/agent/")
    assert ("file", "test_agent.py") in entities
    assert ("symbol", "AgentOrchestrator") in entities
    assert ("symbol", "generate_embedding") in entities
    assert ("folder", "src/agent") in entities


def test_extract_requested_entities_ignores_plain_words():
    # Single capitalized words and ordinary prose must not become entities.
    assert _kinds("What is the difference between both files?") == set()
    assert _kinds("Explain this project") == set()


def test_matches_entity_symbol_and_folder():
    klass = {"symbol_name": "AgentOrchestrator", "relative_file_path": "backend/agent/orchestrator.py", "folder": "backend/agent"}
    method = {"symbol_name": "SomeClass", "content": "def generate_embedding(text):\n    return 1", "relative_file_path": "backend/rag/embedder.py"}
    [class_entity] = [e for e in extract_requested_entities("Explain AgentOrchestrator")]
    assert matches_entity(klass, class_entity) is True
    [fn_entity] = [e for e in extract_requested_entities("what does generate_embedding() do")]
    assert matches_entity(method, fn_entity) is True  # matched via definition in content
    [folder_entity] = [e for e in extract_requested_entities("look in backend/agent/")]
    assert matches_entity(klass, folder_entity) is True


def test_extract_file_references_finds_names_and_paths():
    assert extract_file_references("Explain a2a_client.py") == ["a2a_client.py"]
    assert extract_file_references("What does task_router.py do?") == ["task_router.py"]
    assert extract_file_references("Difference between a2a_client.py and a2a_server.py") == [
        "a2a_client.py",
        "a2a_server.py",
    ]
    assert extract_file_references("Look at backend/agent/orchestrator.py") == ["backend/agent/orchestrator.py"]


def test_extract_file_references_ignores_plain_text():
    assert extract_file_references("Explain the orchestrator") == []
    assert extract_file_references("What is the difference between both?") == []
    assert extract_file_references("e.g. something i.e. nothing") == []


def test_matches_file_reference_is_exact_and_avoids_similar_names():
    server = {"relative_file_path": "agents/a2a_server.py", "file_path": "C:/p/agents/a2a_server.py"}
    client = {"relative_file_path": "agents/a2a_client.py", "file_path": "C:/p/agents/a2a_client.py"}

    assert matches_file_reference(client, ["a2a_client.py"]) == "a2a_client.py"
    # A similar-named file must NOT match the wrong reference.
    assert matches_file_reference(server, ["a2a_client.py"]) is None
    # Path-qualified reference matches only as a real suffix.
    assert matches_file_reference(client, ["agents/a2a_client.py"]) == "agents/a2a_client.py"
    assert matches_file_reference(server, ["agents/a2a_client.py"]) is None
