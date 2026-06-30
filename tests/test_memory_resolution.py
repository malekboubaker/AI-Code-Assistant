import pytest
from backend.agent.conversation_memory import ConversationMemoryStore, ConversationTurn

def test_resolve_previous_file():
    store = ConversationMemoryStore()
    
    turn1 = ConversationTurn(
        user_intent="explain agent.py",
        task="file_explain",
        active_file="utils.py",
        files_referenced=["agent.py"]
    )
    store.add_turn("/project", turn1)
    
    resolved = store.resolve_references("/project", "explain the previous file")
    assert resolved == ["agent.py"], f"Expected ['agent.py'], got {resolved}"

def test_resolve_both():
    store = ConversationMemoryStore()
    
    turn1 = ConversationTurn(
        user_intent="explain agent.py",
        task="file_explain",
        active_file=None,
        files_referenced=["agent.py"]
    )
    store.add_turn("/project", turn1)
    
    turn2 = ConversationTurn(
        user_intent="how does planner.py work",
        task="file_explain",
        active_file=None,
        files_referenced=["planner.py"]
    )
    store.add_turn("/project", turn2)
    
    resolved = store.resolve_references("/project", "compare both")
    # Newest is planner.py, oldest is agent.py
    assert set(resolved) == {"planner.py", "agent.py"}, f"Expected both, got {resolved}"

def test_resolve_with_active_file():
    store = ConversationMemoryStore()
    
    turn1 = ConversationTurn(
        user_intent="how to use",
        task="project_explain",
        active_file="client.py",
        files_referenced=[]
    )
    store.add_turn("/project", turn1)
    
    resolved = store.resolve_references("/project", "explain the file")
    # "the file" should resolve to client.py because it was the last active file in history
    assert resolved == ["client.py"], f"Expected ['client.py'], got {resolved}"

def test_no_history_returns_empty():
    store = ConversationMemoryStore()
    resolved = store.resolve_references("/project", "compare both")
    assert resolved == [], "Expected empty resolution when no history exists."
