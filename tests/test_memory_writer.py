from backend.agent.memory_writer import MemoryWriterAgent
from backend.api.schemas import ValidationResult


class FakeEmbedder:
    def embed(self, content):
        return [1.0]


class FakeStore:
    def __init__(self):
        self.rows = []

    def upsert(self, vector, payload):
        self.rows.append((vector, payload))


def test_memory_writer_skips_placeholder_assert_true():
    store = FakeStore()
    writer = MemoryWriterAgent(embedder=FakeEmbedder(), store=store)
    stored = writer.maybe_store(
        "def test_generated_behavior():\n    assert True\n",
        "python",
        "test_gen",
        ValidationResult(valid=True, syntax_valid=True),
        accepted=True,
    )
    assert stored is False
    assert store.rows == []


def test_memory_writer_requires_acceptance_or_tests():
    store = FakeStore()
    writer = MemoryWriterAgent(embedder=FakeEmbedder(), store=store)
    stored = writer.maybe_store(
        "def useful():\n    return 1\n",
        "python",
        "code_gen",
        ValidationResult(valid=True, syntax_valid=True),
    )
    assert stored is False
    assert store.rows == []
