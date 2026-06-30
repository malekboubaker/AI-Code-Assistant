from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationOptions:
    max_tokens: int = 768
    temperature: float = 0.2
    top_p: float = 0.9
    response_id: str | None = None
    metadata: dict[str, object] | None = None


class ModelProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str, options: GenerationOptions | None = None) -> str:
        """Generate text from a local model runtime."""

    @abstractmethod
    def stream_generate(self, prompt: str, options: GenerationOptions | None = None):
        """Yield text chunks from a local model runtime."""
        pass
