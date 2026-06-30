from __future__ import annotations

import abc
import math

class Tokenizer(abc.ABC):
    """Abstract base class for modular tokenizers."""
    
    @abc.abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

class CharacterFallbackTokenizer(Tokenizer):
    """Fallback tokenizer using character approximation (1 token ~= 3.5 characters)."""
    
    def __init__(self, chars_per_token: float = 3.5):
        self.chars_per_token = chars_per_token

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(math.ceil(len(text) / self.chars_per_token)))

class TokenManager:
    """Manages dynamic token budget allocation for language models."""
    
    def __init__(self, tokenizer: Tokenizer | None = None):
        self.tokenizer = tokenizer or CharacterFallbackTokenizer()

    def count_tokens(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    def calculate_budget(self, prompt_tokens: int, context_window: int, safety_margin_pct: float = 0.15) -> dict[str, int]:
        """
        Calculates the available generation budget based on the context window.
        """
        safety_margin = int(context_window * safety_margin_pct)
        available = context_window - prompt_tokens - safety_margin
        
        # Enforce a minimum floor to avoid API errors (e.g., negative budget)
        budget = max(256, available)
        
        return {
            "prompt_tokens": prompt_tokens,
            "context_window": context_window,
            "safety_margin": safety_margin,
            "generation_budget": budget
        }

# Global singleton instance
token_manager = TokenManager()
