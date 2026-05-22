from __future__ import annotations

from backend.config.settings import settings
from backend.model.base import ModelProvider
from backend.model.finetuned_qwen import FinetunedQwenProvider
from backend.model.pretrained_qwen import PretrainedQwenProvider


def create_model_provider(provider_name: str | None = None) -> ModelProvider:
    provider = provider_name or settings.model_provider
    if provider == "pretrained_qwen":
        return PretrainedQwenProvider()
    if provider == "finetuned_qwen":
        return FinetunedQwenProvider()
    raise ValueError(f"Unsupported model provider: {provider}")
