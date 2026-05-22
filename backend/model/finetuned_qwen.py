from __future__ import annotations

from backend.model.base import GenerationOptions, ModelProvider


class FinetunedQwenProvider(ModelProvider):
    name = "finetuned_qwen"

    def __init__(self) -> None:
        self._loaded = False

    def generate(self, prompt: str, options: GenerationOptions | None = None) -> str:
        raise NotImplementedError(
            "Fine-tuned Qwen + LoRA provider placeholder. Plug the Azure adapter here when training finishes."
        )
