from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request

from backend.config.settings import settings


class LocalEmbedder:
    def __init__(self, model: str | None = None, base_url: str | None = None, vector_size: int = 768) -> None:
        self.model = model or settings.ollama_embedding_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.vector_size = vector_size

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.model, "prompt": text}
        request = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
                vector = body.get("embedding")
                if isinstance(vector, list) and vector:
                    return _normalize([float(value) for value in vector])
        except (urllib.error.URLError, TimeoutError):
            pass
        return self._hash_embedding(text)

    def _hash_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.vector_size
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.vector_size
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _normalize(vector)


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
