from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    payload = {
        "model": model,
        "prompt": "Return exactly: OK",
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "num_predict": 8,
            "temperature": 0.1,
            "top_p": 0.9,
        },
    }
    request = urllib.request.Request(
        f"{base_url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Failed to warm up Ollama model {model}: {exc}", file=sys.stderr)
        return 1

    text = str(body.get("response") or "").strip()
    print(f"Warmed up Ollama model {model} with keep_alive=30m. Response: {text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
