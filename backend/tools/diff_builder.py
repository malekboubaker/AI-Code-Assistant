from __future__ import annotations

import difflib


def build_unified_diff(original: str, generated: str, file_path: str | None = None) -> str:
    name = file_path or "selection"
    return "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            generated.splitlines(),
            fromfile=f"{name}:original",
            tofile=f"{name}:generated",
            lineterm="",
        )
    )
