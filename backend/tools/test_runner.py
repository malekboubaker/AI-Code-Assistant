from __future__ import annotations

import subprocess
from pathlib import Path


def run_tests(project_path: str | None) -> tuple[bool | None, list[str]]:
    if not project_path:
        return None, ["No project path supplied; skipped tests."]
    root = Path(project_path)
    if not root.exists():
        return None, [f"Project path does not exist: {project_path}"]
    candidates = [
        ["pytest", "-q"],
        ["python", "-m", "pytest", "-q"],
    ]
    for command in candidates:
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=60)
            return result.returncode == 0, [(result.stdout + result.stderr).strip()[-2000:]]
        except FileNotFoundError:
            continue
        except subprocess.SubprocessError as exc:
            return False, [str(exc)]
    return None, ["No supported local test runner found."]
