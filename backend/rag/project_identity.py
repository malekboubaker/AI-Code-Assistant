from __future__ import annotations

import hashlib
import os
from pathlib import Path


def normalize_project_path(project_path: str | Path) -> str:
    resolved = Path(project_path).expanduser().resolve()
    return os.path.normcase(str(resolved)).replace("\\", "/")


def project_id_for_path(project_path: str | Path) -> str:
    normalized = normalize_project_path(project_path)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def project_name_for_path(project_path: str | Path) -> str:
    return Path(project_path).expanduser().resolve().name or "workspace"
