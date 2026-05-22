from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from backend.tools.language_detector import EXTENSION_LANGUAGE, FILENAME_LANGUAGE, detect_language


EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "out",
    "target",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".cache",
    ".parcel-cache",
    ".next",
    ".nuxt",
    "coverage",
    "qdrant_storage",
    "models",
}

EXCLUDED_SUFFIXES = {
    ".arrow",
    ".bin",
    ".ckpt",
    ".db",
    ".dll",
    ".dylib",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".ipynb",
    ".jar",
    ".jpeg",
    ".jpg",
    ".log",
    ".mp4",
    ".onnx",
    ".parquet",
    ".pdf",
    ".png",
    ".pt",
    ".pyc",
    ".pyd",
    ".so",
    ".sqlite",
    ".tar",
    ".tmp",
    ".zip",
}

MAX_FILE_BYTES = 1_000_000


@dataclass
class ScanStats:
    root: str
    files: list[Path] = field(default_factory=list)
    skipped_files: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=dict)
    languages: dict[str, int] = field(default_factory=dict)

    @property
    def files_scanned(self) -> int:
        return len(self.files)

    def skip(self, reason: str) -> None:
        self.skipped_files += 1
        self.skipped_by_reason[reason] = self.skipped_by_reason.get(reason, 0) + 1

    def add_file(self, path: Path) -> None:
        self.files.append(path)
        language = detect_language(str(path))
        self.languages[language] = self.languages.get(language, 0) + 1


def scan_project(project_path: str, max_files: int | None = None) -> list[Path]:
    return scan_project_with_stats(project_path, max_files=max_files).files


def scan_project_with_stats(project_path: str, max_files: int | None = None) -> ScanStats:
    root = Path(project_path).resolve()
    stats = ScanStats(root=str(root))
    for path in root.rglob("*"):
        if _has_excluded_dir(path, root):
            if path.is_file():
                stats.skip("excluded_dir")
            continue
        if not path.is_file():
            continue
        reason = skip_reason(path)
        if reason:
            stats.skip(reason)
            continue
        stats.add_file(path)
        if max_files and stats.files_scanned >= max_files:
            break
    return stats


def skip_reason(path: Path) -> str | None:
    if not is_supported_path(path):
        return "unsupported_type"
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return "too_large"
    except OSError:
        return "stat_failed"
    if is_binary_file(path):
        return "binary"
    return None


def is_supported_path(path: Path) -> bool:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in EXCLUDED_SUFFIXES:
        return False
    if name in FILENAME_LANGUAGE:
        return True
    if name.startswith(".env"):
        return True
    return suffix in EXTENSION_LANGUAGE


def is_binary_file(path: Path, sample_size: int = 4096) -> bool:
    try:
        sample = path.read_bytes()[:sample_size]
    except OSError:
        return True
    if b"\0" in sample:
        return True
    if not sample:
        return False
    text_bytes = sum(1 for byte in sample if byte in {9, 10, 13} or 32 <= byte <= 126 or byte >= 128)
    return text_bytes / len(sample) < 0.75


def _has_excluded_dir(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part in EXCLUDED_DIRS for part in parts[:-1])
