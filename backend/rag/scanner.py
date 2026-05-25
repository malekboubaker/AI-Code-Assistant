from __future__ import annotations

import os
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
    "cache",
    "caches",
    ".parcel-cache",
    ".next",
    ".nuxt",
    "coverage",
    "logs",
    "log",
    "qdrant_storage",
    "models",
}

EXCLUDED_SUFFIXES = {
    ".arrow",
    ".bin",
    ".ckpt",
    ".crt",
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
    ".key",
    ".log",
    ".mp4",
    ".onnx",
    ".parquet",
    ".pem",
    ".pdf",
    ".png",
    ".pt",
    ".pth",
    ".pyc",
    ".pyd",
    ".so",
    ".safetensors",
    ".sqlite",
    ".tar",
    ".tmp",
    ".zip",
}

SECRET_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}

MAX_FILE_BYTES = 1_000_000


@dataclass
class ScanStats:
    root: str
    files: list[Path] = field(default_factory=list)
    skipped_files: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=dict)
    skipped_folders: dict[str, int] = field(default_factory=dict)
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

    def skip_folder(self, folder: str) -> None:
        self.skipped_folders[folder] = self.skipped_folders.get(folder, 0) + 1


def scan_project(project_path: str, max_files: int | None = None) -> list[Path]:
    return scan_project_with_stats(project_path, max_files=max_files).files


def scan_project_with_stats(project_path: str, max_files: int | None = None) -> ScanStats:
    root = Path(project_path).resolve()
    stats = ScanStats(root=str(root))
    for current_root, dirnames, filenames in os.walk(root):
        current_path = Path(current_root)
        skipped_dirs = [name for name in dirnames if name in EXCLUDED_DIRS]
        for dirname in skipped_dirs:
            skipped_path = current_path / dirname
            stats.skip_folder(str(skipped_path.relative_to(root)).replace("\\", "/"))
        dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIRS]

        for filename in filenames:
            path = current_path / filename
            reason = skip_reason(path)
            if reason:
                stats.skip(reason)
                continue
            stats.add_file(path)
            if max_files and stats.files_scanned >= max_files:
                return stats
    return stats


def skip_reason(path: Path) -> str | None:
    if is_secret_path(path):
        return "secret"
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
    return suffix in EXTENSION_LANGUAGE


def is_secret_path(path: Path) -> bool:
    name = path.name.lower()
    if name in SECRET_FILE_NAMES or (name.startswith(".env") and name != ".env.example"):
        return True
    return path.suffix.lower() in {".pem", ".key", ".crt"}


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

