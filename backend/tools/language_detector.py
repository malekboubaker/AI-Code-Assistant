from __future__ import annotations

from pathlib import Path


EXTENSION_LANGUAGE = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rs": "rust",
    ".md": "markdown",
    ".markdown": "markdown",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".json": "json",
    ".jsonc": "json",
    ".toml": "config",
    ".ini": "config",
    ".cfg": "config",
    ".conf": "config",
    ".env": "config",
    ".properties": "config",
    ".gradle": "config",
    ".xml": "config",
}

FILENAME_LANGUAGE = {
    "dockerfile": "dockerfile",
    "docker-compose.yml": "yaml",
    "docker-compose.yaml": "yaml",
    "compose.yml": "yaml",
    "compose.yaml": "yaml",
    "makefile": "config",
    "cmakelists.txt": "config",
    "package.json": "json",
    "tsconfig.json": "json",
    "pyproject.toml": "config",
    "cargo.toml": "config",
    "requirements.txt": "config",
    "readme": "markdown",
}

SUPPORTED_LANGUAGES = {
    "python",
    "java",
    "javascript",
    "typescript",
    "cpp",
    "csharp",
    "rust",
    "markdown",
    "yaml",
    "json",
    "dockerfile",
    "config",
}


def detect_language(file_path: str | None = None, code: str = "", fallback: str = "python") -> str:
    if file_path:
        path = Path(file_path)
        language = FILENAME_LANGUAGE.get(path.name.lower()) or EXTENSION_LANGUAGE.get(path.suffix.lower())
        if language:
            return language
    stripped = code.lstrip()
    if stripped.startswith("# ") or stripped.startswith("## "):
        return "markdown"
    if stripped.startswith("{") or stripped.startswith("["):
        return "json"
    if stripped.startswith("FROM ") or "\nRUN " in code:
        return "dockerfile"
    if "fn " in code and ("let " in code or "use " in code):
        return "rust"
    if stripped.startswith("using ") or "namespace " in stripped:
        return "csharp"
    if "#include" in code or "std::" in code:
        return "cpp"
    if "public class " in code:
        return "java"
    if "function " in code or "const " in code or "let " in code:
        return "javascript"
    return fallback
