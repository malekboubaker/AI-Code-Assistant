from pathlib import Path

from backend.rag.chunker import chunk_file


def test_python_chunker_extracts_functions(tmp_path: Path):
    path = tmp_path / "sample.py"
    path.write_text("import os\n\ndef hello():\n    print(os.getcwd())\n    return 'hi'\n", encoding="utf-8")
    chunks = chunk_file(path)
    assert len(chunks) == 1
    assert chunks[0].payload["symbol_name"] == "hello"
    assert chunks[0].payload["language"] == "python"
    assert chunks[0].payload["called_functions"] == ["print", "getcwd"]
    assert chunks[0].payload["is_test_file"] is False


def test_typescript_chunker_extracts_function_metadata(tmp_path: Path):
    path = tmp_path / "requestHandler.ts"
    path.write_text(
        "import { logger } from './logger';\n\n"
        "export function requestHandler(req: Request) {\n"
        "  logger.info('handling request');\n"
        "  return fetch(req.url);\n"
        "}\n",
        encoding="utf-8",
    )

    chunks = chunk_file(path, project_root=tmp_path)

    assert len(chunks) == 1
    assert chunks[0].payload["symbol_name"] == "requestHandler"
    assert chunks[0].payload["relative_path"] == "requestHandler.ts"
    assert "fetch" in chunks[0].payload["called_functions"]
    assert chunks[0].payload["imports"]
